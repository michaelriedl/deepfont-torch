import logging
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from .config import DeepFontConfig, DeepFontAEConfig

# A logger for this file
logger = logging.getLogger(__name__)

# AlexNet-style LocalResponseNorm hyperparameters. The DeepFont paper does not
# state these explicitly but cites the AlexNet ImageNet structure (Krizhevsky
# 2012), which uses these values for its two LRN layers (after Conv1 and
# Conv2). PyTorch's LocalResponseNorm defaults match except for k, which we
# override to AlexNet's 2.0.
LRN_SIZE = 5
LRN_ALPHA = 1e-4
LRN_BETA = 0.75
LRN_K = 2.0


def _make_norm_layer(
    norm_type: Literal["none", "lrn", "batch"],
    num_channels: int,
) -> nn.Module | None:
    """Build the normalization layer for an encoder stage.

    Returns None when norm_type == "none" so the caller can skip appending.
    """
    if norm_type == "none":
        return None
    if norm_type == "lrn":
        return nn.LocalResponseNorm(size=LRN_SIZE, alpha=LRN_ALPHA, beta=LRN_BETA, k=LRN_K)
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    raise ValueError(f"Unknown norm_type '{norm_type}'. Expected 'none', 'lrn', or 'batch'.")


class TiedConvTranspose2d(nn.Module):
    """Transposed convolution whose weight is tied to a paired encoder Conv2d.

    The decoder layer reuses the weight tensor of an encoder Conv2d via
    F.conv_transpose2d at forward time, constraining the encoder filter and
    decoder filter to be transposes of each other. This follows the original
    Masci 2011 stacked convolutional auto-encoder formulation cited by the
    DeepFont paper.

    A Conv2d weight has shape (out_channels, in_channels, kH, kW), which is
    exactly the layout F.conv_transpose2d expects when going in the reverse
    direction (in_channels and out_channels swap roles for the transpose), so
    no reshaping is needed.

    The decoder layer still owns its own bias parameter; only the weight is
    shared. Gradients flow back into the encoder Conv2d's weight from both
    the encoder and decoder paths during backward, which keeps the tying
    consistent through training.

    Attributes:
        encoder_conv: The encoder Conv2d whose weight is reused. Stored as a
            plain attribute (not a child module) to avoid the encoder weight
            being registered twice in the decoder's state_dict.
        stride: Stride for the transposed convolution.
        padding: Padding for the transposed convolution.
        bias: Per-output-channel bias parameter owned by this layer.
    """

    encoder_conv: nn.Conv2d

    def __init__(self, encoder_conv: nn.Conv2d, stride: int, padding: int):
        """Initialize a tied transposed convolution paired with an encoder Conv2d.

        Args:
            encoder_conv: The encoder Conv2d whose weight tensor will be reused
                as the transposed-conv filter. Its weight shape determines the
                input and output channel counts of this layer (in_channels and
                out_channels swap relative to the encoder).
            stride: Stride for the transposed convolution. Should match the
                paired encoder Conv2d's stride.
            padding: Padding for the transposed convolution. Should match the
                paired encoder Conv2d's padding.
        """
        super().__init__()
        # Avoid registering the encoder Conv2d as a child module so its weight
        # is not duplicated in this layer's state_dict, but keep a live
        # reference so weight updates remain visible at forward time.
        object.__setattr__(self, "encoder_conv", encoder_conv)
        self.stride = stride
        self.padding = padding
        # ConvTranspose2d output channels equal the paired Conv2d's
        # in_channels because the channel direction flips in the transpose.
        self.bias = nn.Parameter(torch.zeros(encoder_conv.in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the tied transposed convolution to x."""
        return F.conv_transpose2d(
            x,
            self.encoder_conv.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def extra_repr(self) -> str:
        w = self.encoder_conv.weight
        out_ch, in_ch, kh, kw = w.shape
        return (
            f"in_channels={out_ch}, out_channels={in_ch}, "
            f"kernel_size=({kh}, {kw}), stride={self.stride}, "
            f"padding={self.padding}, tied=True"
        )


def _build_encoder(
    in_channels: int,
    channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    strides: tuple[int, ...],
    paddings: tuple[int, ...],
    pool_kernel_size: int,
    norm_type: Literal["none", "lrn", "batch"],
) -> nn.Sequential:
    """Build a multi-stage convolutional encoder.

    Each stage consists of Conv2d -> [Norm] -> MaxPool2d -> ReLU, where the
    Norm layer is LocalResponseNorm, BatchNorm2d, or omitted depending on
    norm_type. The 'lrn' option matches the paper's Fig. 5 architecture.

    Args:
        in_channels: Number of channels in the input image.
        channels: Output channel count for each convolutional stage.
        kernel_sizes: Kernel size for each stage's Conv2d.
        strides: Stride for each stage's Conv2d.
        paddings: Padding for each stage's Conv2d.
        pool_kernel_size: Kernel size for MaxPool2d after each stage.
        norm_type: Normalization layer type. One of 'none', 'lrn', 'batch'.

    Returns:
        An nn.Sequential module implementing the encoder.
    """
    layers: list[nn.Module] = []
    prev_ch = in_channels
    for ch, k, s, p in zip(channels, kernel_sizes, strides, paddings, strict=True):
        layers.append(nn.Conv2d(prev_ch, ch, kernel_size=k, stride=s, padding=p))
        norm_layer = _make_norm_layer(norm_type, ch)
        if norm_layer is not None:
            layers.append(norm_layer)
        layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
        layers.append(nn.ReLU())
        prev_ch = ch
    return nn.Sequential(*layers)


def _encoder_conv_layers(encoder: nn.Sequential) -> list[nn.Conv2d]:
    """Return the Conv2d layers in encoder order."""
    return [m for m in encoder if isinstance(m, nn.Conv2d)]


def _build_decoder(
    out_channels: int,
    encoder_channels: tuple[int, ...],
    encoder_kernel_sizes: tuple[int, ...],
    encoder_strides: tuple[int, ...],
    encoder_paddings: tuple[int, ...],
    pool_kernel_size: int,
    output_activation: str | None,
    encoder_convs: list[nn.Conv2d] | None = None,
) -> nn.Sequential:
    """Build a decoder that mirrors the encoder structure.

    For each encoder stage (processed in reverse), the decoder applies
    Upsample -> ConvTranspose2d -> ReLU, except the final layer which
    omits the ReLU (or replaces it with the requested output_activation).

    The transposed convolution at each stage uses the same kernel size,
    stride, and padding as its corresponding encoder Conv2d, which ensures
    the transpose inverts the spatial transform of the forward convolution.

    When encoder_convs is provided, each decoder stage is built with a
    TiedConvTranspose2d that reuses the paired encoder Conv2d's weight.
    Otherwise an independent ConvTranspose2d is used at each stage.

    Args:
        out_channels: Number of channels the decoder should produce (typically
            equals the encoder's in_channels).
        encoder_channels: Channel counts from the encoder (in encoder order).
        encoder_kernel_sizes: Kernel sizes from the encoder (in encoder order).
        encoder_strides: Strides from the encoder (in encoder order).
        encoder_paddings: Paddings from the encoder (in encoder order).
        pool_kernel_size: Pool kernel size used in the encoder.
        output_activation: Optional final activation ("sigmoid" or "relu").
        encoder_convs: When not None, the encoder's Conv2d layers in encoder
            order. Each decoder stage will tie its weight to the matching
            encoder Conv2d via TiedConvTranspose2d.

    Returns:
        An nn.Sequential module implementing the decoder.
    """
    layers: list[nn.Module] = []
    n_stages = len(encoder_channels)
    reversed_channels = list(reversed(encoder_channels))
    reversed_kernel_sizes = list(reversed(encoder_kernel_sizes))
    reversed_strides = list(reversed(encoder_strides))
    reversed_paddings = list(reversed(encoder_paddings))
    if encoder_convs is not None:
        if len(encoder_convs) != n_stages:
            raise ValueError(
                f"encoder_convs has {len(encoder_convs)} layers but encoder has {n_stages} stages."
            )
        reversed_convs = list(reversed(encoder_convs))
    else:
        reversed_convs = None

    for i in range(n_stages):
        in_ch = reversed_channels[i]
        # Output channel: next reversed channel, or out_channels for the last stage
        target_ch = reversed_channels[i + 1] if i < n_stages - 1 else out_channels
        k = reversed_kernel_sizes[i]
        s = reversed_strides[i]
        p = reversed_paddings[i]

        # Upsample to undo the MaxPool2d
        layers.append(nn.Upsample(scale_factor=pool_kernel_size))
        # Transposed conv (tied or independent) to undo the Conv2d
        if reversed_convs is not None:
            layers.append(TiedConvTranspose2d(reversed_convs[i], stride=s, padding=p))
        else:
            layers.append(nn.ConvTranspose2d(in_ch, target_ch, kernel_size=k, stride=s, padding=p))

        # Add activation (ReLU for intermediate layers, optional for last)
        is_last = i == n_stages - 1
        if not is_last:
            layers.append(nn.ReLU())

    # Optional output activation
    if output_activation is not None:
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif output_activation == "relu":
            layers.append(nn.ReLU())
        else:
            raise ValueError(f"Unknown output activation function: {output_activation}")

    return nn.Sequential(*layers)


class DeepFontAE(nn.Module):
    """Convolutional autoencoder for unsupervised font representation learning.

    This autoencoder implements the DeepFont architecture's pretraining stage,
    designed to learn robust feature representations from font images through
    reconstruction. The encoder compresses input images into a latent representation,
    while the decoder reconstructs the original image from this compressed form.

    The architecture uses:
        - Encoder: Convolutional layers with max pooling for feature extraction
        - Decoder: Transposed convolutions and upsampling for reconstruction

    This model is typically pretrained on a large dataset of font images before
    the encoder weights are transferred to the DeepFont classifier for fine-tuning.

    All architectural hyper-parameters (channel counts, kernel sizes, strides,
    etc.) are controlled via a DeepFontAEConfig instance, whose defaults
    reproduce the original paper architecture.

    Attributes:
        config: The frozen configuration used to build this model.
        encoder: Sequential module containing convolutional and pooling layers that
            compress the input image into a latent representation.
        decoder: Sequential module containing transposed convolutions that reconstruct
            the image from the latent representation.
    """

    def __init__(self, config: DeepFontAEConfig | None = None):
        """Initializes the DeepFontAE autoencoder architecture.

        Constructs the encoder-decoder network from the provided configuration.
        The encoder uses standard convolutions with ReLU activations and max pooling,
        while the decoder mirrors the encoder using transposed convolutions and
        upsampling.

        Args:
            config: A DeepFontAEConfig controlling every architectural
                parameter.  When None, a default config is used.

        Raises:
            ValueError: If config validation fails (e.g. mismatched tuple
                lengths, invalid channel counts).

        Note:
            The choice of output activation should match your input normalization:
            - [0, 1] normalization -> use "sigmoid"
            - [-1, 1] normalization -> use None (or tanh, though not implemented)
            - [0, 255] normalization -> use None or "relu"
        """
        super().__init__()
        if config is None:
            config = DeepFontAEConfig()
        self.config = config

        self.encoder = _build_encoder(
            in_channels=config.in_channels,
            channels=config.encoder_channels,
            kernel_sizes=config.encoder_kernel_sizes,
            strides=config.encoder_strides,
            paddings=config.encoder_paddings,
            pool_kernel_size=config.pool_kernel_size,
            norm_type="none",
        )
        encoder_convs = _encoder_conv_layers(self.encoder) if config.tied_weights else None
        self.decoder = _build_decoder(
            out_channels=config.in_channels,
            encoder_channels=config.encoder_channels,
            encoder_kernel_sizes=config.encoder_kernel_sizes,
            encoder_strides=config.encoder_strides,
            encoder_paddings=config.encoder_paddings,
            pool_kernel_size=config.pool_kernel_size,
            output_activation=config.output_activation,
            encoder_convs=encoder_convs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the autoencoder.

        Encodes the input image into a latent representation and then decodes it
        back to reconstruct the original image. The reconstruction is used to compute
        a loss (typically MSE or L1) during training.

        Args:
            x: Input image tensor of shape (batch_size, in_channels, H, W).
                For the paper defaults, shape is (batch_size, 1, 105, 105).

        Returns:
            Reconstructed image tensor.  The spatial dimensions may differ
            slightly from the input when non-default encoder parameters are
            used; with default settings the output shape equals the input shape.

        Note:
            The actual input size should be 105x105 based on the DeepFont paper.
        """
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class DeepFont(nn.Module):
    """Deep convolutional neural network for font classification.

    This model implements the full DeepFont architecture for supervised font
    recognition. It consists of an encoder (which can be pretrained using DeepFontAE),
    additional convolutional layers for feature refinement, and fully connected layers
    for classification.

    The architecture follows:
        1. Encoder: conv layers with optional batch norm, pooling, and ReLU
           (can use pretrained weights)
        2. Convolutional part: additional conv layers with batch norm and ReLU
           for deeper features
        3. Fully connected part: FC layers with dropout for classification

    All architectural hyper-parameters are controlled via a DeepFontConfig
    instance, whose defaults reproduce the original paper architecture.

    This model supports transfer learning by loading pretrained encoder weights from
    the autoencoder pretraining stage, which typically improves convergence and final
    accuracy compared to training from scratch.

    Attributes:
        config: The frozen configuration used to build this model.
        encoder: Convolutional encoder layers, optionally loaded from pretrained autoencoder.
        conv_part: Additional convolutional layers for feature extraction.
        fc_part: Fully connected layers for final classification.
    """

    def __init__(self, config: DeepFontConfig | None = None):
        """Initializes the DeepFont classification model.

        Constructs the full architecture including encoder, convolutional layers,
        and fully connected classification head. The encoder portion can later be
        initialized with pretrained weights using load_encoder_weights().

        Args:
            config: A DeepFontConfig controlling every architectural
                parameter.  When None, a default config is used.

        Raises:
            ValueError: If config validation fails (e.g. spatial dimensions
                reduced to zero, mismatched tuple lengths).

        Note:
            The model expects square input images whose size matches
            config.input_size (default 105).
        """
        super().__init__()
        if config is None:
            config = DeepFontConfig()
        self.config = config

        # Encoder
        self.encoder = _build_encoder(
            in_channels=config.in_channels,
            channels=config.encoder_channels,
            kernel_sizes=config.encoder_kernel_sizes,
            strides=config.encoder_strides,
            paddings=config.encoder_paddings,
            pool_kernel_size=config.pool_kernel_size,
            norm_type=config.encoder_norm_type,
        )

        # Additional conv layers
        conv_layers: list[nn.Module] = []
        # Paper Fig. 5 shows Conv3/4/5 with no normalization layer between
        # them, so conv_part is fixed as Conv2d -> ReLU per stage.
        prev_ch = config.encoder_channels[-1]
        for _ in range(config.num_conv_layers):
            conv_layers.append(
                nn.Conv2d(
                    prev_ch,
                    config.conv_channels,
                    kernel_size=config.conv_kernel_size,
                    padding="same",
                )
            )
            conv_layers.append(nn.ReLU())
            prev_ch = config.conv_channels
        self.conv_part = nn.Sequential(*conv_layers)

        # Fully-connected head
        # Compute the spatial size after the encoder
        spatial = config.input_size
        for k, s, p in zip(
            config.encoder_kernel_sizes,
            config.encoder_strides,
            config.encoder_paddings,
            strict=True,
        ):
            spatial = (spatial - k + 2 * p) // s + 1
            spatial = spatial // config.pool_kernel_size

        flatten_dim = spatial * spatial * config.conv_channels

        fc_layers: list[nn.Module] = [nn.Flatten()]
        prev_dim = flatten_dim
        for hidden_dim in config.fc_hidden_dims:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(nn.Dropout(config.dropout_rate))
            fc_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        fc_layers.append(nn.Linear(prev_dim, config.num_classes))
        self.fc_part = nn.Sequential(*fc_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the classification network.

        Processes the input image through the encoder, convolutional layers, and
        fully connected layers to produce class logits for font classification.

        Args:
            x: Input image tensor of shape (batch_size, in_channels, H, W)
                where H = W = config.input_size (default 105).

        Returns:
            Class logits tensor of shape (batch_size, num_classes) containing
            raw scores for each font class. Apply softmax to get probabilities,
            or use with CrossEntropyLoss which applies softmax internally.

        Note:
            The output logits are not normalized. Use torch.nn.functional.softmax
            for probability distributions, or pass directly to CrossEntropyLoss
            during training.
        """
        x = self.encoder(x)
        x = self.conv_part(x)
        x = self.fc_part(x)

        return x

    def load_encoder_weights(self, encoder_weights_file: str):
        """Loads pretrained encoder weights from an autoencoder checkpoint.

        This method enables transfer learning by initializing the encoder with weights
        learned during autoencoder pretraining. It extracts encoder weights from the
        checkpoint, maps them to the correct layer names (accounting for batch norm
        layers in the classifier), and freezes the loaded layers to preserve the
        pretrained features.

        The weight mapping handles the structural differences between DeepFontAE
        (no norm layers) and DeepFont (optionally with norm layers) by computing
        the correct index offsets based on the classifier's encoder_norm_type.

        Args:
            encoder_weights_file: Path to the saved autoencoder model checkpoint (.pt or
                .pth file). The checkpoint should contain a state_dict with encoder weights
                saved from a DeepFontAE model.

        Raises:
            FileNotFoundError: If the encoder_weights_file doesn't exist.
            RuntimeError: If the weight shapes don't match or keys are missing.

        Note:
            This method only loads the convolutional weights and biases, not
            batch-normalization parameters (which may not exist in the autoencoder).
            The loaded layers are automatically frozen (requires_grad=False)
            to prevent their modification during fine-tuning.  Each frozen layer
            is logged for verification.

        Example:
            >>> model = DeepFont(DeepFontConfig(num_classes=2383))
            >>> model.load_encoder_weights('pretrained_ae.pt')
            >>> # Now train with frozen encoder weights
        """
        logger.info("Loading encoder weights from: %s", encoder_weights_file)

        # Load the weights; unwrap Fabric/Lightning checkpoints which store the
        # model state dict under a "model" key alongside optimizer, epoch, etc.
        state_dict = torch.load(encoder_weights_file, map_location=torch.device("cpu"))
        if "model" in state_dict and isinstance(state_dict["model"], dict):
            logger.info("Detected Fabric checkpoint — unwrapping 'model' key.")
            state_dict = state_dict["model"]

        # Keep only the encoder part
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if "encoder" in k}
        if not state_dict:
            raise RuntimeError(
                f"No encoder weights found in '{encoder_weights_file}'. "
                "Expected keys containing 'encoder.' in the state dict."
            )
        logger.info("Found %d encoder tensor(s) in checkpoint.", len(state_dict))

        # Compute the layer index mapping between the source AE encoder and this
        # classifier's encoder.  Each encoder stage has a variable number of
        # sub-layers depending on the chosen norm:
        #   none : Conv2d, MaxPool2d, ReLU                       -> 3 sub-layers per stage
        #   lrn  : Conv2d, LocalResponseNorm, MaxPool2d, ReLU    -> 4 sub-layers per stage
        #   batch: Conv2d, BatchNorm2d, MaxPool2d, ReLU          -> 4 sub-layers per stage
        #
        # We need to detect the source layout from the checkpoint keys and map
        # Conv2d weight/bias keys to the correct indices in this encoder.
        src_conv_indices = sorted({int(k.split(".")[0]) for k in state_dict})
        dst_stride = 3 if self.config.encoder_norm_type == "none" else 4

        layer_map: dict[str, str] = {}
        for stage_i, src_idx in enumerate(src_conv_indices):
            dst_idx = stage_i * dst_stride
            for suffix in ("weight", "bias"):
                src_key = f"{src_idx}.{suffix}"
                if src_key in state_dict:
                    layer_map[src_key] = f"{dst_idx}.{suffix}"

        if not layer_map:
            raise RuntimeError(
                "Could not map any encoder keys from the checkpoint to the model encoder. "
                f"Checkpoint encoder keys: {list(state_dict.keys())}"
            )

        new_state_dict = {}
        for src_key, dst_key in layer_map.items():
            src_shape = state_dict[src_key].shape
            dst_param = dict(self.encoder.named_parameters()).get(dst_key)
            if dst_param is None:
                raise RuntimeError(
                    f"Destination key '{dst_key}' not found in encoder. "
                    f"Cannot map checkpoint key '{src_key}'."
                )
            if src_shape != dst_param.shape:
                raise RuntimeError(
                    f"Shape mismatch for '{src_key}' → '{dst_key}': "
                    f"checkpoint has {src_shape}, model expects {dst_param.shape}."
                )
            new_state_dict[dst_key] = state_dict[src_key]
            logger.info("  Matched: checkpoint '%s' → encoder '%s' %s", src_key, dst_key, src_shape)

        # Load the weights
        self.encoder.load_state_dict(new_state_dict, strict=False)
        logger.info("Successfully loaded %d encoder weight tensor(s).", len(new_state_dict))

        # Freeze the loaded layers
        frozen = []
        for param_name, param in self.encoder.named_parameters():
            if param_name in new_state_dict:
                param.requires_grad_(False)
                frozen.append(param_name)
        logger.info("Frozen %d encoder parameter(s): %s", len(frozen), frozen)
