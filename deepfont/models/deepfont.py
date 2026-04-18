import logging

import torch
import torch.nn as nn

from .config import DeepFontConfig, DeepFontAEConfig

# A logger for this file
logger = logging.getLogger(__name__)


def _build_encoder(
    in_channels: int,
    channels: tuple[int, ...],
    kernel_sizes: tuple[int, ...],
    strides: tuple[int, ...],
    paddings: tuple[int, ...],
    pool_kernel_size: int,
    use_batch_norm: bool,
) -> nn.Sequential:
    """Build a multi-stage convolutional encoder.

    Each stage consists of Conv2d -> [BatchNorm2d] -> MaxPool2d -> ReLU.

    Args:
        in_channels: Number of channels in the input image.
        channels: Output channel count for each convolutional stage.
        kernel_sizes: Kernel size for each stage's Conv2d.
        strides: Stride for each stage's Conv2d.
        paddings: Padding for each stage's Conv2d.
        pool_kernel_size: Kernel size for MaxPool2d after each stage.
        use_batch_norm: Whether to include BatchNorm2d after each Conv2d.

    Returns:
        An nn.Sequential module implementing the encoder.
    """
    layers: list[nn.Module] = []
    prev_ch = in_channels
    for ch, k, s, p in zip(channels, kernel_sizes, strides, paddings, strict=True):
        layers.append(nn.Conv2d(prev_ch, ch, kernel_size=k, stride=s, padding=p))
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(ch))
        layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
        layers.append(nn.ReLU())
        prev_ch = ch
    return nn.Sequential(*layers)


def _build_decoder(
    out_channels: int,
    encoder_channels: tuple[int, ...],
    encoder_kernel_sizes: tuple[int, ...],
    encoder_strides: tuple[int, ...],
    encoder_paddings: tuple[int, ...],
    pool_kernel_size: int,
    output_activation: str | None,
) -> nn.Sequential:
    """Build a decoder that mirrors the encoder structure.

    For each encoder stage (processed in reverse), the decoder applies
    Upsample -> ConvTranspose2d -> ReLU, except the final layer which
    omits the ReLU (or replaces it with the requested output_activation).

    The ConvTranspose2d at each stage uses the same kernel size, stride,
    and padding as its corresponding encoder Conv2d, which ensures the
    transpose convolution inverts the spatial transform of the forward
    convolution.

    Args:
        out_channels: Number of channels the decoder should produce (typically
            equals the encoder's in_channels).
        encoder_channels: Channel counts from the encoder (in encoder order).
        encoder_kernel_sizes: Kernel sizes from the encoder (in encoder order).
        encoder_strides: Strides from the encoder (in encoder order).
        encoder_paddings: Paddings from the encoder (in encoder order).
        pool_kernel_size: Pool kernel size used in the encoder.
        output_activation: Optional final activation ("sigmoid" or "relu").

    Returns:
        An nn.Sequential module implementing the decoder.
    """
    layers: list[nn.Module] = []
    n_stages = len(encoder_channels)
    reversed_channels = list(reversed(encoder_channels))
    reversed_kernel_sizes = list(reversed(encoder_kernel_sizes))
    reversed_strides = list(reversed(encoder_strides))
    reversed_paddings = list(reversed(encoder_paddings))

    for i in range(n_stages):
        in_ch = reversed_channels[i]
        # Output channel: next reversed channel, or out_channels for the last stage
        target_ch = reversed_channels[i + 1] if i < n_stages - 1 else out_channels
        k = reversed_kernel_sizes[i]
        s = reversed_strides[i]
        p = reversed_paddings[i]

        # Upsample to undo the MaxPool2d
        layers.append(nn.Upsample(scale_factor=pool_kernel_size))
        # ConvTranspose2d to undo the Conv2d
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
            use_batch_norm=config.use_batch_norm,
        )
        self.decoder = _build_decoder(
            out_channels=config.in_channels,
            encoder_channels=config.encoder_channels,
            encoder_kernel_sizes=config.encoder_kernel_sizes,
            encoder_strides=config.encoder_strides,
            encoder_paddings=config.encoder_paddings,
            pool_kernel_size=config.pool_kernel_size,
            output_activation=config.output_activation,
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
            config.input_size (default 105).  The encoder includes batch
            normalization layers by default, unlike the autoencoder version.
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
            use_batch_norm=config.use_encoder_batch_norm,
        )

        # Additional conv layers
        conv_layers: list[nn.Module] = []
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
            if config.use_conv_batch_norm:
                conv_layers.append(nn.BatchNorm2d(config.conv_channels))
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
        (with or without batch norm) and DeepFont (with or without batch norm) by
        computing the correct index offsets based on whether batch norm is present.

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
        # sub-layers depending on whether batch norm is used:
        #   without BN: Conv2d, MaxPool2d, ReLU  -> 3 sub-layers per stage
        #   with BN:    Conv2d, BatchNorm2d, MaxPool2d, ReLU  -> 4 sub-layers per stage
        #
        # We need to detect the source layout from the checkpoint keys and map
        # Conv2d weight/bias keys to the correct indices in this encoder.
        src_conv_indices = sorted({int(k.split(".")[0]) for k in state_dict})
        dst_stride = 4 if self.config.use_encoder_batch_norm else 3

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
