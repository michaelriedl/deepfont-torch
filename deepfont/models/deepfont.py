import logging

import torch
import torch.nn as nn

# A logger for this file
logger = logging.getLogger(__name__)


class DeepFontAE(nn.Module):
    """Convolutional autoencoder for unsupervised font representation learning.

    This autoencoder implements the DeepFont architecture's pretraining stage,
    designed to learn robust feature representations from font images through
    reconstruction. The encoder compresses input images into a latent representation,
    while the decoder reconstructs the original image from this compressed form.

    The architecture uses:
        - Encoder: Two convolutional layers with max pooling for feature extraction
        - Decoder: Transposed convolutions and upsampling for reconstruction

    This model is typically pretrained on a large dataset of font images before
    the encoder weights are transferred to the DeepFont classifier for fine-tuning.

    Attributes:
        encoder: Sequential module containing convolutional and pooling layers that
            compress the input image into a latent representation.
        decoder: Sequential module containing transposed convolutions that reconstruct
            the image from the latent representation.
    """

    def __init__(self, output_activation: str = None):
        """Initializes the DeepFontAE autoencoder architecture.

        Constructs the encoder-decoder network with configurable output activation.
        The encoder uses standard convolutions with ReLU activations and max pooling,
        while the decoder uses transposed convolutions for upsampling.

        Args:
            output_activation: Optional activation function to apply to the decoder's
                final output. Options are:
                - None: No activation (linear output)
                - "sigmoid": Sigmoid activation, useful when input is normalized to [0, 1]
                - "relu": ReLU activation, useful when input is normalized to [0, ∞)
                Default is None.

        Raises:
            ValueError: If output_activation is not None, "sigmoid", or "relu".

        Note:
            The choice of output activation should match your input normalization:
            - [0, 1] normalization → use "sigmoid"
            - [-1, 1] normalization → use None (or tanh, though not implemented)
            - [0, 255] normalization → use None or "relu"
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=2),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(64, 1, kernel_size=11, stride=2),
        )
        if output_activation is not None:
            if output_activation == "sigmoid":
                self.decoder.add_module("sigmoid", nn.Sigmoid())
            elif output_activation == "relu":
                self.decoder.add_module("relu", nn.ReLU())
            else:
                raise ValueError(f"Unknown output activation function: {output_activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the autoencoder.

        Encodes the input image into a latent representation and then decodes it
        back to reconstruct the original image. The reconstruction is used to compute
        a loss (typically MSE or L1) during training.

        Args:
            x: Input image tensor of shape (batch_size, 1, H, W) where H and W are
                the height and width. Typically 105x105 for DeepFont, though the
                docstring mentions 96x96.

        Returns:
            Reconstructed image tensor of the same shape as input (batch_size, 1, H, W).
            The output values depend on the configured output_activation.

        Note:
            The actual input size should be 105x105 based on the DeepFont paper,
            not 96x96 as mentioned in the original docstring.
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
        1. Encoder: 2 conv layers with batch norm, pooling, and ReLU (can use pretrained weights)
        2. Convolutional part: 3 conv layers with batch norm and ReLU for deeper features
        3. Fully connected part: 3 FC layers with dropout for classification

    This model supports transfer learning by loading pretrained encoder weights from
    the autoencoder pretraining stage, which typically improves convergence and final
    accuracy compared to training from scratch.

    Attributes:
        num_out: Number of output classes (fonts) to classify.
        encoder: Convolutional encoder layers, optionally loaded from pretrained autoencoder.
        conv_part: Additional convolutional layers for feature extraction.
        fc_part: Fully connected layers for final classification.
    """

    def __init__(self, num_out: int):
        """Initializes the DeepFont classification model.

        Constructs the full architecture including encoder, convolutional layers,
        and fully connected classification head. The encoder portion can later be
        initialized with pretrained weights using load_encoder_weights().

        Args:
            num_out: The number of font classes to classify. This determines the
                dimension of the final output layer. Should match the number of unique
                fonts in your dataset.

        Note:
            The model expects input images of shape (batch_size, 1, 105, 105).
            The encoder includes batch normalization layers, unlike the autoencoder
            version, to improve training stability during supervised learning.
        """
        super().__init__()
        # Store the number of output classes
        self.num_out = num_out
        # Create the encoder part
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )
        # Create the convolutional part
        self.conv_part = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        # Create the fully connected part
        self.fc_part = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 12 * 256, 4096),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(4096, num_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the classification network.

        Processes the input image through the encoder, convolutional layers, and
        fully connected layers to produce class logits for font classification.

        Args:
            x: Input image tensor of shape (batch_size, 1, H, W) where H and W are
                typically 105x105 for DeepFont (not 96x96 as originally documented).

        Returns:
            Class logits tensor of shape (batch_size, num_out) containing raw scores
            for each font class. Apply softmax to get probabilities, or use with
            CrossEntropyLoss which applies softmax internally.

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
        (no batch norm) and DeepFont (with batch norm) by manually mapping layer indices.

        Args:
            encoder_weights_file: Path to the saved autoencoder model checkpoint (.pt or
                .pth file). The checkpoint should contain a state_dict with encoder weights
                saved from a DeepFontAE model.

        Raises:
            FileNotFoundError: If the encoder_weights_file doesn't exist.
            RuntimeError: If the weight shapes don't match or keys are missing.

        Note:
            This method only loads the convolutional and bias weights, not the batch
            normalization layers (which don't exist in the autoencoder). The loaded
            layers are automatically frozen (requires_grad=False) to prevent their
            modification during fine-tuning. Each frozen layer is logged for verification.

        Example:
            >>> model = DeepFont(num_out=2383)
            >>> model.load_encoder_weights('pretrained_ae.pt')
            >>> # Now train with frozen encoder weights
        """
        # Load the weights
        state_dict = torch.load(encoder_weights_file, map_location=torch.device("cpu"))
        # Keep only the encoder part
        state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items() if "encoder" in k}
        # Manually map the weights
        layer_map = {
            "0.weight": "0.weight",
            "0.bias": "0.bias",
            "3.weight": "4.weight",
            "3.bias": "4.bias",
        }
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[layer_map[k]] = v
        # Load the weights
        self.encoder.load_state_dict(new_state_dict, strict=False)
        # Freeze the loaded layers
        for param_name, param in self.encoder.named_parameters():
            if param_name in new_state_dict:
                param.requires_grad_(False)
                # Log the frozen layer
                logger.info(f"Freezing layer: {param_name}")
