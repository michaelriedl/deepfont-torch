"""Pydantic configuration classes for DeepFont model architectures.

These configuration classes parameterize the DeepFontAE autoencoder and
DeepFont classifier architectures, exposing all key architectural
hyper-parameters that are otherwise hard-coded in the model definitions.

The default values reproduce the original DeepFont architecture described in:

    Wang et al., "DeepFont: Identify Your Font from An Image",
    ACM Multimedia 2015.  (arXiv:1507.03196)

Architecture summary (paper defaults):

- Input: 1-channel (grayscale) 105x105 patches.
- Encoder (shared by both models): two convolutional stages, each consisting
  of Conv2d -> [BatchNorm2d] -> MaxPool2d -> ReLU.  Stage 1 uses 64 filters
  with an 11x11 kernel at stride 2 (AlexNet-style); stage 2 uses 128 filters
  with a 5x5 kernel at stride 1 with padding 2.
- Decoder (autoencoder only): mirrors the encoder using
  Upsample -> ConvTranspose2d -> ReLU in reverse order.
- Conv part (classifier only): three additional 3x3 conv layers with 256
  filters each, using same padding and batch normalization.
- FC part (classifier only): two hidden layers of 4096 units with dropout
  (rate 0.1) followed by the classification head.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, BaseModel, ConfigDict, field_validator, model_validator


class DeepFontAEConfig(BaseModel):
    """Configuration for the DeepFontAE autoencoder.

    Controls the encoder/decoder architecture and output activation. All tuple
    parameters that describe the encoder layers must have the same length;
    the number of elements equals the number of convolutional stages in the
    encoder (default: 2).

    Example:
        >>> from deepfont.models.config import DeepFontAEConfig
        >>> from deepfont.models.deepfont import DeepFontAE
        >>> # Paper defaults
        >>> config = DeepFontAEConfig()
        >>> model = DeepFontAE(config)
        >>> # Wider first layer, batch-normalized encoder, sigmoid output
        >>> config = DeepFontAEConfig(
        ...     encoder_channels=(96, 192),
        ...     use_batch_norm=True,
        ...     output_activation="sigmoid",
        ... )
        >>> model = DeepFontAE(config)
    """

    model_config = ConfigDict(frozen=True)

    # Input
    in_channels: int = Field(
        default=1,
        ge=1,
        le=4,
        description=(
            "Number of input image channels.  Use 1 for grayscale (paper default) or 3 for RGB."
        ),
    )

    # Encoder architecture
    encoder_channels: tuple[int, ...] = Field(
        default=(64, 128),
        min_length=1,
        description=(
            "Output channel count for each encoder convolutional stage.  "
            "Length determines the number of encoder stages (paper default: 2)."
        ),
    )
    encoder_kernel_sizes: tuple[int, ...] = Field(
        default=(11, 5),
        min_length=1,
        description="Kernel size for each encoder convolutional stage.",
    )
    encoder_strides: tuple[int, ...] = Field(
        default=(2, 1),
        min_length=1,
        description=(
            "Stride for each encoder Conv2d.  A stride > 1 performs spatial "
            "down-sampling within the convolution itself."
        ),
    )
    encoder_paddings: tuple[int, ...] = Field(
        default=(0, 2),
        min_length=1,
        description="Padding for each encoder Conv2d.",
    )
    pool_kernel_size: int = Field(
        default=2,
        ge=1,
        description="Kernel size for MaxPool2d applied after each encoder conv stage.",
    )
    use_batch_norm: bool = Field(
        default=False,
        description=(
            "If True, add BatchNorm2d after each encoder Conv2d.  The original "
            "autoencoder does not use batch normalization (False)."
        ),
    )

    # Output
    output_activation: Literal["sigmoid", "relu"] | None = Field(
        default=None,
        description=(
            "Activation applied to the decoder's final output.  Use 'sigmoid' "
            "when inputs are normalized to [0, 1], 'relu' for [0, inf), or "
            "None for no activation (linear output)."
        ),
    )

    # Validators

    @field_validator("encoder_channels", mode="after")
    @classmethod
    def _channels_positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(ch < 1 for ch in v):
            raise ValueError("All encoder channel counts must be >= 1.")
        return v

    @field_validator("encoder_kernel_sizes", mode="after")
    @classmethod
    def _kernel_sizes_positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(k < 1 for k in v):
            raise ValueError("All encoder kernel sizes must be >= 1.")
        return v

    @field_validator("encoder_strides", mode="after")
    @classmethod
    def _strides_positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(s < 1 for s in v):
            raise ValueError("All encoder strides must be >= 1.")
        return v

    @field_validator("encoder_paddings", mode="after")
    @classmethod
    def _paddings_non_negative(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(p < 0 for p in v):
            raise ValueError("All encoder paddings must be >= 0.")
        return v

    @model_validator(mode="after")
    def _encoder_tuples_same_length(self) -> DeepFontAEConfig:
        """Ensure all per-stage encoder tuples share the same length."""
        n = len(self.encoder_channels)
        names = ("encoder_kernel_sizes", "encoder_strides", "encoder_paddings")
        for name in names:
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"Length of '{name}' ({len(getattr(self, name))}) must match "
                    f"length of 'encoder_channels' ({n})."
                )
        return self


class DeepFontConfig(BaseModel):
    """Configuration for the DeepFont classifier.

    Parameterizes all three sub-networks: the shared encoder, the additional
    convolutional feature layers, and the fully-connected classification head.

    Example:
        >>> from deepfont.models.config import DeepFontConfig
        >>> from deepfont.models.deepfont import DeepFont
        >>> # Paper defaults (2383 font classes)
        >>> config = DeepFontConfig()
        >>> model = DeepFont(config)
        >>> # Smaller model for a 100-class subset
        >>> config = DeepFontConfig(
        ...     num_classes=100,
        ...     conv_channels=128,
        ...     fc_hidden_dims=(2048,),
        ...     dropout_rate=0.2,
        ... )
        >>> model = DeepFont(config)
    """

    model_config = ConfigDict(frozen=True)

    # Input / output
    in_channels: int = Field(
        default=1,
        ge=1,
        le=4,
        description=(
            "Number of input image channels.  Use 1 for grayscale (paper default) or 3 for RGB."
        ),
    )
    input_size: int = Field(
        default=105,
        ge=1,
        description=(
            "Spatial size (height = width) of the square input image.  Used to "
            "compute the flattened feature dimension before the first FC layer."
        ),
    )
    num_classes: int = Field(
        default=2383,
        ge=2,
        description=(
            "Number of font classes for the final classification layer.  "
            "The original AdobeVFR dataset uses 2 383 classes."
        ),
    )

    # Encoder architecture
    encoder_channels: tuple[int, ...] = Field(
        default=(64, 128),
        min_length=1,
        description="Output channel count for each encoder convolutional stage.",
    )
    encoder_kernel_sizes: tuple[int, ...] = Field(
        default=(11, 5),
        min_length=1,
        description="Kernel size for each encoder Conv2d.",
    )
    encoder_strides: tuple[int, ...] = Field(
        default=(2, 1),
        min_length=1,
        description="Stride for each encoder Conv2d.",
    )
    encoder_paddings: tuple[int, ...] = Field(
        default=(0, 2),
        min_length=1,
        description="Padding for each encoder Conv2d.",
    )
    pool_kernel_size: int = Field(
        default=2,
        ge=1,
        description="Kernel size for MaxPool2d applied after each encoder conv stage.",
    )
    use_encoder_batch_norm: bool = Field(
        default=True,
        description=(
            "If True, add BatchNorm2d after each encoder Conv2d.  The original "
            "classifier uses batch normalization (True)."
        ),
    )

    # Convolutional feature layers
    num_conv_layers: int = Field(
        default=3,
        ge=1,
        description=(
            "Number of additional convolutional layers after the encoder.  "
            "The paper uses 3 layers with 256 filters each."
        ),
    )
    conv_channels: int = Field(
        default=256,
        ge=1,
        description="Output channel count for each additional conv layer.",
    )
    conv_kernel_size: int = Field(
        default=3,
        ge=1,
        description=(
            "Kernel size for each additional conv layer.  Uses padding='same' "
            "to preserve spatial dimensions."
        ),
    )
    use_conv_batch_norm: bool = Field(
        default=True,
        description="If True, add BatchNorm2d after each additional conv layer.",
    )

    # Fully-connected head
    fc_hidden_dims: tuple[int, ...] = Field(
        default=(4096, 4096),
        min_length=1,
        description=(
            "Hidden-layer dimensions for the FC classification head.  "
            "Length determines the number of hidden layers (paper default: 2)."
        ),
    )
    dropout_rate: float = Field(
        default=0.1,
        ge=0.0,
        lt=1.0,
        description=(
            "Dropout probability applied after each hidden FC layer.  The paper "
            "uses 0.1, considerably lower than the 0.5 common in AlexNet/VGG."
        ),
    )

    # Validators

    @field_validator("encoder_channels", mode="after")
    @classmethod
    def _channels_positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(ch < 1 for ch in v):
            raise ValueError("All encoder channel counts must be >= 1.")
        return v

    @field_validator("encoder_kernel_sizes", mode="after")
    @classmethod
    def _kernel_sizes_positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(k < 1 for k in v):
            raise ValueError("All encoder kernel sizes must be >= 1.")
        return v

    @field_validator("encoder_strides", mode="after")
    @classmethod
    def _strides_positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(s < 1 for s in v):
            raise ValueError("All encoder strides must be >= 1.")
        return v

    @field_validator("encoder_paddings", mode="after")
    @classmethod
    def _paddings_non_negative(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(p < 0 for p in v):
            raise ValueError("All encoder paddings must be >= 0.")
        return v

    @field_validator("fc_hidden_dims", mode="after")
    @classmethod
    def _fc_dims_positive(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if any(d < 1 for d in v):
            raise ValueError("All FC hidden dimensions must be >= 1.")
        return v

    @model_validator(mode="after")
    def _encoder_tuples_same_length(self) -> DeepFontConfig:
        """Ensure all per-stage encoder tuples share the same length."""
        n = len(self.encoder_channels)
        names = ("encoder_kernel_sizes", "encoder_strides", "encoder_paddings")
        for name in names:
            if len(getattr(self, name)) != n:
                raise ValueError(
                    f"Length of '{name}' ({len(getattr(self, name))}) must match "
                    f"length of 'encoder_channels' ({n})."
                )
        return self

    @model_validator(mode="after")
    def _spatial_dim_positive(self) -> DeepFontConfig:
        """Verify the encoder does not reduce spatial dimensions to zero."""
        h = self.input_size
        for k, s, p in zip(
            self.encoder_kernel_sizes, self.encoder_strides, self.encoder_paddings, strict=True
        ):
            h = (h - k + 2 * p) // s + 1
            if h < 1:
                raise ValueError(
                    f"Encoder conv reduces spatial dimension to {h} "
                    f"(kernel={k}, stride={s}, padding={p}).  "
                    f"Increase input_size or adjust encoder parameters."
                )
            h = h // self.pool_kernel_size
            if h < 1:
                raise ValueError(
                    f"MaxPool2d(kernel_size={self.pool_kernel_size}) reduces "
                    f"spatial dimension to {h}.  Increase input_size or "
                    f"reduce pool_kernel_size."
                )
        return self
