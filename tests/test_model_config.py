"""Tests for deepfont.models.config Pydantic configuration classes.

These tests lock the default values and validate the constraints of both
model configuration classes so that accidental changes produce immediate,
descriptive failures.

Test classes:
    TestDeepFontAEConfigDefaults        -- pins every field default in DeepFontAEConfig
    TestDeepFontAEConfigValidation      -- field validators, model validators, immutability
    TestDeepFontConfigDefaults          -- pins every field default in DeepFontConfig
    TestDeepFontConfigValidation        -- field validators, model validators, immutability
    TestDeepFontConfigSpatialValidator  -- spatial dimension reduction checks
"""

import pytest
from pydantic import ValidationError

from deepfont.models.config import DeepFontConfig, DeepFontAEConfig

# Shared helpers


def _ae_config(**overrides) -> DeepFontAEConfig:
    """Return a DeepFontAEConfig with sensible defaults, applying overrides."""
    defaults: dict = {}
    defaults.update(overrides)
    return DeepFontAEConfig(**defaults)


def _df_config(**overrides) -> DeepFontConfig:
    """Return a DeepFontConfig with sensible defaults, applying overrides."""
    defaults: dict = {}
    defaults.update(overrides)
    return DeepFontConfig(**defaults)


class TestDeepFontAEConfigDefaults:
    """Pin default values for all DeepFontAEConfig fields."""

    def test_in_channels_default(self):
        """Default in_channels is 1 (grayscale)."""
        assert DeepFontAEConfig().in_channels == 1

    def test_encoder_channels_default(self):
        """Default encoder channels match the paper (64, 128)."""
        assert DeepFontAEConfig().encoder_channels == (64, 128)

    def test_encoder_kernel_sizes_default(self):
        """Default encoder kernel sizes match the paper (10, 5)."""
        assert DeepFontAEConfig().encoder_kernel_sizes == (10, 5)

    def test_encoder_strides_default(self):
        """Default encoder strides match the paper (2, 1)."""
        assert DeepFontAEConfig().encoder_strides == (2, 1)

    def test_encoder_paddings_default(self):
        """Default encoder paddings match the paper (0, 2)."""
        assert DeepFontAEConfig().encoder_paddings == (0, 2)

    def test_pool_kernel_size_default(self):
        """Default pool kernel size is 2."""
        assert DeepFontAEConfig().pool_kernel_size == 2

    def test_use_batch_norm_default(self):
        """Default autoencoder does not use batch normalization."""
        assert DeepFontAEConfig().use_batch_norm is False

    def test_output_activation_default(self):
        """Default output activation is None (linear output)."""
        assert DeepFontAEConfig().output_activation is None


class TestDeepFontAEConfigValidation:
    """Field validators, model validators, and immutability for DeepFontAEConfig."""

    # Immutability

    def test_config_is_frozen(self):
        """Config instances are immutable (frozen=True)."""
        config = DeepFontAEConfig()
        with pytest.raises(ValidationError):
            config.in_channels = 3

    # Field overrides

    def test_custom_in_channels(self):
        """in_channels can be set to 3 for RGB."""
        assert _ae_config(in_channels=3).in_channels == 3

    def test_custom_encoder_channels(self):
        """Custom encoder channels are accepted."""
        cfg = _ae_config(
            encoder_channels=(32, 64, 128),
            encoder_kernel_sizes=(7, 5, 3),
            encoder_strides=(1, 1, 1),
            encoder_paddings=(3, 2, 1),
        )
        assert cfg.encoder_channels == (32, 64, 128)

    def test_custom_output_activation_sigmoid(self):
        """output_activation accepts 'sigmoid'."""
        assert _ae_config(output_activation="sigmoid").output_activation == "sigmoid"

    def test_custom_output_activation_relu(self):
        """output_activation accepts 'relu'."""
        assert _ae_config(output_activation="relu").output_activation == "relu"

    def test_custom_use_batch_norm(self):
        """use_batch_norm can be enabled."""
        assert _ae_config(use_batch_norm=True).use_batch_norm is True

    # Field validators: positive values

    def test_in_channels_below_minimum_rejected(self):
        """in_channels < 1 is rejected."""
        with pytest.raises(ValidationError, match="in_channels"):
            DeepFontAEConfig(in_channels=0)

    def test_in_channels_above_maximum_rejected(self):
        """in_channels > 4 is rejected."""
        with pytest.raises(ValidationError, match="in_channels"):
            DeepFontAEConfig(in_channels=5)

    def test_encoder_channels_zero_rejected(self):
        """Encoder channel counts of zero are rejected."""
        with pytest.raises(ValidationError, match="channel"):
            DeepFontAEConfig(encoder_channels=(0, 128))

    def test_encoder_kernel_sizes_zero_rejected(self):
        """Encoder kernel sizes of zero are rejected."""
        with pytest.raises(ValidationError, match="kernel"):
            DeepFontAEConfig(encoder_kernel_sizes=(0, 5))

    def test_encoder_strides_zero_rejected(self):
        """Encoder strides of zero are rejected."""
        with pytest.raises(ValidationError, match="stride"):
            DeepFontAEConfig(encoder_strides=(0, 1))

    def test_encoder_paddings_negative_rejected(self):
        """Negative encoder paddings are rejected."""
        with pytest.raises(ValidationError, match="padding"):
            DeepFontAEConfig(encoder_paddings=(-1, 2))

    def test_pool_kernel_size_zero_rejected(self):
        """pool_kernel_size < 1 is rejected."""
        with pytest.raises(ValidationError, match="pool_kernel_size"):
            DeepFontAEConfig(pool_kernel_size=0)

    def test_empty_encoder_channels_rejected(self):
        """At least one encoder stage is required."""
        with pytest.raises(ValidationError):
            DeepFontAEConfig(encoder_channels=())

    # Model validator: tuple length mismatch

    def test_mismatched_kernel_sizes_length_rejected(self):
        """encoder_kernel_sizes length must match encoder_channels length."""
        with pytest.raises(ValidationError, match="encoder_kernel_sizes"):
            DeepFontAEConfig(encoder_channels=(64, 128), encoder_kernel_sizes=(11,))

    def test_mismatched_strides_length_rejected(self):
        """encoder_strides length must match encoder_channels length."""
        with pytest.raises(ValidationError, match="encoder_strides"):
            DeepFontAEConfig(encoder_channels=(64, 128), encoder_strides=(2,))

    def test_mismatched_paddings_length_rejected(self):
        """encoder_paddings length must match encoder_channels length."""
        with pytest.raises(ValidationError, match="encoder_paddings"):
            DeepFontAEConfig(encoder_channels=(64, 128), encoder_paddings=(0,))

    # Invalid output_activation

    def test_invalid_output_activation_rejected(self):
        """output_activation only accepts 'sigmoid', 'relu', or None."""
        with pytest.raises(ValidationError):
            DeepFontAEConfig(output_activation="tanh")

    # Single-stage encoder

    def test_single_stage_encoder(self):
        """A single-stage encoder config is valid."""
        cfg = DeepFontAEConfig(
            encoder_channels=(64,),
            encoder_kernel_sizes=(11,),
            encoder_strides=(2,),
            encoder_paddings=(0,),
        )
        assert len(cfg.encoder_channels) == 1

    # Three-stage encoder

    def test_three_stage_encoder(self):
        """A three-stage encoder config is valid."""
        cfg = DeepFontAEConfig(
            encoder_channels=(32, 64, 128),
            encoder_kernel_sizes=(7, 5, 3),
            encoder_strides=(1, 1, 1),
            encoder_paddings=(3, 2, 1),
        )
        assert len(cfg.encoder_channels) == 3


class TestDeepFontConfigDefaults:
    """Pin default values for all DeepFontConfig fields."""

    # Input / output

    def test_in_channels_default(self):
        """Default in_channels is 1 (grayscale)."""
        assert DeepFontConfig().in_channels == 1

    def test_input_size_default(self):
        """Default input size is 105."""
        assert DeepFontConfig().input_size == 105

    def test_num_classes_default(self):
        """Default num_classes matches the AdobeVFR dataset (2383)."""
        assert DeepFontConfig().num_classes == 2383

    # Encoder

    def test_encoder_channels_default(self):
        """Default encoder channels match the paper (64, 128)."""
        assert DeepFontConfig().encoder_channels == (64, 128)

    def test_encoder_kernel_sizes_default(self):
        """Default encoder kernel sizes match the paper (10, 5)."""
        assert DeepFontConfig().encoder_kernel_sizes == (10, 5)

    def test_encoder_strides_default(self):
        """Default encoder strides match the paper (2, 1)."""
        assert DeepFontConfig().encoder_strides == (2, 1)

    def test_encoder_paddings_default(self):
        """Default encoder paddings match the paper (0, 2)."""
        assert DeepFontConfig().encoder_paddings == (0, 2)

    def test_pool_kernel_size_default(self):
        """Default pool kernel size is 2."""
        assert DeepFontConfig().pool_kernel_size == 2

    def test_use_encoder_batch_norm_default(self):
        """Default classifier encoder uses batch normalization."""
        assert DeepFontConfig().use_encoder_batch_norm is True

    # Convolutional feature layers

    def test_num_conv_layers_default(self):
        """Default number of additional conv layers is 3."""
        assert DeepFontConfig().num_conv_layers == 3

    def test_conv_channels_default(self):
        """Default conv channels is 256."""
        assert DeepFontConfig().conv_channels == 256

    def test_conv_kernel_size_default(self):
        """Default conv kernel size is 3."""
        assert DeepFontConfig().conv_kernel_size == 3

    def test_use_conv_batch_norm_default(self):
        """Default conv layers use batch normalization."""
        assert DeepFontConfig().use_conv_batch_norm is True

    # Fully-connected head

    def test_fc_hidden_dims_default(self):
        """Default FC hidden dims match the paper (4096, 4096)."""
        assert DeepFontConfig().fc_hidden_dims == (4096, 4096)

    def test_dropout_rate_default(self):
        """Default dropout rate matches the paper (0.1)."""
        assert DeepFontConfig().dropout_rate == pytest.approx(0.1)


class TestDeepFontConfigValidation:
    """Field validators, model validators, and immutability for DeepFontConfig."""

    # Immutability

    def test_config_is_frozen(self):
        """Config instances are immutable (frozen=True)."""
        config = DeepFontConfig()
        with pytest.raises(ValidationError):
            config.num_classes = 100

    # Field overrides

    def test_custom_num_classes(self):
        """num_classes can be overridden."""
        assert _df_config(num_classes=100).num_classes == 100

    def test_custom_input_size(self):
        """input_size can be overridden."""
        assert _df_config(input_size=64).input_size == 64

    def test_custom_dropout_rate(self):
        """dropout_rate can be overridden."""
        assert _df_config(dropout_rate=0.5).dropout_rate == pytest.approx(0.5)

    def test_custom_fc_hidden_dims(self):
        """fc_hidden_dims can be overridden."""
        assert _df_config(fc_hidden_dims=(2048,)).fc_hidden_dims == (2048,)

    def test_custom_conv_channels(self):
        """conv_channels can be overridden."""
        assert _df_config(conv_channels=128).conv_channels == 128

    def test_custom_use_encoder_batch_norm_disabled(self):
        """use_encoder_batch_norm can be disabled."""
        assert _df_config(use_encoder_batch_norm=False).use_encoder_batch_norm is False

    def test_custom_use_conv_batch_norm_disabled(self):
        """use_conv_batch_norm can be disabled."""
        assert _df_config(use_conv_batch_norm=False).use_conv_batch_norm is False

    # Field validators: boundary values

    def test_in_channels_below_minimum_rejected(self):
        """in_channels < 1 is rejected."""
        with pytest.raises(ValidationError, match="in_channels"):
            DeepFontConfig(in_channels=0)

    def test_in_channels_above_maximum_rejected(self):
        """in_channels > 4 is rejected."""
        with pytest.raises(ValidationError, match="in_channels"):
            DeepFontConfig(in_channels=5)

    def test_input_size_zero_rejected(self):
        """input_size < 1 is rejected."""
        with pytest.raises(ValidationError, match="input_size"):
            DeepFontConfig(input_size=0)

    def test_num_classes_below_minimum_rejected(self):
        """num_classes < 2 is rejected."""
        with pytest.raises(ValidationError, match="num_classes"):
            DeepFontConfig(num_classes=1)

    def test_encoder_channels_zero_rejected(self):
        """Encoder channel counts of zero are rejected."""
        with pytest.raises(ValidationError, match="channel"):
            DeepFontConfig(encoder_channels=(0, 128))

    def test_encoder_kernel_sizes_zero_rejected(self):
        """Encoder kernel sizes of zero are rejected."""
        with pytest.raises(ValidationError, match="kernel"):
            DeepFontConfig(encoder_kernel_sizes=(0, 5))

    def test_encoder_strides_zero_rejected(self):
        """Encoder strides of zero are rejected."""
        with pytest.raises(ValidationError, match="stride"):
            DeepFontConfig(encoder_strides=(0, 1))

    def test_encoder_paddings_negative_rejected(self):
        """Negative encoder paddings are rejected."""
        with pytest.raises(ValidationError, match="padding"):
            DeepFontConfig(encoder_paddings=(-1, 2))

    def test_fc_hidden_dims_zero_rejected(self):
        """FC hidden dimensions of zero are rejected."""
        with pytest.raises(ValidationError, match="FC"):
            DeepFontConfig(fc_hidden_dims=(0,))

    def test_fc_hidden_dims_empty_rejected(self):
        """At least one FC hidden layer is required."""
        with pytest.raises(ValidationError):
            DeepFontConfig(fc_hidden_dims=())

    def test_num_conv_layers_zero_rejected(self):
        """num_conv_layers < 1 is rejected."""
        with pytest.raises(ValidationError, match="num_conv_layers"):
            DeepFontConfig(num_conv_layers=0)

    def test_conv_channels_zero_rejected(self):
        """conv_channels < 1 is rejected."""
        with pytest.raises(ValidationError, match="conv_channels"):
            DeepFontConfig(conv_channels=0)

    def test_conv_kernel_size_zero_rejected(self):
        """conv_kernel_size < 1 is rejected."""
        with pytest.raises(ValidationError, match="conv_kernel_size"):
            DeepFontConfig(conv_kernel_size=0)

    def test_dropout_rate_negative_rejected(self):
        """Negative dropout_rate is rejected."""
        with pytest.raises(ValidationError, match="dropout_rate"):
            DeepFontConfig(dropout_rate=-0.1)

    def test_dropout_rate_one_rejected(self):
        """dropout_rate >= 1.0 is rejected."""
        with pytest.raises(ValidationError, match="dropout_rate"):
            DeepFontConfig(dropout_rate=1.0)

    def test_dropout_rate_zero_accepted(self):
        """dropout_rate of 0.0 is valid."""
        assert _df_config(dropout_rate=0.0).dropout_rate == pytest.approx(0.0)

    # Model validator: tuple length mismatch

    def test_mismatched_kernel_sizes_length_rejected(self):
        """encoder_kernel_sizes length must match encoder_channels length."""
        with pytest.raises(ValidationError, match="encoder_kernel_sizes"):
            DeepFontConfig(encoder_channels=(64, 128), encoder_kernel_sizes=(11,))

    def test_mismatched_strides_length_rejected(self):
        """encoder_strides length must match encoder_channels length."""
        with pytest.raises(ValidationError, match="encoder_strides"):
            DeepFontConfig(encoder_channels=(64, 128), encoder_strides=(2,))

    def test_mismatched_paddings_length_rejected(self):
        """encoder_paddings length must match encoder_channels length."""
        with pytest.raises(ValidationError, match="encoder_paddings"):
            DeepFontConfig(encoder_channels=(64, 128), encoder_paddings=(0,))


class TestDeepFontConfigSpatialValidator:
    """Tests for the _spatial_dim_positive model validator on DeepFontConfig."""

    def test_default_config_passes_spatial_check(self):
        """Default config (105x105) does not collapse spatial dimensions."""
        config = DeepFontConfig()
        assert config.input_size == 105

    def test_tiny_input_size_rejected(self):
        """An input_size too small for the default encoder is rejected."""
        with pytest.raises(ValidationError, match="spatial dimension"):
            DeepFontConfig(input_size=3)

    def test_large_kernel_collapses_spatial_dimension(self):
        """A kernel larger than the input spatial size causes collapse."""
        with pytest.raises(ValidationError, match="spatial dimension"):
            DeepFontConfig(
                input_size=10,
                encoder_channels=(64,),
                encoder_kernel_sizes=(20,),
                encoder_strides=(1,),
                encoder_paddings=(0,),
            )

    def test_large_pool_collapses_spatial_dimension(self):
        """A pool kernel that reduces the post-conv size to zero is rejected."""
        with pytest.raises(ValidationError, match="spatial dimension"):
            DeepFontConfig(
                input_size=12,
                encoder_channels=(64,),
                encoder_kernel_sizes=(3,),
                encoder_strides=(1,),
                encoder_paddings=(1,),
                pool_kernel_size=100,
            )

    def test_small_valid_input_size_accepted(self):
        """A small but sufficient input_size passes the spatial check."""
        config = DeepFontConfig(
            input_size=32,
            encoder_channels=(64,),
            encoder_kernel_sizes=(3,),
            encoder_strides=(1,),
            encoder_paddings=(1,),
            pool_kernel_size=2,
        )
        assert config.input_size == 32
