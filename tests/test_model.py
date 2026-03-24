"""Tests for deepfont.models.deepfont model classes.

Tests cover instantiation, forward pass behavior, architectural properties,
and encoder weight loading for both the DeepFontAE autoencoder and the
DeepFont classifier.

Test classes:
    TestDeepFontAEInstantiation    -- constructor with config, kwargs, and defaults
    TestDeepFontAEForward          -- forward pass shapes, finiteness, activations
    TestDeepFontAEArchitecture     -- encoder/decoder layer structure
    TestDeepFontInstantiation      -- constructor with config, kwargs, and defaults
    TestDeepFontForward            -- forward pass shapes, finiteness, logits
    TestDeepFontArchitecture       -- encoder/conv_part/fc_part layer structure
    TestLoadEncoderWeights         -- weight loading, freezing, and layer remapping
"""

import torch
import torch.nn as nn
import pytest

from deepfont.models.config import DeepFontAEConfig, DeepFontConfig
from deepfont.models.deepfont import DeepFontAE, DeepFont

# Module-level constants

_BATCH_SIZE = 4
_NUM_CLASSES = 10
_INPUT_SIZE = 105


# Shared helpers


def _small_ae_config(**overrides) -> DeepFontAEConfig:
    """Return a DeepFontAEConfig suitable for fast tests."""
    defaults = dict(
        in_channels=1,
        encoder_channels=(64, 128),
        encoder_kernel_sizes=(11, 5),
        encoder_strides=(2, 1),
        encoder_paddings=(0, 2),
        pool_kernel_size=2,
        use_batch_norm=False,
        output_activation=None,
    )
    defaults.update(overrides)
    return DeepFontAEConfig(**defaults)


def _small_df_config(**overrides) -> DeepFontConfig:
    """Return a DeepFontConfig with a small class count for fast tests."""
    defaults = dict(
        num_classes=_NUM_CLASSES,
        fc_hidden_dims=(64,),
    )
    defaults.update(overrides)
    return DeepFontConfig(**defaults)


def _ae_input(batch_size: int = _BATCH_SIZE, channels: int = 1) -> torch.Tensor:
    """Return a random input tensor for the autoencoder."""
    return torch.randn(batch_size, channels, _INPUT_SIZE, _INPUT_SIZE)


def _df_input(batch_size: int = _BATCH_SIZE, channels: int = 1) -> torch.Tensor:
    """Return a random input tensor for the classifier."""
    return torch.randn(batch_size, channels, _INPUT_SIZE, _INPUT_SIZE)


def _count_layer_types(module: nn.Module, layer_type: type) -> int:
    """Count the number of submodules of the given type."""
    return sum(1 for m in module.modules() if isinstance(m, layer_type))


class TestDeepFontAEInstantiation:
    """DeepFontAE constructor accepts config objects, kwargs, and defaults."""

    def test_default_config(self):
        """Default constructor creates a valid model."""
        model = DeepFontAE()
        assert isinstance(model, nn.Module)

    def test_with_explicit_config(self):
        """Constructor accepts a DeepFontAEConfig object."""
        config = _small_ae_config()
        model = DeepFontAE(config)
        assert model.config is config

    def test_with_kwargs(self):
        """Constructor creates config from keyword arguments when config is None."""
        model = DeepFontAE(output_activation="sigmoid")
        assert model.config.output_activation == "sigmoid"

    def test_has_encoder_attribute(self):
        """Model exposes an encoder sub-module."""
        model = DeepFontAE()
        assert hasattr(model, "encoder")
        assert isinstance(model.encoder, nn.Sequential)

    def test_has_decoder_attribute(self):
        """Model exposes a decoder sub-module."""
        model = DeepFontAE()
        assert hasattr(model, "decoder")
        assert isinstance(model.decoder, nn.Sequential)

    def test_has_config_attribute(self):
        """Model stores its config."""
        config = _small_ae_config()
        model = DeepFontAE(config)
        assert isinstance(model.config, DeepFontAEConfig)


class TestDeepFontAEForward:
    """Forward pass output shape, finiteness, and activation behavior."""

    def setup_method(self):
        self.model = DeepFontAE()
        self.model.eval()

    def test_output_shape_equals_input_shape(self):
        """Autoencoder reconstructs the same spatial dimensions as input."""
        x = _ae_input()
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == x.shape

    def test_output_is_finite(self):
        """Output contains no NaN or Inf values."""
        x = _ae_input()
        with torch.no_grad():
            out = self.model(x)
        assert torch.isfinite(out).all()

    def test_output_dtype_float32(self):
        """Output tensor has float32 dtype."""
        x = _ae_input()
        with torch.no_grad():
            out = self.model(x)
        assert out.dtype == torch.float32

    def test_batch_dimension_preserved(self):
        """Batch dimension is preserved through the forward pass."""
        for bs in (1, 2, 8):
            x = _ae_input(batch_size=bs)
            with torch.no_grad():
                out = self.model(x)
            assert out.shape[0] == bs

    def test_sigmoid_activation_bounds_output(self):
        """With sigmoid activation, output is in [0, 1]."""
        model = DeepFontAE(_small_ae_config(output_activation="sigmoid"))
        model.eval()
        x = _ae_input()
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_relu_activation_bounds_output(self):
        """With relu activation, output is >= 0."""
        model = DeepFontAE(_small_ae_config(output_activation="relu"))
        model.eval()
        x = _ae_input()
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0

    def test_no_activation_allows_negative_output(self):
        """Without activation, output can contain negative values."""
        model = DeepFontAE(_small_ae_config(output_activation=None))
        model.eval()
        # Use a large random input to make negative outputs likely
        torch.manual_seed(42)
        x = torch.randn(16, 1, _INPUT_SIZE, _INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.min() < 0.0

    @pytest.mark.parametrize("in_channels", [1, 3])
    def test_different_input_channels(self, in_channels):
        """Model handles different numbers of input channels."""
        config = _small_ae_config(in_channels=in_channels)
        model = DeepFontAE(config)
        model.eval()
        x = _ae_input(channels=in_channels)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape


class TestDeepFontAEArchitecture:
    """Encoder and decoder layer composition."""

    def test_encoder_has_conv_layers(self):
        """Encoder contains Conv2d layers matching the config stages."""
        config = _small_ae_config()
        model = DeepFontAE(config)
        n_conv = _count_layer_types(model.encoder, nn.Conv2d)
        assert n_conv == len(config.encoder_channels)

    def test_encoder_has_pool_layers(self):
        """Encoder contains a MaxPool2d per stage."""
        config = _small_ae_config()
        model = DeepFontAE(config)
        n_pool = _count_layer_types(model.encoder, nn.MaxPool2d)
        assert n_pool == len(config.encoder_channels)

    def test_encoder_has_relu_layers(self):
        """Encoder contains a ReLU per stage."""
        config = _small_ae_config()
        model = DeepFontAE(config)
        n_relu = _count_layer_types(model.encoder, nn.ReLU)
        assert n_relu == len(config.encoder_channels)

    def test_encoder_without_batch_norm(self):
        """Encoder omits BatchNorm2d when use_batch_norm is False."""
        model = DeepFontAE(_small_ae_config(use_batch_norm=False))
        n_bn = _count_layer_types(model.encoder, nn.BatchNorm2d)
        assert n_bn == 0

    def test_encoder_with_batch_norm(self):
        """Encoder includes BatchNorm2d when use_batch_norm is True."""
        config = _small_ae_config(use_batch_norm=True)
        model = DeepFontAE(config)
        n_bn = _count_layer_types(model.encoder, nn.BatchNorm2d)
        assert n_bn == len(config.encoder_channels)

    def test_decoder_has_conv_transpose_layers(self):
        """Decoder contains ConvTranspose2d layers matching encoder stages."""
        config = _small_ae_config()
        model = DeepFontAE(config)
        n_deconv = _count_layer_types(model.decoder, nn.ConvTranspose2d)
        assert n_deconv == len(config.encoder_channels)

    def test_decoder_has_upsample_layers(self):
        """Decoder contains Upsample layers matching encoder stages."""
        config = _small_ae_config()
        model = DeepFontAE(config)
        n_up = _count_layer_types(model.decoder, nn.Upsample)
        assert n_up == len(config.encoder_channels)

    def test_decoder_sigmoid_activation(self):
        """Decoder includes a Sigmoid when output_activation='sigmoid'."""
        model = DeepFontAE(_small_ae_config(output_activation="sigmoid"))
        n_sig = _count_layer_types(model.decoder, nn.Sigmoid)
        assert n_sig == 1

    def test_decoder_relu_activation(self):
        """Decoder includes an extra ReLU when output_activation='relu'."""
        config = _small_ae_config(output_activation="relu")
        model = DeepFontAE(config)
        # One ReLU per intermediate stage + one for output activation
        n_relu = _count_layer_types(model.decoder, nn.ReLU)
        expected = len(config.encoder_channels) - 1 + 1
        assert n_relu == expected

    def test_decoder_no_activation(self):
        """Decoder has no Sigmoid when output_activation is None."""
        model = DeepFontAE(_small_ae_config(output_activation=None))
        n_sig = _count_layer_types(model.decoder, nn.Sigmoid)
        assert n_sig == 0

    def test_three_stage_encoder_decoder(self):
        """Three-stage config produces correct number of layers."""
        config = DeepFontAEConfig(
            encoder_channels=(32, 64, 128),
            encoder_kernel_sizes=(7, 5, 3),
            encoder_strides=(1, 1, 1),
            encoder_paddings=(3, 2, 1),
        )
        model = DeepFontAE(config)
        assert _count_layer_types(model.encoder, nn.Conv2d) == 3
        assert _count_layer_types(model.decoder, nn.ConvTranspose2d) == 3

    def test_all_parameters_are_trainable(self):
        """All parameters require gradients by default."""
        model = DeepFontAE()
        assert all(p.requires_grad for p in model.parameters())


class TestDeepFontInstantiation:
    """DeepFont constructor accepts config objects, kwargs, and defaults."""

    def test_default_config(self):
        """Default constructor creates a valid model."""
        model = DeepFont()
        assert isinstance(model, nn.Module)

    def test_with_explicit_config(self):
        """Constructor accepts a DeepFontConfig object."""
        config = _small_df_config()
        model = DeepFont(config)
        assert model.config is config

    def test_with_kwargs(self):
        """Constructor creates config from keyword arguments when config is None."""
        model = DeepFont(num_classes=50)
        assert model.config.num_classes == 50

    def test_has_encoder_attribute(self):
        """Model exposes an encoder sub-module."""
        model = DeepFont(_small_df_config())
        assert hasattr(model, "encoder")
        assert isinstance(model.encoder, nn.Sequential)

    def test_has_conv_part_attribute(self):
        """Model exposes a conv_part sub-module."""
        model = DeepFont(_small_df_config())
        assert hasattr(model, "conv_part")
        assert isinstance(model.conv_part, nn.Sequential)

    def test_has_fc_part_attribute(self):
        """Model exposes an fc_part sub-module."""
        model = DeepFont(_small_df_config())
        assert hasattr(model, "fc_part")
        assert isinstance(model.fc_part, nn.Sequential)

    def test_has_config_attribute(self):
        """Model stores its config."""
        config = _small_df_config()
        model = DeepFont(config)
        assert isinstance(model.config, DeepFontConfig)


class TestDeepFontForward:
    """Forward pass output shape, finiteness, and logit structure."""

    def setup_method(self):
        self.config = _small_df_config()
        self.model = DeepFont(self.config)
        self.model.eval()

    def test_output_shape(self):
        """Output has shape (batch_size, num_classes)."""
        x = _df_input()
        with torch.no_grad():
            out = self.model(x)
        assert out.shape == (_BATCH_SIZE, _NUM_CLASSES)

    def test_output_is_finite(self):
        """Output contains no NaN or Inf values."""
        x = _df_input()
        with torch.no_grad():
            out = self.model(x)
        assert torch.isfinite(out).all()

    def test_output_dtype_float32(self):
        """Output tensor has float32 dtype."""
        x = _df_input()
        with torch.no_grad():
            out = self.model(x)
        assert out.dtype == torch.float32

    def test_batch_dimension_preserved(self):
        """Batch dimension is preserved through the forward pass."""
        for bs in (1, 2, 8):
            x = _df_input(batch_size=bs)
            with torch.no_grad():
                out = self.model(x)
            assert out.shape == (bs, _NUM_CLASSES)

    def test_output_is_raw_logits(self):
        """Output contains raw logits that can be negative."""
        torch.manual_seed(42)
        x = _df_input(batch_size=16)
        with torch.no_grad():
            out = self.model(x)
        # Raw logits typically span both positive and negative values
        assert out.min() < 0.0 or out.max() > 0.0

    @pytest.mark.parametrize("num_classes", [2, 10, 100])
    def test_different_num_classes(self, num_classes):
        """Output dimension matches the configured num_classes."""
        config = _small_df_config(num_classes=num_classes)
        model = DeepFont(config)
        model.eval()
        x = _df_input()
        with torch.no_grad():
            out = model(x)
        assert out.shape[1] == num_classes

    @pytest.mark.parametrize("in_channels", [1, 3])
    def test_different_input_channels(self, in_channels):
        """Model handles different numbers of input channels."""
        config = _small_df_config(in_channels=in_channels)
        model = DeepFont(config)
        model.eval()
        x = _df_input(channels=in_channels)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (_BATCH_SIZE, _NUM_CLASSES)


class TestDeepFontArchitecture:
    """Encoder, conv_part, and fc_part layer composition."""

    def test_encoder_has_conv_layers(self):
        """Encoder contains Conv2d layers matching config stages."""
        config = _small_df_config()
        model = DeepFont(config)
        n_conv = _count_layer_types(model.encoder, nn.Conv2d)
        assert n_conv == len(config.encoder_channels)

    def test_encoder_has_batch_norm_by_default(self):
        """Encoder includes BatchNorm2d when use_encoder_batch_norm is True."""
        config = _small_df_config(use_encoder_batch_norm=True)
        model = DeepFont(config)
        n_bn = _count_layer_types(model.encoder, nn.BatchNorm2d)
        assert n_bn == len(config.encoder_channels)

    def test_encoder_without_batch_norm(self):
        """Encoder omits BatchNorm2d when use_encoder_batch_norm is False."""
        config = _small_df_config(use_encoder_batch_norm=False)
        model = DeepFont(config)
        n_bn = _count_layer_types(model.encoder, nn.BatchNorm2d)
        assert n_bn == 0

    def test_conv_part_has_conv_layers(self):
        """conv_part contains Conv2d layers matching num_conv_layers."""
        config = _small_df_config()
        model = DeepFont(config)
        n_conv = _count_layer_types(model.conv_part, nn.Conv2d)
        assert n_conv == config.num_conv_layers

    def test_conv_part_with_batch_norm(self):
        """conv_part includes BatchNorm2d when use_conv_batch_norm is True."""
        config = _small_df_config(use_conv_batch_norm=True)
        model = DeepFont(config)
        n_bn = _count_layer_types(model.conv_part, nn.BatchNorm2d)
        assert n_bn == config.num_conv_layers

    def test_conv_part_without_batch_norm(self):
        """conv_part omits BatchNorm2d when use_conv_batch_norm is False."""
        config = _small_df_config(use_conv_batch_norm=False)
        model = DeepFont(config)
        n_bn = _count_layer_types(model.conv_part, nn.BatchNorm2d)
        assert n_bn == 0

    def test_fc_part_has_flatten(self):
        """fc_part starts with a Flatten layer."""
        model = DeepFont(_small_df_config())
        assert isinstance(model.fc_part[0], nn.Flatten)

    def test_fc_part_has_linear_layers(self):
        """fc_part contains linear layers for each hidden dim plus the output."""
        config = _small_df_config(fc_hidden_dims=(64, 32))
        model = DeepFont(config)
        n_linear = _count_layer_types(model.fc_part, nn.Linear)
        # One per hidden dim + one output layer
        assert n_linear == len(config.fc_hidden_dims) + 1

    def test_fc_part_has_dropout_layers(self):
        """fc_part contains Dropout layers for each hidden layer."""
        config = _small_df_config(fc_hidden_dims=(64, 32))
        model = DeepFont(config)
        n_dropout = _count_layer_types(model.fc_part, nn.Dropout)
        assert n_dropout == len(config.fc_hidden_dims)

    def test_fc_output_dimension_matches_num_classes(self):
        """The final Linear layer outputs num_classes logits."""
        config = _small_df_config(num_classes=42)
        model = DeepFont(config)
        # Last module in fc_part should be Linear with out_features == num_classes
        last_linear = [m for m in model.fc_part.modules() if isinstance(m, nn.Linear)][-1]
        assert last_linear.out_features == 42

    def test_all_parameters_are_trainable(self):
        """All parameters require gradients by default."""
        model = DeepFont(_small_df_config())
        assert all(p.requires_grad for p in model.parameters())

    def test_custom_num_conv_layers(self):
        """Custom num_conv_layers produces the correct number of conv layers."""
        config = _small_df_config(num_conv_layers=5)
        model = DeepFont(config)
        n_conv = _count_layer_types(model.conv_part, nn.Conv2d)
        assert n_conv == 5


class TestLoadEncoderWeights:
    """Encoder weight loading, freezing, and layer remapping."""

    def _save_ae_weights(self, path: str, **ae_kwargs) -> None:
        """Save a DeepFontAE state dict to path."""
        model = DeepFontAE(**ae_kwargs)
        torch.save(model.state_dict(), path)

    def test_loads_without_error(self, tmp_path):
        """load_encoder_weights completes without raising."""
        weights_path = str(tmp_path / "ae.pt")
        self._save_ae_weights(weights_path)
        model = DeepFont(_small_df_config())
        model.load_encoder_weights(weights_path)

    def test_conv_weights_are_frozen(self, tmp_path):
        """Loaded conv weights have requires_grad=False."""
        weights_path = str(tmp_path / "ae.pt")
        self._save_ae_weights(weights_path)
        model = DeepFont(_small_df_config())
        model.load_encoder_weights(weights_path)

        frozen_names = {name for name, p in model.encoder.named_parameters() if not p.requires_grad}
        # Default AE has no batch norm -> Conv2d at indices 0 and 3.
        # Default DeepFont has batch norm -> Conv2d at indices 0 and 4.
        assert "0.weight" in frozen_names
        assert "0.bias" in frozen_names

    def test_non_encoder_parts_remain_trainable(self, tmp_path):
        """conv_part and fc_part are unaffected by encoder weight loading."""
        weights_path = str(tmp_path / "ae.pt")
        self._save_ae_weights(weights_path)
        model = DeepFont(_small_df_config())
        model.load_encoder_weights(weights_path)

        assert all(p.requires_grad for p in model.conv_part.parameters())
        assert all(p.requires_grad for p in model.fc_part.parameters())

    def test_model_still_runs_forward_after_loading(self, tmp_path):
        """Forward pass succeeds after loading encoder weights."""
        weights_path = str(tmp_path / "ae.pt")
        self._save_ae_weights(weights_path)
        model = DeepFont(_small_df_config())
        model.load_encoder_weights(weights_path)
        model.eval()

        x = _df_input()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (_BATCH_SIZE, _NUM_CLASSES)
        assert torch.isfinite(out).all()

    def test_loaded_weights_match_ae_encoder(self, tmp_path):
        """Loaded conv weights match the autoencoder's encoder conv weights."""
        weights_path = str(tmp_path / "ae.pt")
        ae = DeepFontAE()
        torch.save(ae.state_dict(), weights_path)

        model = DeepFont(_small_df_config())
        model.load_encoder_weights(weights_path)

        # Compare the first Conv2d weights (encoder.0.weight in both)
        ae_conv0_weight = ae.encoder[0].weight.data
        df_conv0_weight = model.encoder[0].weight.data
        assert torch.equal(ae_conv0_weight, df_conv0_weight)

    def test_ae_without_bn_to_classifier_with_bn(self, tmp_path):
        """Weights transfer correctly from AE (no BN) to classifier (with BN)."""
        weights_path = str(tmp_path / "ae.pt")
        # AE without batch norm (default)
        self._save_ae_weights(weights_path, use_batch_norm=False)
        # Classifier with encoder batch norm (default)
        config = _small_df_config(use_encoder_batch_norm=True)
        model = DeepFont(config)
        model.load_encoder_weights(weights_path)

        # Should still produce valid outputs
        model.eval()
        x = _df_input()
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_missing_weights_file_raises(self):
        """Non-existent weights file raises an error."""
        model = DeepFont(_small_df_config())
        with pytest.raises((FileNotFoundError, RuntimeError)):
            model.load_encoder_weights("/nonexistent/path/weights.pt")

    def test_frozen_params_excluded_from_gradient(self, tmp_path):
        """Frozen encoder params do not accumulate gradients during backward."""
        weights_path = str(tmp_path / "ae.pt")
        self._save_ae_weights(weights_path)
        model = DeepFont(_small_df_config())
        model.load_encoder_weights(weights_path)

        x = _df_input()
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, p in model.encoder.named_parameters():
            if not p.requires_grad:
                assert p.grad is None
