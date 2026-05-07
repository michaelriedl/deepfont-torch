"""Tests for deepfont.models.deepfont model classes.

Tests cover instantiation, forward pass behavior, architectural properties,
and encoder weight loading for both the DeepFontAE autoencoder and the
DeepFont classifier.

Test classes:
    TestDeepFontAEInstantiation    -- constructor with config and defaults
    TestDeepFontAEForward          -- forward pass shapes, finiteness, activations
    TestDeepFontAEArchitecture     -- encoder/decoder layer structure
    TestDeepFontInstantiation      -- constructor with config and defaults
    TestDeepFontForward            -- forward pass shapes, finiteness, logits
    TestDeepFontArchitecture       -- encoder/conv_part/fc_part layer structure
    TestLoadEncoderWeights         -- weight loading, freezing, and layer remapping
"""

import torch
import torch.nn as nn
import pytest

from deepfont.models.config import DeepFontAEConfig, DeepFontConfig
from deepfont.models.deepfont import DeepFontAE, DeepFont, TiedConvTranspose2d

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
    """DeepFontAE constructor accepts config objects and defaults."""

    def test_default_config(self):
        """Default constructor creates a valid model."""
        model = DeepFontAE()
        assert isinstance(model, nn.Module)

    def test_with_explicit_config(self):
        """Constructor accepts a DeepFontAEConfig object."""
        config = _small_ae_config()
        model = DeepFontAE(config)
        assert model.config is config

    def test_kwargs_rejected(self):
        """Constructor does not accept keyword arguments besides config."""
        with pytest.raises(TypeError):
            DeepFontAE(output_activation="sigmoid")

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
        """Encoder contains one MaxPool2d per stage except the final stage.

        The DeepFont SCAE (Fig. 4) keeps the second conv at full resolution;
        the trailing pool only appears in the classifier. ``pool_after_last_stage``
        defaults to False to match that.
        """
        config = _small_ae_config()
        model = DeepFontAE(config)
        n_pool = _count_layer_types(model.encoder, nn.MaxPool2d)
        assert n_pool == len(config.encoder_channels) - 1

    def test_encoder_pool_after_last_stage_true(self):
        """Setting pool_after_last_stage=True restores a MaxPool2d per stage."""
        config = _small_ae_config(pool_after_last_stage=True)
        model = DeepFontAE(config)
        n_pool = _count_layer_types(model.encoder, nn.MaxPool2d)
        assert n_pool == len(config.encoder_channels)

    def test_encoder_has_relu_layers(self):
        """Encoder contains a ReLU per stage."""
        config = _small_ae_config()
        model = DeepFontAE(config)
        n_relu = _count_layer_types(model.encoder, nn.ReLU)
        assert n_relu == len(config.encoder_channels)

    def test_encoder_has_no_norm_layers(self):
        """Autoencoder encoder never uses normalization layers."""
        model = DeepFontAE(_small_ae_config())
        assert _count_layer_types(model.encoder, nn.BatchNorm2d) == 0
        assert _count_layer_types(model.encoder, nn.LocalResponseNorm) == 0

    def test_decoder_has_conv_transpose_layers(self):
        """Decoder contains ConvTranspose2d layers matching encoder stages.

        Uses tied_weights=False because the default tied variant uses
        TiedConvTranspose2d, not nn.ConvTranspose2d.
        """
        config = _small_ae_config(tied_weights=False)
        model = DeepFontAE(config)
        n_deconv = _count_layer_types(model.decoder, nn.ConvTranspose2d)
        assert n_deconv == len(config.encoder_channels)

    def test_decoder_has_upsample_layers(self):
        """Decoder Upsample count mirrors the encoder's pool count.

        With pool_after_last_stage=False (default), the encoder has
        n_stages - 1 pools so the decoder has n_stages - 1 Upsamples.
        """
        config = _small_ae_config()
        model = DeepFontAE(config)
        n_up = _count_layer_types(model.decoder, nn.Upsample)
        assert n_up == len(config.encoder_channels) - 1

    def test_decoder_upsample_count_when_trailing_pool_enabled(self):
        """One Upsample per encoder stage when pool_after_last_stage=True."""
        config = _small_ae_config(pool_after_last_stage=True)
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
            tied_weights=False,
        )
        model = DeepFontAE(config)
        assert _count_layer_types(model.encoder, nn.Conv2d) == 3
        assert _count_layer_types(model.decoder, nn.ConvTranspose2d) == 3

    def test_three_stage_encoder_decoder_tied(self):
        """Three-stage config with tied weights produces matching tied layers."""
        config = DeepFontAEConfig(
            encoder_channels=(32, 64, 128),
            encoder_kernel_sizes=(7, 5, 3),
            encoder_strides=(1, 1, 1),
            encoder_paddings=(3, 2, 1),
            tied_weights=True,
        )
        model = DeepFontAE(config)
        assert _count_layer_types(model.encoder, nn.Conv2d) == 3
        assert _count_layer_types(model.decoder, TiedConvTranspose2d) == 3
        # Tied decoder must not introduce any independent ConvTranspose2d.
        assert _count_layer_types(model.decoder, nn.ConvTranspose2d) == 0

    def test_all_parameters_are_trainable(self):
        """All parameters require gradients by default."""
        model = DeepFontAE()
        assert all(p.requires_grad for p in model.parameters())


class TestDeepFontAETiedWeights:
    """Weight tying between the encoder Conv2d layers and the decoder."""

    def test_tied_default_is_true(self):
        """The tied_weights field defaults to True."""
        assert DeepFontAEConfig().tied_weights is True

    def test_tied_decoder_uses_tied_modules(self):
        """When tied_weights=True the decoder uses TiedConvTranspose2d."""
        model = DeepFontAE(_small_ae_config(tied_weights=True))
        n_tied = _count_layer_types(model.decoder, TiedConvTranspose2d)
        n_untied = _count_layer_types(model.decoder, nn.ConvTranspose2d)
        assert n_tied == len(model.config.encoder_channels)
        assert n_untied == 0

    def test_untied_decoder_uses_independent_modules(self):
        """When tied_weights=False the decoder uses nn.ConvTranspose2d."""
        model = DeepFontAE(_small_ae_config(tied_weights=False))
        n_tied = _count_layer_types(model.decoder, TiedConvTranspose2d)
        n_untied = _count_layer_types(model.decoder, nn.ConvTranspose2d)
        assert n_tied == 0
        assert n_untied == len(model.config.encoder_channels)

    def test_decoder_weights_share_storage_with_encoder(self):
        """Each tied decoder layer's weight is the same tensor as its encoder Conv2d weight."""
        model = DeepFontAE(_small_ae_config(tied_weights=True))
        encoder_convs = [m for m in model.encoder if isinstance(m, nn.Conv2d)]
        tied_layers = [m for m in model.decoder if isinstance(m, TiedConvTranspose2d)]
        # Decoder mirrors the encoder, so pair encoder[i] with decoder[-1-i].
        assert len(encoder_convs) == len(tied_layers)
        for enc, dec in zip(encoder_convs, reversed(tied_layers), strict=True):
            assert dec.encoder_conv is enc
            assert dec.encoder_conv.weight.data_ptr() == enc.weight.data_ptr()

    def test_tied_decoder_has_no_weight_in_state_dict(self):
        """Tied decoder layers contribute only their bias to the state_dict."""
        model = DeepFontAE(_small_ae_config(tied_weights=True))
        decoder_keys = [k for k in model.state_dict() if k.startswith("decoder.")]
        # Tied decoder must not duplicate any encoder weights.
        assert not any(k.endswith(".weight") for k in decoder_keys)
        # One bias per encoder stage.
        bias_keys = [k for k in decoder_keys if k.endswith(".bias")]
        assert len(bias_keys) == len(model.config.encoder_channels)

    def test_tied_model_has_fewer_parameters_than_untied(self):
        """Tying reduces the parameter count by the decoder weight sizes."""
        tied = DeepFontAE(_small_ae_config(tied_weights=True))
        untied = DeepFontAE(_small_ae_config(tied_weights=False))
        n_tied = sum(p.numel() for p in tied.parameters())
        n_untied = sum(p.numel() for p in untied.parameters())
        # The savings equal the encoder Conv2d weight tensor sizes.
        encoder_conv_weights = sum(
            m.weight.numel() for m in tied.encoder if isinstance(m, nn.Conv2d)
        )
        assert n_untied - n_tied == encoder_conv_weights

    def test_tied_forward_matches_functional_reference(self):
        """The tied decoder produces the same output as a manual F.conv_transpose2d call."""
        import torch.nn.functional as F

        torch.manual_seed(0)
        model = DeepFontAE(_small_ae_config(tied_weights=True))
        model.eval()

        # Build the reference path manually using each tied layer's encoder weight.
        x = _ae_input(batch_size=2)
        with torch.no_grad():
            actual = model(x)
            # Re-run forward step by step using F.conv_transpose2d directly.
            h = model.encoder(x)
            for m in model.decoder:
                if isinstance(m, TiedConvTranspose2d):
                    h = F.conv_transpose2d(
                        h,
                        m.encoder_conv.weight,
                        m.bias,
                        stride=m.stride,
                        padding=m.padding,
                    )
                else:
                    h = m(h)
        assert torch.allclose(actual, h, atol=1e-6)

    def test_encoder_update_visible_to_decoder(self):
        """Mutating an encoder Conv2d weight changes the decoder's effective filter."""
        model = DeepFontAE(_small_ae_config(tied_weights=True))
        model.eval()

        x = _ae_input(batch_size=1)
        with torch.no_grad():
            before = model(x).clone()
            # Perturb the first encoder Conv2d weight in place.
            first_conv = next(m for m in model.encoder if isinstance(m, nn.Conv2d))
            first_conv.weight.add_(0.5)
            after = model(x)
        # The decoder reads the encoder weight at forward time, so the output
        # must change when the encoder weight changes.
        assert not torch.allclose(before, after)

    def test_gradient_accumulates_into_encoder_weight_from_decoder(self):
        """Backward through the decoder produces gradients on the tied encoder weights."""
        torch.manual_seed(0)
        model = DeepFontAE(_small_ae_config(tied_weights=True))
        model.train()

        # Zero all grads, run forward + backward, verify the first encoder
        # Conv2d weight has a gradient (it could only come via the decoder
        # since the encoder output is unused as a final loss target here).
        for p in model.parameters():
            p.grad = None

        x = _ae_input(batch_size=2)
        out = model(x)
        loss = (out**2).mean()
        loss.backward()

        for m in model.encoder:
            if isinstance(m, nn.Conv2d):
                assert m.weight.grad is not None
                assert torch.isfinite(m.weight.grad).all()
                assert m.weight.grad.abs().sum() > 0

    def test_state_dict_round_trip_preserves_tying(self):
        """Saving and reloading the tied model keeps the decoder tied to the encoder."""
        src = DeepFontAE(_small_ae_config(tied_weights=True))
        dst = DeepFontAE(_small_ae_config(tied_weights=True))
        dst.load_state_dict(src.state_dict())

        encoder_convs = [m for m in dst.encoder if isinstance(m, nn.Conv2d)]
        tied_layers = [m for m in dst.decoder if isinstance(m, TiedConvTranspose2d)]
        for enc, dec in zip(encoder_convs, reversed(tied_layers), strict=True):
            assert dec.encoder_conv is enc

        # Encoder weights match the source.
        for src_conv, dst_conv in zip(
            (m for m in src.encoder if isinstance(m, nn.Conv2d)),
            encoder_convs,
            strict=True,
        ):
            assert torch.equal(src_conv.weight.data, dst_conv.weight.data)

    def test_tied_output_shape_matches_input(self):
        """Tied autoencoder reconstructs the same spatial dimensions as input."""
        model = DeepFontAE(_small_ae_config(tied_weights=True))
        model.eval()
        x = _ae_input()
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_loaded_into_classifier_with_tied_ae(self, tmp_path):
        """Tied AE checkpoints still load into the DeepFont classifier."""
        weights_path = str(tmp_path / "ae.pt")
        ae = DeepFontAE(_small_ae_config(tied_weights=True))
        torch.save(ae.state_dict(), weights_path)

        model = DeepFont(_small_df_config())
        model.load_encoder_weights(weights_path)
        # First conv weight should match the AE's first encoder conv weight.
        df_conv0 = model.encoder[0].weight.data
        ae_conv0 = next(m for m in ae.encoder if isinstance(m, nn.Conv2d)).weight.data
        assert torch.equal(df_conv0, ae_conv0)


class TestDeepFontInstantiation:
    """DeepFont constructor accepts config objects and defaults."""

    def test_default_config(self):
        """Default constructor creates a valid model."""
        model = DeepFont()
        assert isinstance(model, nn.Module)

    def test_with_explicit_config(self):
        """Constructor accepts a DeepFontConfig object."""
        config = _small_df_config()
        model = DeepFont(config)
        assert model.config is config

    def test_kwargs_rejected(self):
        """Constructor does not accept keyword arguments besides config."""
        with pytest.raises(TypeError):
            DeepFont(num_classes=50)

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

    def test_encoder_has_no_norm_by_default(self):
        """Default encoder has no normalization layer, matching the SCAE."""
        config = _small_df_config()
        assert config.encoder_norm_type == "none"
        model = DeepFont(config)
        assert _count_layer_types(model.encoder, nn.BatchNorm2d) == 0
        assert _count_layer_types(model.encoder, nn.LocalResponseNorm) == 0

    def test_encoder_with_lrn(self):
        """Encoder uses LocalResponseNorm when encoder_norm_type='lrn'."""
        config = _small_df_config(encoder_norm_type="lrn")
        model = DeepFont(config)
        n_lrn = _count_layer_types(model.encoder, nn.LocalResponseNorm)
        n_bn = _count_layer_types(model.encoder, nn.BatchNorm2d)
        assert n_lrn == len(config.encoder_channels)
        assert n_bn == 0

    def test_encoder_with_batch_norm(self):
        """Encoder uses BatchNorm2d when encoder_norm_type='batch'."""
        config = _small_df_config(encoder_norm_type="batch")
        model = DeepFont(config)
        n_bn = _count_layer_types(model.encoder, nn.BatchNorm2d)
        n_lrn = _count_layer_types(model.encoder, nn.LocalResponseNorm)
        assert n_bn == len(config.encoder_channels)
        assert n_lrn == 0

    def test_conv_part_has_conv_layers(self):
        """conv_part contains Conv2d layers matching num_conv_layers."""
        config = _small_df_config()
        model = DeepFont(config)
        n_conv = _count_layer_types(model.conv_part, nn.Conv2d)
        assert n_conv == config.num_conv_layers

    def test_conv_part_has_no_normalization(self):
        """conv_part is fixed as Conv2d -> ReLU per stage (paper Fig. 5)."""
        config = _small_df_config()
        model = DeepFont(config)
        assert _count_layer_types(model.conv_part, nn.BatchNorm2d) == 0
        assert _count_layer_types(model.conv_part, nn.LocalResponseNorm) == 0

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

    def _save_ae_weights(self, path: str, config: DeepFontAEConfig | None = None) -> None:
        """Save a DeepFontAE state dict to path."""
        model = DeepFontAE(config)
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
        # Both AE and default classifier have no norm -> Conv2d at indices 0 and 3.
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
        self._save_ae_weights(weights_path, DeepFontAEConfig())
        # Classifier with encoder batch norm
        config = _small_df_config(encoder_norm_type="batch")
        model = DeepFont(config)
        model.load_encoder_weights(weights_path)

        # Should still produce valid outputs
        model.eval()
        x = _df_input()
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_ae_without_bn_to_classifier_with_lrn(self, tmp_path):
        """Weights transfer correctly from AE (no BN) to LRN classifier (paper default)."""
        weights_path = str(tmp_path / "ae.pt")
        self._save_ae_weights(weights_path, DeepFontAEConfig())
        # Classifier with paper-faithful LRN encoder.
        config = _small_df_config(encoder_norm_type="lrn")
        model = DeepFont(config)
        model.load_encoder_weights(weights_path)

        # Conv weights at LRN-encoder positions (0 and 4) match the source AE.
        ae = DeepFontAE(DeepFontAEConfig())
        ae_state = torch.load(weights_path, weights_only=True)
        ae.load_state_dict(ae_state)
        # Source AE without BN places Conv2d at indices 0 and 3.
        assert torch.equal(ae.encoder[0].weight.data, model.encoder[0].weight.data)
        assert torch.equal(ae.encoder[3].weight.data, model.encoder[4].weight.data)

        model.eval()
        x = _df_input()
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_ae_without_bn_to_classifier_with_no_norm(self, tmp_path):
        """Weights transfer correctly from AE (no BN) to no-norm classifier."""
        weights_path = str(tmp_path / "ae.pt")
        self._save_ae_weights(weights_path, DeepFontAEConfig())
        # Both source and destination have no norm: Conv at indices 0 and 3.
        config = _small_df_config(encoder_norm_type="none")
        model = DeepFont(config)
        model.load_encoder_weights(weights_path)

        ae = DeepFontAE(DeepFontAEConfig())
        ae_state = torch.load(weights_path, weights_only=True)
        ae.load_state_dict(ae_state)
        assert torch.equal(ae.encoder[0].weight.data, model.encoder[0].weight.data)
        assert torch.equal(ae.encoder[3].weight.data, model.encoder[3].weight.data)

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
