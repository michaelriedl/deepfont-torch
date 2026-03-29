"""Tests for deepfont.trainer.config Pydantic configuration classes.

These tests lock the default values of all three config classes so that
accidental field changes produce an immediate, descriptive failure rather than a
silent regression.

Test classes:
    TestTrainerConfigDefaults    -- pins every field default in TrainerConfig
    TestTrainerConfigValidation  -- field constraints and immutability
    TestPretrainConfigDefaults   -- pins every field default in PretrainConfig
    TestFinetuneConfigDefaults   -- pins every field default in FinetuneConfig
    TestConfigInheritance        -- isinstance, override propagation, mutable defaults
"""

import pytest
from pydantic import ValidationError

from deepfont.trainer.config import TrainerConfig, FinetuneConfig, PretrainConfig


class TestTrainerConfigDefaults:
    """Pin default values for all TrainerConfig fields."""

    def test_accelerator_default(self):
        assert TrainerConfig().accelerator == "auto"

    def test_devices_default(self):
        assert TrainerConfig().devices == "auto"

    def test_strategy_default(self):
        assert TrainerConfig().strategy == "auto"

    def test_precision_default(self):
        assert TrainerConfig().precision == "32-true"

    def test_num_workers_default(self):
        assert TrainerConfig().num_workers == 4

    def test_max_epochs_default(self):
        assert TrainerConfig().max_epochs == 100

    def test_batch_size_default(self):
        assert TrainerConfig().batch_size == 64

    def test_val_batch_size_default(self):
        assert TrainerConfig().val_batch_size == 64

    def test_train_ratio_default(self):
        assert TrainerConfig().train_ratio == 0.8

    def test_grad_accum_steps_default(self):
        assert TrainerConfig().grad_accum_steps == 1

    def test_gradient_clip_val_default(self):
        assert TrainerConfig().gradient_clip_val is None

    def test_val_frequency_default(self):
        assert TrainerConfig().val_frequency == 1

    def test_checkpoint_dir_default(self):
        assert TrainerConfig().checkpoint_dir == "./checkpoints"

    def test_checkpoint_frequency_default(self):
        assert TrainerConfig().checkpoint_frequency == 1

    def test_log_every_n_steps_default(self):
        assert TrainerConfig().log_every_n_steps == 10

    def test_seed_default(self):
        assert TrainerConfig().seed is None


class TestTrainerConfigValidation:
    """Field constraints and immutability for TrainerConfig."""

    def test_config_is_frozen(self):
        config = TrainerConfig()
        with pytest.raises(ValidationError):
            config.batch_size = 128

    def test_max_epochs_none_accepted(self):
        assert TrainerConfig(max_epochs=None).max_epochs is None

    def test_max_epochs_positive_accepted(self):
        assert TrainerConfig(max_epochs=50).max_epochs == 50

    def test_max_epochs_zero_rejected(self):
        with pytest.raises(ValidationError, match="max_epochs"):
            TrainerConfig(max_epochs=0)

    def test_max_epochs_negative_rejected(self):
        with pytest.raises(ValidationError, match="max_epochs"):
            TrainerConfig(max_epochs=-1)

    def test_batch_size_zero_rejected(self):
        with pytest.raises(ValidationError, match="batch_size"):
            TrainerConfig(batch_size=0)

    def test_num_workers_negative_rejected(self):
        with pytest.raises(ValidationError, match="num_workers"):
            TrainerConfig(num_workers=-1)

    def test_train_ratio_zero_rejected(self):
        with pytest.raises(ValidationError, match="train_ratio"):
            TrainerConfig(train_ratio=0.0)

    def test_train_ratio_one_rejected(self):
        with pytest.raises(ValidationError, match="train_ratio"):
            TrainerConfig(train_ratio=1.0)


class TestPretrainConfigDefaults:
    """Pin default values for all PretrainConfig-specific fields."""

    def test_upsample_real_images_default(self):
        assert PretrainConfig().upsample_real_images is True

    def test_num_images_to_cache_default(self):
        assert PretrainConfig().num_images_to_cache == 0

    def test_learning_rate_default(self):
        assert PretrainConfig().learning_rate == 1e-3

    def test_weight_decay_default(self):
        assert PretrainConfig().weight_decay == 0.0

    def test_scheduler_type_default(self):
        assert PretrainConfig().scheduler_type is None

    def test_scheduler_kwargs_default(self):
        assert PretrainConfig().scheduler_kwargs == {}

    def test_reconstruction_loss_default(self):
        assert PretrainConfig().reconstruction_loss == "mse"


class TestFinetuneConfigDefaults:
    """Pin default values for all FinetuneConfig-specific fields."""

    def test_num_images_to_cache_default(self):
        assert FinetuneConfig().num_images_to_cache == 0

    def test_encoder_weights_path_default(self):
        assert FinetuneConfig().encoder_weights_path is None

    def test_learning_rate_default(self):
        assert FinetuneConfig().learning_rate == 1e-4

    def test_weight_decay_default(self):
        assert FinetuneConfig().weight_decay == 0.0

    def test_scheduler_type_default(self):
        assert FinetuneConfig().scheduler_type is None

    def test_scheduler_kwargs_default(self):
        assert FinetuneConfig().scheduler_kwargs == {}


class TestConfigInheritance:
    """Test inheritance, field overrides, and mutable default independence."""

    def test_pretrain_is_trainer_config(self):
        """PretrainConfig is a subclass of TrainerConfig."""
        assert isinstance(PretrainConfig(), TrainerConfig)

    def test_finetune_is_trainer_config(self):
        """FinetuneConfig is a subclass of TrainerConfig."""
        assert isinstance(FinetuneConfig(), TrainerConfig)

    def test_pretrain_field_override(self):
        """Overriding a TrainerConfig field on PretrainConfig takes effect."""
        config = PretrainConfig(max_epochs=42)
        assert config.max_epochs == 42

    def test_finetune_field_override(self):
        """Overriding a TrainerConfig field on FinetuneConfig takes effect."""
        config = FinetuneConfig(batch_size=128)
        assert config.batch_size == 128

    def test_scheduler_kwargs_mutable_default_independence(self):
        """Two separate PretrainConfig instances share no mutable default dict."""
        a = PretrainConfig()
        b = PretrainConfig()
        a.scheduler_kwargs["key"] = "value"
        assert "key" not in b.scheduler_kwargs
