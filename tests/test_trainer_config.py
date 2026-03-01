"""Tests for deepfont.trainer.config dataclasses.

These tests lock the default values of all three config dataclasses so that
accidental field changes produce an immediate, descriptive failure rather than a
silent regression.

Test classes:
    TestTrainerConfigDefaults    -- pins every field default in TrainerConfig
    TestPretrainConfigDefaults   -- pins every field default in PretrainConfig
    TestFinetuneConfigDefaults   -- pins every field default in FinetuneConfig
    TestConfigInheritance        -- isinstance, override propagation, mutable defaults
"""

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


class TestPretrainConfigDefaults:
    """Pin default values for all PretrainConfig-specific fields."""

    def test_bcf_store_file_default(self):
        assert PretrainConfig().bcf_store_file == ""

    def test_data_folder_name_default(self):
        assert PretrainConfig().data_folder_name is None

    def test_aug_prob_default(self):
        assert PretrainConfig().aug_prob == 0.5

    def test_image_normalization_default(self):
        assert PretrainConfig().image_normalization == "0to1"

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

    def test_output_activation_default(self):
        assert PretrainConfig().output_activation is None


class TestFinetuneConfigDefaults:
    """Pin default values for all FinetuneConfig-specific fields."""

    def test_bcf_store_file_default(self):
        assert FinetuneConfig().bcf_store_file == ""

    def test_label_file_default(self):
        assert FinetuneConfig().label_file == ""

    def test_aug_prob_default(self):
        assert FinetuneConfig().aug_prob == 0.5

    def test_image_normalization_default(self):
        assert FinetuneConfig().image_normalization == "0to1"

    def test_num_images_to_cache_default(self):
        assert FinetuneConfig().num_images_to_cache == 0

    def test_eval_bcf_store_file_default(self):
        assert FinetuneConfig().eval_bcf_store_file == ""

    def test_eval_label_file_default(self):
        assert FinetuneConfig().eval_label_file == ""

    def test_num_image_crops_default(self):
        assert FinetuneConfig().num_image_crops == 15

    def test_learning_rate_default(self):
        assert FinetuneConfig().learning_rate == 1e-4

    def test_weight_decay_default(self):
        assert FinetuneConfig().weight_decay == 0.0

    def test_scheduler_type_default(self):
        assert FinetuneConfig().scheduler_type is None

    def test_scheduler_kwargs_default(self):
        assert FinetuneConfig().scheduler_kwargs == {}

    def test_num_classes_default(self):
        assert FinetuneConfig().num_classes == 2383

    def test_encoder_weights_path_default(self):
        assert FinetuneConfig().encoder_weights_path is None


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
