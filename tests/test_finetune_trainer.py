"""Tests for deepfont.trainer.finetune.FinetuneTrainer.

Tests cover all methods that can be exercised without a real BCF dataset.
create_dataloaders() is excluded because it requires a BCFStoreFile and a
label file on disk.

Test classes:
    TestTrainingStep      -- training_step() output structure, finiteness, and accuracy
    TestValidationStep    -- validation_step() output structure and finiteness
    TestCreateOptimizer   -- optimizer type, lr, weight_decay, scheduler wiring,
                             and exclusion of frozen encoder parameters
    TestCreateModel       -- returned model type, num_classes, encoder freezing
    TestEvaluate          -- ValueError when eval data paths are not configured
"""

import torch
import pytest

from deepfont.trainer.config import FinetuneConfig
from deepfont.models.deepfont import DeepFont, DeepFontAE
from deepfont.trainer.finetune import FinetuneTrainer

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------

_NUM_CLASSES = 10  # small class count keeps forward passes fast


def _make_trainer(**overrides) -> FinetuneTrainer:
    """Return a CPU FinetuneTrainer configured for fast unit tests.

    Keyword arguments in *overrides* replace the defaults, so callers can
    set any FinetuneConfig field (including num_classes) without conflicts.
    """
    defaults = dict(
        accelerator="cpu",
        devices=1,
        num_workers=0,
        num_classes=_NUM_CLASSES,
    )
    defaults.update(overrides)
    return FinetuneTrainer(FinetuneConfig(**defaults))


def _save_fake_encoder_weights(path: str) -> None:
    """Save a DeepFontAE state dict to *path* as fake pretrained encoder weights."""
    torch.save(DeepFontAE().state_dict(), path)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """training_step() returns a finite scalar loss and a bounded accuracy."""

    def setup_method(self):
        self.trainer = _make_trainer()
        self.model = DeepFont(num_out=_NUM_CLASSES)

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.randn(4, 1, 105, 105)
        labels = torch.randint(0, _NUM_CLASSES, (4,))
        return images, labels

    def test_returns_dict_with_loss_key(self):
        out = self.trainer.training_step(self.model, self._batch(), 0)
        assert "loss" in out

    def test_returns_dict_with_acc_key(self):
        out = self.trainer.training_step(self.model, self._batch(), 0)
        assert "acc" in out

    def test_loss_is_scalar(self):
        out = self.trainer.training_step(self.model, self._batch(), 0)
        assert out["loss"].ndim == 0

    def test_loss_is_finite(self):
        out = self.trainer.training_step(self.model, self._batch(), 0)
        assert torch.isfinite(out["loss"])

    def test_acc_is_scalar(self):
        out = self.trainer.training_step(self.model, self._batch(), 0)
        assert out["acc"].ndim == 0

    def test_acc_in_unit_interval(self):
        out = self.trainer.training_step(self.model, self._batch(), 0)
        assert 0.0 <= out["acc"].item() <= 1.0

    def test_batch_idx_is_unused(self):
        """Same batch passed with different batch_idx values produces the same loss.

        The model is switched to eval mode so that BatchNorm uses frozen running
        statistics instead of updating them, making two forward passes on the
        same batch deterministic.
        """
        self.model.eval()
        batch = self._batch()
        out0 = self.trainer.training_step(self.model, batch, 0)
        out99 = self.trainer.training_step(self.model, batch, 99)
        assert out0["loss"].item() == pytest.approx(out99["loss"].item())

    def test_perfect_predictions_give_unit_accuracy(self):
        """When model always predicts the correct class, accuracy equals 1.0."""
        model = DeepFont(num_out=2)
        # Force the last linear layer to predict class 0 with certainty
        with torch.no_grad():
            model.fc_part[-1].weight.zero_()
            model.fc_part[-1].bias.zero_()
            model.fc_part[-1].bias[0] = 1e6

        images = torch.randn(4, 1, 105, 105)
        labels = torch.zeros(4, dtype=torch.long)  # all class 0
        trainer = _make_trainer(num_classes=2)
        out = trainer.training_step(model, (images, labels), 0)
        assert out["acc"].item() == pytest.approx(1.0)


class TestValidationStep:
    """validation_step() returns a finite scalar loss and a bounded accuracy."""

    def setup_method(self):
        self.trainer = _make_trainer()
        self.model = DeepFont(num_out=_NUM_CLASSES)

    def _batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.randn(4, 1, 105, 105)
        labels = torch.randint(0, _NUM_CLASSES, (4,))
        return images, labels

    def test_returns_dict_with_loss_key(self):
        out = self.trainer.validation_step(self.model, self._batch(), 0)
        assert "loss" in out

    def test_returns_dict_with_acc_key(self):
        out = self.trainer.validation_step(self.model, self._batch(), 0)
        assert "acc" in out

    def test_loss_is_scalar(self):
        out = self.trainer.validation_step(self.model, self._batch(), 0)
        assert out["loss"].ndim == 0

    def test_loss_is_finite(self):
        out = self.trainer.validation_step(self.model, self._batch(), 0)
        assert torch.isfinite(out["loss"])

    def test_acc_is_scalar(self):
        out = self.trainer.validation_step(self.model, self._batch(), 0)
        assert out["acc"].ndim == 0

    def test_acc_in_unit_interval(self):
        out = self.trainer.validation_step(self.model, self._batch(), 0)
        assert 0.0 <= out["acc"].item() <= 1.0


class TestCreateOptimizer:
    """create_optimizer() returns an Adam optimizer with the configured hyperparams."""

    def setup_method(self):
        self.model = DeepFont(num_out=_NUM_CLASSES)

    def test_returns_adam_optimizer(self):
        optim, _ = _make_trainer().create_optimizer(self.model)
        assert isinstance(optim, torch.optim.Adam)

    def test_learning_rate_matches_config(self):
        lr = 5e-4
        optim, _ = _make_trainer(learning_rate=lr).create_optimizer(self.model)
        assert optim.param_groups[0]["lr"] == pytest.approx(lr)

    def test_weight_decay_matches_config(self):
        wd = 1e-4
        optim, _ = _make_trainer(weight_decay=wd).create_optimizer(self.model)
        assert optim.param_groups[0]["weight_decay"] == pytest.approx(wd)

    def test_no_scheduler_by_default(self):
        _, sched = _make_trainer().create_optimizer(self.model)
        assert sched is None

    def test_cosine_scheduler_when_configured(self):
        _, sched = _make_trainer(
            scheduler_type="cosine", scheduler_kwargs={"T_max": 10}
        ).create_optimizer(self.model)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step_scheduler_when_configured(self):
        _, sched = _make_trainer(
            scheduler_type="step", scheduler_kwargs={"step_size": 5}
        ).create_optimizer(self.model)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_frozen_params_excluded_from_optimizer(self, tmp_path):
        """Optimizer param count equals the number of trainable parameters only."""
        encoder_weights = str(tmp_path / "encoder.pt")
        _save_fake_encoder_weights(encoder_weights)

        model = DeepFont(num_out=_NUM_CLASSES)
        model.load_encoder_weights(encoder_weights)

        optim, _ = _make_trainer(encoder_weights_path=encoder_weights).create_optimizer(model)

        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        optimizer_count = sum(len(pg["params"]) for pg in optim.param_groups)
        assert optimizer_count == trainable_count


class TestCreateModel:
    """create_model() returns a correctly configured DeepFont instance."""

    def test_returns_deepfont_instance(self):
        trainer = _make_trainer()
        model = trainer.create_model()
        assert isinstance(model, DeepFont)

    def test_num_classes_respected(self):
        trainer = _make_trainer(num_classes=5)
        model = trainer.create_model()
        assert model.num_out == 5

    def test_all_params_trainable_without_encoder_weights(self):
        """Without encoder_weights_path every parameter should require gradients."""
        trainer = _make_trainer()
        model = trainer.create_model()
        assert all(p.requires_grad for p in model.parameters())

    def test_encoder_conv_weights_frozen_with_encoder_weights_path(self, tmp_path):
        """Conv weights of encoder[0] and encoder[4] are frozen after weight loading."""
        encoder_weights = str(tmp_path / "encoder.pt")
        _save_fake_encoder_weights(encoder_weights)

        trainer = _make_trainer(encoder_weights_path=encoder_weights)
        model = trainer.create_model()

        frozen = {name for name, p in model.encoder.named_parameters() if not p.requires_grad}
        assert "0.weight" in frozen
        assert "0.bias" in frozen
        assert "4.weight" in frozen
        assert "4.bias" in frozen

    def test_non_encoder_parts_remain_trainable_with_encoder_weights_path(self, tmp_path):
        """conv_part and fc_part are unaffected by encoder weight loading."""
        encoder_weights = str(tmp_path / "encoder.pt")
        _save_fake_encoder_weights(encoder_weights)

        trainer = _make_trainer(encoder_weights_path=encoder_weights)
        model = trainer.create_model()

        assert all(p.requires_grad for p in model.conv_part.parameters())
        assert all(p.requires_grad for p in model.fc_part.parameters())


class TestEvaluate:
    """evaluate() raises ValueError when eval data paths are not configured."""

    def test_raises_when_eval_bcf_store_file_is_empty(self):
        """Missing eval_bcf_store_file triggers a descriptive ValueError."""
        trainer = _make_trainer(eval_label_file="test.labels")
        with pytest.raises(ValueError, match="eval_bcf_store_file"):
            trainer.evaluate()

    def test_raises_when_eval_label_file_is_empty(self):
        """Missing eval_label_file triggers a descriptive ValueError."""
        trainer = _make_trainer(eval_bcf_store_file="test.bcf")
        with pytest.raises(ValueError, match="eval_label_file"):
            trainer.evaluate()

    def test_raises_when_both_eval_paths_are_empty(self):
        """Default config (both paths empty) raises ValueError."""
        trainer = _make_trainer()
        with pytest.raises(ValueError):
            trainer.evaluate()
