"""Tests for deepfont.trainer.pretrain.PretrainTrainer.

Tests cover all methods that can be exercised without a real BCF dataset.
create_dataloaders() is excluded because it requires a BCFStoreFile and an
image directory.

Test classes:
    TestReconstructionLoss    -- _reconstruction_loss() values and error handling
    TestTrainingStep          -- training_step() output structure and finiteness
    TestValidationStep        -- validation_step() output structure and finiteness
    TestCreateOptimizer       -- optimizer type, lr, weight_decay, scheduler wiring
    TestSaveEncoderWeights    -- encoder weights extraction and round-trip equality
"""

import os

import torch
import pytest

from deepfont.trainer.config import PretrainConfig
from deepfont.models.deepfont import DeepFontAE
from deepfont.trainer.pretrain import PretrainTrainer

# ---------------------------------------------------------------------------
# Helper factory
# ---------------------------------------------------------------------------


def _make_trainer(**overrides) -> PretrainTrainer:
    """Return a CPU PretrainTrainer configured for fast unit tests."""
    config = PretrainConfig(
        accelerator="cpu",
        devices=1,
        num_workers=0,
        **overrides,
    )
    return PretrainTrainer(config)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestReconstructionLoss:
    """_reconstruction_loss() computes the correct scalar for mse and l1."""

    def test_mse_identical_tensors_is_zero(self):
        """MSE loss between a tensor and itself is zero."""
        trainer = _make_trainer(reconstruction_loss="mse")
        x = torch.randn(2, 1, 105, 105)
        assert trainer._reconstruction_loss(x, x).item() == pytest.approx(0.0)

    def test_l1_identical_tensors_is_zero(self):
        """L1 loss between a tensor and itself is zero."""
        trainer = _make_trainer(reconstruction_loss="l1")
        x = torch.randn(2, 1, 105, 105)
        assert trainer._reconstruction_loss(x, x).item() == pytest.approx(0.0)

    def test_mse_nonzero_for_different_tensors(self):
        """MSE loss is positive when pred and target differ."""
        trainer = _make_trainer(reconstruction_loss="mse")
        pred = torch.ones(2, 1, 105, 105)
        target = torch.zeros(2, 1, 105, 105)
        assert trainer._reconstruction_loss(pred, target).item() > 0.0

    def test_l1_nonzero_for_different_tensors(self):
        """L1 loss is positive when pred and target differ."""
        trainer = _make_trainer(reconstruction_loss="l1")
        pred = torch.ones(2, 1, 105, 105)
        target = torch.zeros(2, 1, 105, 105)
        assert trainer._reconstruction_loss(pred, target).item() > 0.0

    def test_mse_and_l1_differ_for_large_error(self):
        """For pred=2, target=0: MSE==4 and L1==2 (distinct values)."""
        pred = torch.full((1,), 2.0)
        target = torch.zeros(1)
        mse_val = _make_trainer(reconstruction_loss="mse")._reconstruction_loss(pred, target)
        l1_val = _make_trainer(reconstruction_loss="l1")._reconstruction_loss(pred, target)
        assert mse_val.item() == pytest.approx(4.0)
        assert l1_val.item() == pytest.approx(2.0)

    def test_unknown_loss_type_raises_value_error(self):
        """An unrecognized reconstruction_loss string raises ValueError."""
        trainer = _make_trainer(reconstruction_loss="huber")
        x = torch.randn(2, 1, 105, 105)
        with pytest.raises(ValueError, match="Unknown reconstruction_loss"):
            trainer._reconstruction_loss(x, x)


class TestTrainingStep:
    """training_step() returns a finite scalar loss dict."""

    def setup_method(self):
        self.trainer = _make_trainer()
        self.model = DeepFontAE()

    def _batch(self) -> torch.Tensor:
        return torch.randn(4, 1, 105, 105)

    def test_returns_dict_with_loss_key(self):
        out = self.trainer.training_step(self.model, self._batch(), 0)
        assert "loss" in out

    def test_loss_is_scalar(self):
        out = self.trainer.training_step(self.model, self._batch(), 0)
        assert out["loss"].ndim == 0

    def test_loss_is_finite(self):
        out = self.trainer.training_step(self.model, self._batch(), 0)
        assert torch.isfinite(out["loss"])

    def test_batch_idx_is_unused(self):
        """Same batch with different batch_idx produces an identical loss."""
        batch = self._batch()
        out0 = self.trainer.training_step(self.model, batch, 0)
        out99 = self.trainer.training_step(self.model, batch, 99)
        assert out0["loss"].item() == pytest.approx(out99["loss"].item())


class TestValidationStep:
    """validation_step() returns a finite scalar loss dict."""

    def setup_method(self):
        self.trainer = _make_trainer()
        self.model = DeepFontAE()

    def _batch(self) -> torch.Tensor:
        return torch.randn(4, 1, 105, 105)

    def test_returns_dict_with_loss_key(self):
        out = self.trainer.validation_step(self.model, self._batch(), 0)
        assert "loss" in out

    def test_loss_is_scalar(self):
        out = self.trainer.validation_step(self.model, self._batch(), 0)
        assert out["loss"].ndim == 0

    def test_loss_is_finite(self):
        out = self.trainer.validation_step(self.model, self._batch(), 0)
        assert torch.isfinite(out["loss"])


class TestCreateOptimizer:
    """create_optimizer() returns an Adam optimizer with the configured hyperparams."""

    def setup_method(self):
        self.model = DeepFontAE()

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
        optim, sched = _make_trainer(
            scheduler_type="cosine", scheduler_kwargs={"T_max": 10}
        ).create_optimizer(self.model)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step_scheduler_when_configured(self):
        optim, sched = _make_trainer(
            scheduler_type="step", scheduler_kwargs={"step_size": 5}
        ).create_optimizer(self.model)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)


class TestSaveEncoderWeights:
    """save_encoder_weights() extracts and saves encoder state correctly."""

    def _write_fake_ckpt(self, path: str) -> dict:
        """Save a fake Fabric-style checkpoint and return the model state dict."""
        model_state = DeepFontAE().state_dict()
        torch.save({"model": model_state}, path)
        return model_state

    def test_output_file_is_created(self, tmp_path):
        """save_encoder_weights creates the output file on disk."""
        trainer = _make_trainer()
        ckpt_path = str(tmp_path / "fake.ckpt")
        output_path = str(tmp_path / "encoder.pt")
        self._write_fake_ckpt(ckpt_path)

        trainer.save_encoder_weights(ckpt_path=ckpt_path, output_path=output_path)
        assert os.path.exists(output_path)

    def test_output_has_same_keys_as_deepfontae_state_dict(self, tmp_path):
        """The saved file contains exactly the same parameter keys as DeepFontAE."""
        trainer = _make_trainer()
        ckpt_path = str(tmp_path / "fake.ckpt")
        output_path = str(tmp_path / "encoder.pt")
        model_state = self._write_fake_ckpt(ckpt_path)

        trainer.save_encoder_weights(ckpt_path=ckpt_path, output_path=output_path)
        saved = torch.load(output_path, map_location="cpu", weights_only=False)
        assert set(saved.keys()) == set(model_state.keys())

    def test_saved_weights_match_original_tensors(self, tmp_path):
        """Every parameter tensor in the saved file is identical to the original."""
        trainer = _make_trainer()
        ckpt_path = str(tmp_path / "fake.ckpt")
        output_path = str(tmp_path / "encoder.pt")
        model_state = self._write_fake_ckpt(ckpt_path)

        trainer.save_encoder_weights(ckpt_path=ckpt_path, output_path=output_path)
        saved = torch.load(output_path, map_location="cpu", weights_only=False)
        for key in model_state:
            assert torch.equal(saved[key], model_state[key])
