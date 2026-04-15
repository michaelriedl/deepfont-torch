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

from deepfont.data.config import PretrainDataConfig
from deepfont.models.config import DeepFontAEConfig
from deepfont.models.deepfont import DeepFontAE
from deepfont.trainer.config import PretrainConfig
from deepfont.trainer.pretrain import PretrainTrainer

# Helper factory


def _make_trainer(
    config_overrides: dict | None = None,
    model_config: DeepFontAEConfig | None = None,
    data_config: PretrainDataConfig | None = None,
) -> PretrainTrainer:
    """Return a CPU PretrainTrainer configured for fast unit tests."""
    defaults = dict(
        accelerator="cpu",
        devices=1,
        num_workers=0,
    )
    if config_overrides:
        defaults.update(config_overrides)
    config = PretrainConfig(**defaults)
    return PretrainTrainer(
        config,
        model_config=model_config or DeepFontAEConfig(),
        data_config=data_config or PretrainDataConfig(),
    )


# Test classes


class TestReconstructionLoss:
    """_reconstruction_loss() computes the correct scalar for mse and l1."""

    def test_mse_identical_tensors_is_zero(self):
        """MSE loss between a tensor and itself is zero."""
        trainer = _make_trainer(config_overrides={"reconstruction_loss": "mse"})
        x = torch.randn(2, 1, 105, 105)
        assert trainer._reconstruction_loss(x, x).item() == pytest.approx(0.0)

    def test_l1_identical_tensors_is_zero(self):
        """L1 loss between a tensor and itself is zero."""
        trainer = _make_trainer(config_overrides={"reconstruction_loss": "l1"})
        x = torch.randn(2, 1, 105, 105)
        assert trainer._reconstruction_loss(x, x).item() == pytest.approx(0.0)

    def test_mse_nonzero_for_different_tensors(self):
        """MSE loss is positive when pred and target differ."""
        trainer = _make_trainer(config_overrides={"reconstruction_loss": "mse"})
        pred = torch.ones(2, 1, 105, 105)
        target = torch.zeros(2, 1, 105, 105)
        assert trainer._reconstruction_loss(pred, target).item() > 0.0

    def test_l1_nonzero_for_different_tensors(self):
        """L1 loss is positive when pred and target differ."""
        trainer = _make_trainer(config_overrides={"reconstruction_loss": "l1"})
        pred = torch.ones(2, 1, 105, 105)
        target = torch.zeros(2, 1, 105, 105)
        assert trainer._reconstruction_loss(pred, target).item() > 0.0

    def test_mse_and_l1_differ_for_large_error(self):
        """For pred=2, target=0: MSE==4 and L1==2 (distinct values)."""
        pred = torch.full((1,), 2.0)
        target = torch.zeros(1)
        mse_val = _make_trainer(
            config_overrides={"reconstruction_loss": "mse"}
        )._reconstruction_loss(pred, target)
        l1_val = _make_trainer(
            config_overrides={"reconstruction_loss": "l1"}
        )._reconstruction_loss(pred, target)
        assert mse_val.item() == pytest.approx(4.0)
        assert l1_val.item() == pytest.approx(2.0)

    def test_unknown_loss_type_raises_value_error(self):
        """An unrecognized reconstruction_loss string raises ValueError."""
        # Pydantic Literal validation rejects invalid values at config construction,
        # but we test the runtime path by constructing with a valid value and
        # verifying the error message format
        trainer = _make_trainer(config_overrides={"reconstruction_loss": "mse"})
        # Temporarily override for the test
        object.__setattr__(trainer.config, "reconstruction_loss", "huber")
        x = torch.randn(2, 1, 105, 105)
        with pytest.raises(ValueError, match="Unknown reconstruction_loss"):
            trainer._reconstruction_loss(x, x)


def _make_batch(
    b: int = 4,
    num_real: int = 2,
    h: int = 105,
    w: int = 105,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a (images, is_real) batch tuple as produced by the DataLoader."""
    images = torch.randn(b, 1, h, w)
    is_real = torch.zeros(b, dtype=torch.bool)
    is_real[:num_real] = True
    return images, is_real


class TestTrainingStep:
    """training_step() returns a finite scalar loss dict."""

    def setup_method(self):
        self.trainer = _make_trainer()
        self.model = DeepFontAE()

    def test_returns_dict_with_loss_key(self):
        out = self.trainer.training_step(self.model, _make_batch(), 0)
        assert "loss" in out

    def test_loss_is_scalar(self):
        out = self.trainer.training_step(self.model, _make_batch(), 0)
        assert out["loss"].ndim == 0

    def test_loss_is_finite(self):
        out = self.trainer.training_step(self.model, _make_batch(), 0)
        assert torch.isfinite(out["loss"])

    def test_batch_idx_is_unused(self):
        """Same batch with different batch_idx produces an identical loss."""
        batch = _make_batch()
        out0 = self.trainer.training_step(self.model, batch, 0)
        out99 = self.trainer.training_step(self.model, batch, 99)
        assert out0["loss"].item() == pytest.approx(out99["loss"].item())

    def test_real_loss_present_when_batch_has_real_images(self):
        out = self.trainer.training_step(self.model, _make_batch(b=4, num_real=2), 0)
        assert "real_loss" in out
        assert out["real_loss"].ndim == 0
        assert torch.isfinite(out["real_loss"])

    def test_syn_loss_present_when_batch_has_synthetic_images(self):
        out = self.trainer.training_step(self.model, _make_batch(b=4, num_real=2), 0)
        assert "syn_loss" in out
        assert out["syn_loss"].ndim == 0
        assert torch.isfinite(out["syn_loss"])

    def test_real_loss_absent_when_batch_is_all_synthetic(self):
        out = self.trainer.training_step(self.model, _make_batch(b=4, num_real=0), 0)
        assert "real_loss" not in out
        assert "syn_loss" in out

    def test_syn_loss_absent_when_batch_is_all_real(self):
        out = self.trainer.training_step(self.model, _make_batch(b=4, num_real=4), 0)
        assert "syn_loss" not in out
        assert "real_loss" in out


class TestValidationStep:
    """validation_step() returns a finite scalar loss dict."""

    def setup_method(self):
        self.trainer = _make_trainer()
        self.model = DeepFontAE()

    def test_returns_dict_with_loss_key(self):
        out = self.trainer.validation_step(self.model, _make_batch(), 0)
        assert "loss" in out

    def test_loss_is_scalar(self):
        out = self.trainer.validation_step(self.model, _make_batch(), 0)
        assert out["loss"].ndim == 0

    def test_loss_is_finite(self):
        out = self.trainer.validation_step(self.model, _make_batch(), 0)
        assert torch.isfinite(out["loss"])

    def test_real_loss_present_when_batch_has_real_images(self):
        out = self.trainer.validation_step(self.model, _make_batch(b=4, num_real=2), 0)
        assert "real_loss" in out
        assert out["real_loss"].ndim == 0
        assert torch.isfinite(out["real_loss"])

    def test_syn_loss_present_when_batch_has_synthetic_images(self):
        out = self.trainer.validation_step(self.model, _make_batch(b=4, num_real=2), 0)
        assert "syn_loss" in out
        assert out["syn_loss"].ndim == 0
        assert torch.isfinite(out["syn_loss"])

    def test_real_loss_absent_when_batch_is_all_synthetic(self):
        out = self.trainer.validation_step(self.model, _make_batch(b=4, num_real=0), 0)
        assert "real_loss" not in out
        assert "syn_loss" in out

    def test_syn_loss_absent_when_batch_is_all_real(self):
        out = self.trainer.validation_step(self.model, _make_batch(b=4, num_real=4), 0)
        assert "syn_loss" not in out
        assert "real_loss" in out


class TestCreateOptimizer:
    """create_optimizer() returns the configured optimizer with correct hyperparams."""

    def setup_method(self):
        self.model = DeepFontAE()

    def test_default_optimizer_is_sgd(self):
        optim, _ = _make_trainer().create_optimizer(self.model)
        assert isinstance(optim, torch.optim.SGD)

    def test_adam_optimizer_when_configured(self):
        optim, _ = _make_trainer(
            config_overrides={"optimizer_type": "adam"}
        ).create_optimizer(self.model)
        assert isinstance(optim, torch.optim.Adam)

    def test_adamw_optimizer_when_configured(self):
        optim, _ = _make_trainer(
            config_overrides={"optimizer_type": "adamw"}
        ).create_optimizer(self.model)
        assert isinstance(optim, torch.optim.AdamW)

    def test_learning_rate_matches_config(self):
        lr = 5e-4
        optim, _ = _make_trainer(config_overrides={"learning_rate": lr}).create_optimizer(
            self.model
        )
        assert optim.param_groups[0]["lr"] == pytest.approx(lr)

    def test_weight_decay_matches_config(self):
        wd = 1e-4
        optim, _ = _make_trainer(config_overrides={"weight_decay": wd}).create_optimizer(
            self.model
        )
        assert optim.param_groups[0]["weight_decay"] == pytest.approx(wd)

    def test_optimizer_kwargs_forwarded(self):
        """Extra optimizer kwargs (e.g. momentum) are passed through to the optimizer."""
        optim, _ = _make_trainer(
            config_overrides={"optimizer_kwargs": {"momentum": 0.9}}
        ).create_optimizer(self.model)
        assert optim.param_groups[0]["momentum"] == pytest.approx(0.9)

    def test_no_scheduler_by_default(self):
        _, sched = _make_trainer().create_optimizer(self.model)
        assert sched is None

    def test_cosine_scheduler_when_configured(self):
        optim, sched = _make_trainer(
            config_overrides={"scheduler_type": "cosine", "scheduler_kwargs": {"T_max": 10}}
        ).create_optimizer(self.model)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_step_scheduler_when_configured(self):
        optim, sched = _make_trainer(
            config_overrides={"scheduler_type": "step", "scheduler_kwargs": {"step_size": 5}}
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
