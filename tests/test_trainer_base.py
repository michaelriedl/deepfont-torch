"""Tests for deepfont.trainer.base.BaseTrainer loop mechanics.

Uses MinimalTrainer, a concrete BaseTrainer subclass backed by a small
nn.Linear model and in-memory TensorDatasets, to exercise the training loop,
gradient accumulation, checkpointing, validation scheduling, and LR scheduler
helpers without touching real data or heavyweight models.

Test classes:
    MinimalTrainer              -- concrete helper subclass (module-level)
    TestShouldValidate          -- _should_validate property across val_frequency values
    TestBuildScheduler          -- _build_scheduler factory for all supported types
    TestStepScheduler           -- _step_scheduler dispatch for each scheduler kind
    TestFitLoop                 -- global_step, current_epoch, checkpoint creation
    TestGradAccum               -- optimizer step count under gradient accumulation
    TestCheckpointRoundtrip     -- resume restores epoch/step counters
    TestValFrequency            -- val_call_count matches expected call pattern
    TestValMetricPrefix         -- returned metric keys carry "val_" prefix
    TestCheckpointFrequency     -- checkpoint files created at correct epochs
"""

import os
from unittest.mock import patch

import torch
import pytest
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss

from deepfont.trainer.base import BaseTrainer
from deepfont.trainer.config import TrainerConfig

# ---------------------------------------------------------------------------
# Concrete trainer used by all test classes
# ---------------------------------------------------------------------------


class MinimalTrainer(BaseTrainer):
    """Minimal concrete BaseTrainer for exercising the loop skeleton.

    Uses a small nn.Linear model and in-memory TensorDatasets so tests run
    quickly with no I/O or GPU requirement.
    """

    _NUM_TRAIN = 24  # 24 / batch_size=4  →  6 batches per epoch
    _NUM_VAL = 8
    _DIM = 4
    _BATCH_SIZE = 4

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.val_call_count: int = 0
        self.last_val_metrics: dict = {}

    def create_model(self) -> nn.Module:
        return nn.Linear(self._DIM, self._DIM, bias=False)

    def create_dataloaders(self):
        train_x = torch.randn(self._NUM_TRAIN, self._DIM)
        val_x = torch.randn(self._NUM_VAL, self._DIM)
        train_loader = DataLoader(
            TensorDataset(train_x),
            batch_size=self._BATCH_SIZE,
            num_workers=0,
            shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(val_x),
            batch_size=self._BATCH_SIZE,
            num_workers=0,
        )
        return train_loader, val_loader

    def create_optimizer(self, model):
        return SGD(model.parameters(), lr=0.01), None

    def training_step(self, model, batch, batch_idx):
        (x,) = batch
        return {"loss": mse_loss(model(x), x)}

    def validation_step(self, model, batch, batch_idx):
        (x,) = batch
        return {"loss": mse_loss(model(x), x)}

    def _val_loop(self, model, val_loader):
        """Override to count calls and capture the returned metrics."""
        self.val_call_count += 1
        metrics = super()._val_loop(model, val_loader)
        self.last_val_metrics = dict(metrics)
        return metrics


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def _base_config(**overrides) -> TrainerConfig:
    """Return a CPU TrainerConfig suitable for fast unit tests.

    Keyword arguments in *overrides* replace any of the defaults below.
    """
    defaults: dict = {
        "accelerator": "cpu",
        "devices": 1,
        "max_epochs": 1,
        "num_workers": 0,
        "log_every_n_steps": 1,
        "val_frequency": 1,
        "checkpoint_frequency": 1,
    }
    defaults.update(overrides)
    return TrainerConfig(**defaults)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestShouldValidate:
    """_should_validate property returns True at the right epochs."""

    def test_val_frequency_1_always_validates(self):
        """val_frequency=1 means every epoch triggers validation."""
        trainer = MinimalTrainer(_base_config(val_frequency=1))
        for epoch in range(10):
            trainer.current_epoch = epoch
            assert trainer._should_validate

    def test_val_frequency_3_validates_at_multiples_of_3(self):
        """val_frequency=3 validates at epochs 0, 3, 6 and skips all others."""
        trainer = MinimalTrainer(_base_config(val_frequency=3))
        for epoch in [0, 3, 6]:
            trainer.current_epoch = epoch
            assert trainer._should_validate
        for epoch in [1, 2, 4, 5, 7, 8]:
            trainer.current_epoch = epoch
            assert not trainer._should_validate

    def test_epoch_0_always_validates_regardless_of_frequency(self):
        """Epoch 0 always validates because 0 % N == 0 for any N."""
        for freq in [1, 2, 3, 5, 10]:
            trainer = MinimalTrainer(_base_config(val_frequency=freq))
            trainer.current_epoch = 0
            assert trainer._should_validate


class TestBuildScheduler:
    """_build_scheduler creates the correct scheduler type from a string name."""

    def _trainer_and_optim(self, **config_overrides):
        trainer = MinimalTrainer(_base_config(**config_overrides))
        model = nn.Linear(4, 4)
        optim = SGD(model.parameters(), lr=0.1)
        return trainer, optim

    def test_none_returns_none(self):
        trainer, optim = self._trainer_and_optim()
        assert trainer._build_scheduler(optim, None, {}) is None

    def test_cosine_type(self):
        trainer, optim = self._trainer_and_optim()
        sched = trainer._build_scheduler(optim, "cosine", {"T_max": 10})
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_cosine_T_max_auto_from_max_epochs(self):
        """When T_max is not given, it defaults to config.max_epochs."""
        trainer, optim = self._trainer_and_optim(max_epochs=7)
        sched = trainer._build_scheduler(optim, "cosine", {})
        assert sched.T_max == 7

    def test_cosine_T_max_explicit_override(self):
        """An explicit T_max in kwargs takes precedence over max_epochs."""
        trainer, optim = self._trainer_and_optim(max_epochs=7)
        sched = trainer._build_scheduler(optim, "cosine", {"T_max": 3})
        assert sched.T_max == 3

    def test_step_type(self):
        trainer, optim = self._trainer_and_optim()
        sched = trainer._build_scheduler(optim, "step", {"step_size": 5})
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_reduce_on_plateau_type(self):
        trainer, optim = self._trainer_and_optim()
        sched = trainer._build_scheduler(optim, "reduce_on_plateau", {})
        assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)

    def test_linear_type(self):
        trainer, optim = self._trainer_and_optim()
        sched = trainer._build_scheduler(optim, "linear", {})
        assert isinstance(sched, torch.optim.lr_scheduler.LinearLR)

    def test_unknown_type_raises_value_error(self):
        trainer, optim = self._trainer_and_optim()
        with pytest.raises(ValueError, match="Unknown scheduler_type"):
            trainer._build_scheduler(optim, "cyclic", {})


class TestStepScheduler:
    """_step_scheduler dispatches correctly for each scheduler kind."""

    def test_none_scheduler_is_noop(self):
        """Calling _step_scheduler(None, ...) must not raise."""
        trainer = MinimalTrainer(_base_config())
        trainer._step_scheduler(None, {})  # must not raise

    def test_step_lr_gets_stepped(self):
        """StepLR.step() is called, reducing the LR after step_size=1 epochs."""
        model = nn.Linear(4, 4)
        optim = SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(optim, step_size=1, gamma=0.5)
        trainer = MinimalTrainer(_base_config())

        initial_lr = optim.param_groups[0]["lr"]
        trainer._step_scheduler(sched, {})
        assert optim.param_groups[0]["lr"] < initial_lr

    def test_reduce_on_plateau_stepped_with_val_loss(self):
        """ReduceLROnPlateau.step is called with the val_loss scalar."""
        model = nn.Linear(4, 4)
        optim = SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
        trainer = MinimalTrainer(_base_config())

        with patch.object(sched, "step") as mock_step:
            val_metrics = {"val_loss": torch.tensor(1.5)}
            trainer._step_scheduler(sched, val_metrics)
            assert mock_step.call_count == 1

    def test_reduce_on_plateau_not_stepped_when_no_val_metrics(self):
        """ReduceLROnPlateau.step is NOT called when val_metrics is empty."""
        model = nn.Linear(4, 4)
        optim = SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
        trainer = MinimalTrainer(_base_config())

        with patch.object(sched, "step") as mock_step:
            trainer._step_scheduler(sched, {})
            mock_step.assert_not_called()


class TestFitLoop:
    """Core fit() loop: step/epoch counters, checkpoint creation, should_stop."""

    def test_global_step_after_one_epoch(self, tmp_path):
        """24 samples / batch_size=4 = 6 optimizer steps after one epoch."""
        trainer = MinimalTrainer(_base_config(max_epochs=1, checkpoint_dir=str(tmp_path)))
        trainer.fit()
        assert trainer.global_step == 6

    def test_current_epoch_after_two_epochs(self, tmp_path):
        """current_epoch equals max_epochs after fit() completes."""
        trainer = MinimalTrainer(_base_config(max_epochs=2, checkpoint_dir=str(tmp_path)))
        trainer.fit()
        assert trainer.current_epoch == 2

    def test_checkpoint_file_created(self, tmp_path):
        """epoch-0001.ckpt exists on disk after one epoch."""
        trainer = MinimalTrainer(_base_config(max_epochs=1, checkpoint_dir=str(tmp_path)))
        trainer.fit()
        assert os.path.exists(os.path.join(str(tmp_path), "epoch-0001.ckpt"))

    def test_should_stop_reset_after_fit(self, tmp_path):
        """should_stop is reset to False after fit() finishes normally."""
        trainer = MinimalTrainer(_base_config(max_epochs=1, checkpoint_dir=str(tmp_path)))
        trainer.fit()
        assert not trainer.should_stop


class TestGradAccum:
    """Gradient accumulation reduces the number of optimizer steps."""

    def test_grad_accum_steps_2_halves_optimizer_steps(self, tmp_path):
        """grad_accum_steps=2 yields 6/2=3 optimizer steps per epoch."""
        trainer = MinimalTrainer(
            _base_config(max_epochs=1, grad_accum_steps=2, checkpoint_dir=str(tmp_path))
        )
        trainer.fit()
        assert trainer.global_step == 3

    def test_grad_accum_steps_6_yields_one_step_per_epoch(self, tmp_path):
        """grad_accum_steps=6 yields 6/6=1 optimizer step per epoch."""
        trainer = MinimalTrainer(
            _base_config(max_epochs=1, grad_accum_steps=6, checkpoint_dir=str(tmp_path))
        )
        trainer.fit()
        assert trainer.global_step == 1


class TestCheckpointRoundtrip:
    """Resuming from a checkpoint restores epoch and step counters."""

    def test_resume_restores_current_epoch(self, tmp_path):
        """After resuming at epoch 1 and running 1 more epoch, current_epoch==2."""
        trainer1 = MinimalTrainer(_base_config(max_epochs=1, checkpoint_dir=str(tmp_path)))
        trainer1.fit()
        ckpt = os.path.join(str(tmp_path), "epoch-0001.ckpt")

        trainer2 = MinimalTrainer(_base_config(max_epochs=2, checkpoint_dir=str(tmp_path)))
        trainer2.fit(ckpt_path=ckpt)
        assert trainer2.current_epoch == 2

    def test_resume_continues_accumulating_global_step(self, tmp_path):
        """Resumed global_step (6) plus one more epoch of 6 steps equals 12."""
        trainer1 = MinimalTrainer(_base_config(max_epochs=1, checkpoint_dir=str(tmp_path)))
        trainer1.fit()
        ckpt = os.path.join(str(tmp_path), "epoch-0001.ckpt")

        trainer2 = MinimalTrainer(_base_config(max_epochs=2, checkpoint_dir=str(tmp_path)))
        trainer2.fit(ckpt_path=ckpt)
        assert trainer2.global_step == 12


class TestValFrequency:
    """Validation is triggered at the correct subset of epochs."""

    def test_val_called_every_epoch_when_frequency_1(self, tmp_path):
        """val_frequency=1 triggers validation every epoch; call count == max_epochs."""
        trainer = MinimalTrainer(
            _base_config(max_epochs=3, val_frequency=1, checkpoint_dir=str(tmp_path))
        )
        trainer.fit()
        assert trainer.val_call_count == 3

    def test_val_called_at_even_epochs_when_frequency_2(self, tmp_path):
        """val_frequency=2 over 4 epochs validates at epochs 0 and 2 only."""
        trainer = MinimalTrainer(
            _base_config(max_epochs=4, val_frequency=2, checkpoint_dir=str(tmp_path))
        )
        trainer.fit()
        assert trainer.val_call_count == 2


class TestValMetricPrefix:
    """Validation metrics returned by _val_loop carry the 'val_' prefix."""

    def test_all_keys_start_with_val_prefix(self, tmp_path):
        """Every key in the returned metrics dict starts with 'val_'."""
        trainer = MinimalTrainer(_base_config(max_epochs=1, checkpoint_dir=str(tmp_path)))
        trainer.fit()
        for key in trainer.last_val_metrics:
            assert key.startswith("val_")

    def test_val_loss_is_finite(self, tmp_path):
        """val_loss is a finite scalar tensor after one epoch."""
        trainer = MinimalTrainer(_base_config(max_epochs=1, checkpoint_dir=str(tmp_path)))
        trainer.fit()
        assert torch.isfinite(trainer.last_val_metrics["val_loss"])


class TestCheckpointFrequency:
    """Checkpoint files are created at the correct epochs."""

    def test_checkpoint_saved_every_epoch_when_frequency_1(self, tmp_path):
        """checkpoint_frequency=1 creates one file per epoch."""
        trainer = MinimalTrainer(
            _base_config(max_epochs=3, checkpoint_frequency=1, checkpoint_dir=str(tmp_path))
        )
        trainer.fit()
        for epoch in [1, 2, 3]:
            assert os.path.exists(os.path.join(str(tmp_path), f"epoch-{epoch:04d}.ckpt"))

    def test_checkpoint_skips_epoch_1_saves_epoch_2_and_last(self, tmp_path):
        """checkpoint_frequency=2 skips epoch 1, saves epoch 2 and epoch 3 (last)."""
        trainer = MinimalTrainer(
            _base_config(max_epochs=3, checkpoint_frequency=2, checkpoint_dir=str(tmp_path))
        )
        trainer.fit()
        assert not os.path.exists(os.path.join(str(tmp_path), "epoch-0001.ckpt"))
        assert os.path.exists(os.path.join(str(tmp_path), "epoch-0002.ckpt"))
        assert os.path.exists(os.path.join(str(tmp_path), "epoch-0003.ckpt"))
