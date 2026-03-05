"""Tests for deepfont.callbacks.*

Each callback is tested in isolation using a lightweight mock trainer that
exposes only the attributes the callbacks depend on:

    trainer.fabric          – MagicMock with explicit attribute overrides:
                                  fabric.is_global_zero  (bool)
                                  fabric.device          (torch.device)
                                  fabric.print           (MagicMock)
                                  fabric.log_dict        (MagicMock)
                                  fabric.save            (MagicMock)
    trainer.should_stop     – bool flag
    trainer.current_epoch   – int
    trainer.global_step     – int
    trainer.model           – nn.Module | None
    trainer.optimizer       – Optimizer | None
    trainer.scheduler       – LRScheduler | None
    trainer.config          – SimpleNamespace(checkpoint_dir=str)

Test classes:
    TestEarlyStoppingCallbackConfig     -- default field values and _target_
    TestEarlyStoppingCallback           -- construction, fit_start reset,
                                          improvement logic, patience,
                                          min_delta, mode=max, missing key,
                                          verbose=False
    TestGradientNormMonitorCallbackConfig
    TestGradientNormMonitorCallback     -- step-frequency gating, model=None
                                          guard, L1/L2 norm correctness,
                                          no-grad params, step logged
    TestLearningRateMonitorCallbackConfig
    TestLearningRateMonitorCallback     -- single/multi param group, Adam/SGD
                                          momentum, no optimizer, step logged,
                                          _get_momentum static method
    TestModelCheckpointCallbackConfig
    TestModelCheckpointCallback         -- construction, fit_start reset,
                                          save_top_k=1 and =3 for both modes,
                                          eviction correctness (the mode=min
                                          bug that was fixed), missing key,
                                          not-global-zero, filename format,
                                          scheduler presence in state dict
    TestReconstructionVisualizerCallbackConfig
    TestReconstructionVisualizerCallback -- _is_save_epoch, batch capture
                                           logic, file output, value_range
                                           clamping (skipped if torchvision
                                           is not installed)
"""

from __future__ import annotations

import importlib.util
import math
import os
import tempfile
import types
import unittest
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

# Shared helpers

_TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None


def _make_fabric(*, is_global_zero: bool = True):
    """Return a MagicMock that looks like a minimal Lightning Fabric instance."""
    fabric = MagicMock()
    fabric.is_global_zero = is_global_zero
    fabric.device = torch.device("cpu")
    return fabric


def _make_trainer(
    *,
    is_global_zero: bool = True,
    current_epoch: int = 0,
    global_step: int = 0,
    model=None,
    optimizer=None,
    scheduler=None,
    checkpoint_dir: str = "/tmp/deepfont_test_ckpts",
):
    """Return a SimpleNamespace that satisfies the callback protocol."""
    return types.SimpleNamespace(
        fabric=_make_fabric(is_global_zero=is_global_zero),
        should_stop=False,
        current_epoch=current_epoch,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=types.SimpleNamespace(checkpoint_dir=checkpoint_dir),
    )


def _tv(x: float) -> torch.Tensor:
    """Wrap a scalar float as a 0-d tensor (as val_metrics values appear)."""
    return torch.tensor(x)


# EarlyStoppingCallback


class TestEarlyStoppingCallbackConfig(unittest.TestCase):
    def test_defaults(self):
        from deepfont.callbacks import EarlyStoppingCallbackConfig

        cfg = EarlyStoppingCallbackConfig()
        self.assertEqual(cfg._target_, "deepfont.callbacks.EarlyStoppingCallback")
        self.assertEqual(cfg.monitor, "val_loss")
        self.assertEqual(cfg.patience, 5)
        self.assertEqual(cfg.min_delta, 0.0)
        self.assertEqual(cfg.mode, "min")
        self.assertTrue(cfg.verbose)


class TestEarlyStoppingCallback(unittest.TestCase):
    def setUp(self):
        from deepfont.callbacks import EarlyStoppingCallback

        self.CB = EarlyStoppingCallback

    # construction

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.CB(mode="invalid")

    def test_valid_modes_do_not_raise(self):
        self.CB(mode="min")
        self.CB(mode="max")

    def test_initial_state_min(self):
        cb = self.CB(mode="min")
        self.assertEqual(cb._best, float("inf"))
        self.assertEqual(cb._wait, 0)

    def test_initial_state_max(self):
        cb = self.CB(mode="max")
        self.assertEqual(cb._best, float("-inf"))
        self.assertEqual(cb._wait, 0)

    # on_fit_start

    def test_on_fit_start_resets_min(self):
        cb = self.CB(mode="min", patience=3)
        cb._best = 0.1
        cb._wait = 2
        cb.on_fit_start(_make_trainer())
        self.assertEqual(cb._best, float("inf"))
        self.assertEqual(cb._wait, 0)

    def test_on_fit_start_resets_max(self):
        cb = self.CB(mode="max", patience=3)
        cb._best = 0.9
        cb._wait = 2
        cb.on_fit_start(_make_trainer())
        self.assertEqual(cb._best, float("-inf"))
        self.assertEqual(cb._wait, 0)

    def test_on_fit_start_resets_after_partial_run(self):
        """Re-calling fit() on the same callback clears stale state."""
        cb = self.CB(mode="min", patience=2)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.5)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.6)})  # wait=1
        cb.on_fit_start(_make_trainer())
        self.assertEqual(cb._wait, 0)
        self.assertEqual(cb._best, float("inf"))

    # mode=min: improvement

    def test_first_metric_sets_best(self):
        cb = self.CB(mode="min", patience=3)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.5)})
        self.assertAlmostEqual(cb._best, 0.5)
        self.assertEqual(cb._wait, 0)

    def test_improving_metric_updates_best_and_resets_wait(self):
        cb = self.CB(mode="min", patience=3)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.5)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.3)})
        self.assertAlmostEqual(cb._best, 0.3)
        self.assertEqual(cb._wait, 0)

    def test_no_improvement_increments_wait(self):
        cb = self.CB(mode="min", patience=3)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.5)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.6)})
        self.assertEqual(cb._wait, 1)
        self.assertFalse(trainer.should_stop)

    def test_stops_exactly_at_patience(self):
        cb = self.CB(mode="min", patience=3)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.5)})
        for _ in range(3):
            cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.6)})
        self.assertTrue(trainer.should_stop)

    def test_does_not_stop_before_patience(self):
        cb = self.CB(mode="min", patience=3)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.5)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.6)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.6)})
        self.assertFalse(trainer.should_stop)
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.6)})
        self.assertTrue(trainer.should_stop)

    def test_improvement_after_stall_resets_wait_and_prevents_stop(self):
        cb = self.CB(mode="min", patience=3)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.5)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.6)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.6)})
        self.assertEqual(cb._wait, 2)
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.4)})
        self.assertEqual(cb._wait, 0)
        self.assertFalse(trainer.should_stop)

    # min_delta

    def test_min_delta_change_below_threshold_not_improvement(self):
        cb = self.CB(mode="min", patience=3, min_delta=0.1)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.5)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.45)})  # Δ=0.05 < 0.1
        self.assertEqual(cb._wait, 1)

    def test_min_delta_change_above_threshold_is_improvement(self):
        cb = self.CB(mode="min", patience=3, min_delta=0.1)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.5)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.38)})  # Δ=0.12 > 0.1
        self.assertEqual(cb._wait, 0)
        self.assertAlmostEqual(cb._best, 0.38, places=5)

    # mode=max

    def test_mode_max_improvement_increases_best(self):
        cb = self.CB(mode="max", patience=3, monitor="val_acc")
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.5)})
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.7)})
        self.assertAlmostEqual(cb._best, 0.7)
        self.assertEqual(cb._wait, 0)

    def test_mode_max_no_improvement_increments_wait(self):
        cb = self.CB(mode="max", patience=2, monitor="val_acc")
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.7)})
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.5)})
        self.assertEqual(cb._wait, 1)

    def test_mode_max_stops_after_patience(self):
        cb = self.CB(mode="max", patience=2, monitor="val_acc")
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.7)})
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.5)})
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.5)})
        self.assertTrue(trainer.should_stop)

    # missing key

    def test_missing_monitor_key_does_not_change_state(self):
        cb = self.CB(monitor="val_loss", patience=3)
        trainer = _make_trainer()
        original_best = cb._best
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.5)})
        self.assertEqual(cb._best, original_best)
        self.assertEqual(cb._wait, 0)

    def test_missing_monitor_key_prints_warning(self):
        cb = self.CB(monitor="val_loss", patience=3, verbose=True)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.5)})
        trainer.fabric.print.assert_called_once()

    # verbose=False

    def test_verbose_false_no_print_on_stall_or_stop(self):
        cb = self.CB(mode="min", patience=2, verbose=False)
        trainer = _make_trainer()
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.5)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.6)})
        cb.on_validation_epoch_end(trainer, {"val_loss": _tv(0.6)})
        self.assertTrue(trainer.should_stop)
        trainer.fabric.print.assert_not_called()


# GradientNormMonitorCallback


class TestGradientNormMonitorCallbackConfig(unittest.TestCase):
    def test_defaults(self):
        from deepfont.callbacks import GradientNormMonitorCallbackConfig

        cfg = GradientNormMonitorCallbackConfig()
        self.assertEqual(cfg._target_, "deepfont.callbacks.GradientNormMonitorCallback")
        self.assertAlmostEqual(cfg.norm_type, 2.0)
        self.assertEqual(cfg.log_every_n_steps, 10)


class TestGradientNormMonitorCallback(unittest.TestCase):
    def setUp(self):
        from deepfont.callbacks import GradientNormMonitorCallback

        self.CB = GradientNormMonitorCallback

    @staticmethod
    def _model_with_grad(grad_values: list[list[float]]) -> nn.Module:
        """Build a Linear(2,2,bias=False) with manually set gradients."""
        model = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.zeros(2, 2))
        model.weight.grad = torch.tensor(grad_values, dtype=torch.float32)
        return model

    # step-frequency gating

    def test_skips_logging_on_non_multiple_step(self):
        cb = self.CB(log_every_n_steps=10)
        model = self._model_with_grad([[1.0, 0.0], [0.0, 1.0]])
        trainer = _make_trainer(global_step=5, model=model)
        cb.on_before_optimizer_step(trainer, None)
        trainer.fabric.log_dict.assert_not_called()

    def test_logs_on_multiple_step(self):
        cb = self.CB(log_every_n_steps=10)
        model = self._model_with_grad([[1.0, 0.0], [0.0, 1.0]])
        trainer = _make_trainer(global_step=10, model=model)
        cb.on_before_optimizer_step(trainer, None)
        trainer.fabric.log_dict.assert_called_once()

    def test_logs_at_step_zero(self):
        cb = self.CB(log_every_n_steps=10)
        model = self._model_with_grad([[1.0, 0.0], [0.0, 1.0]])
        trainer = _make_trainer(global_step=0, model=model)
        cb.on_before_optimizer_step(trainer, None)
        trainer.fabric.log_dict.assert_called_once()

    def test_log_every_n_steps_1_logs_every_step(self):
        cb = self.CB(log_every_n_steps=1)
        model = self._model_with_grad([[1.0, 0.0], [0.0, 1.0]])
        for step in range(5):
            trainer = _make_trainer(global_step=step, model=model)
            cb.on_before_optimizer_step(trainer, None)
            trainer.fabric.log_dict.assert_called_once()

    # model=None guard

    def test_skips_when_model_is_none(self):
        cb = self.CB(log_every_n_steps=1)
        trainer = _make_trainer(global_step=0, model=None)
        cb.on_before_optimizer_step(trainer, None)
        trainer.fabric.log_dict.assert_not_called()

    # correctness of grad_norm value

    def test_l2_norm_value(self):
        """grad = [[1,0],[0,1]] → global L2 norm = sqrt(1²+0²+0²+1²) = √2."""
        cb = self.CB(norm_type=2.0, log_every_n_steps=1)
        model = self._model_with_grad([[1.0, 0.0], [0.0, 1.0]])
        trainer = _make_trainer(global_step=0, model=model)
        cb.on_before_optimizer_step(trainer, None)
        logged = trainer.fabric.log_dict.call_args[0][0]
        self.assertAlmostEqual(logged["grad_norm"], math.sqrt(2), places=5)

    def test_l1_norm_value(self):
        """grad = [[1,2],[3,4]] → global L1 norm = 1+2+3+4 = 10."""
        cb = self.CB(norm_type=1.0, log_every_n_steps=1)
        model = self._model_with_grad([[1.0, 2.0], [3.0, 4.0]])
        trainer = _make_trainer(global_step=0, model=model)
        cb.on_before_optimizer_step(trainer, None)
        logged = trainer.fabric.log_dict.call_args[0][0]
        self.assertAlmostEqual(logged["grad_norm"], 10.0, places=5)

    def test_zero_grad_logs_zero(self):
        cb = self.CB(norm_type=2.0, log_every_n_steps=1)
        model = self._model_with_grad([[0.0, 0.0], [0.0, 0.0]])
        trainer = _make_trainer(global_step=0, model=model)
        cb.on_before_optimizer_step(trainer, None)
        logged = trainer.fabric.log_dict.call_args[0][0]
        self.assertAlmostEqual(logged["grad_norm"], 0.0, places=5)

    def test_no_grad_params_logs_zero(self):
        """When no parameter has a .grad, the global norm is 0."""
        cb = self.CB(norm_type=2.0, log_every_n_steps=1)
        model = nn.Linear(2, 2, bias=False)  # grads are None until backward()
        trainer = _make_trainer(global_step=0, model=model)
        cb.on_before_optimizer_step(trainer, None)
        logged = trainer.fabric.log_dict.call_args[0][0]
        self.assertAlmostEqual(logged["grad_norm"], 0.0, places=5)

    def test_logged_step_matches_global_step(self):
        cb = self.CB(norm_type=2.0, log_every_n_steps=10)
        model = self._model_with_grad([[1.0, 0.0], [0.0, 1.0]])
        trainer = _make_trainer(global_step=30, model=model)
        cb.on_before_optimizer_step(trainer, None)
        _, kwargs = trainer.fabric.log_dict.call_args
        self.assertEqual(kwargs["step"], 30)


# LearningRateMonitorCallback


class TestLearningRateMonitorCallbackConfig(unittest.TestCase):
    def test_defaults(self):
        from deepfont.callbacks import LearningRateMonitorCallbackConfig

        cfg = LearningRateMonitorCallbackConfig()
        self.assertEqual(cfg._target_, "deepfont.callbacks.LearningRateMonitorCallback")
        self.assertFalse(cfg.log_momentum)


class TestLearningRateMonitorCallback(unittest.TestCase):
    def setUp(self):
        from deepfont.callbacks import LearningRateMonitorCallback

        self.CB = LearningRateMonitorCallback

    # no optimizer

    def test_no_optimizer_returns_without_logging(self):
        cb = self.CB()
        trainer = _make_trainer(optimizer=None)
        cb.on_train_epoch_start(trainer)
        trainer.fabric.log_dict.assert_not_called()

    # single param group

    def test_single_group_logs_lr(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = _make_trainer(optimizer=optimizer, global_step=5)
        cb = self.CB(log_momentum=False)
        cb.on_train_epoch_start(trainer)
        trainer.fabric.log_dict.assert_called_once_with({"lr": pytest.approx(1e-3)}, step=5)

    def test_single_group_log_momentum_false_no_momentum_key(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = _make_trainer(optimizer=optimizer)
        cb = self.CB(log_momentum=False)
        cb.on_train_epoch_start(trainer)
        logged = trainer.fabric.log_dict.call_args[0][0]
        self.assertNotIn("momentum", logged)

    def test_single_group_log_momentum_adam(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        trainer = _make_trainer(optimizer=optimizer)
        cb = self.CB(log_momentum=True)
        cb.on_train_epoch_start(trainer)
        logged = trainer.fabric.log_dict.call_args[0][0]
        self.assertIn("momentum", logged)
        self.assertAlmostEqual(logged["momentum"], 0.9, places=6)

    def test_single_group_log_momentum_sgd(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
        trainer = _make_trainer(optimizer=optimizer)
        cb = self.CB(log_momentum=True)
        cb.on_train_epoch_start(trainer)
        logged = trainer.fabric.log_dict.call_args[0][0]
        self.assertIn("momentum", logged)
        self.assertAlmostEqual(logged["momentum"], 0.95, places=6)

    # multiple param groups

    def test_multiple_groups_logs_lr_group_keys(self):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(
            [{"params": [model.weight], "lr": 1e-3}, {"params": [model.bias], "lr": 1e-4}]
        )
        trainer = _make_trainer(optimizer=optimizer)
        cb = self.CB(log_momentum=False)
        cb.on_train_epoch_start(trainer)
        logged = trainer.fabric.log_dict.call_args[0][0]
        self.assertAlmostEqual(logged["lr_group_0"], 1e-3, places=7)
        self.assertAlmostEqual(logged["lr_group_1"], 1e-4, places=7)
        self.assertNotIn("lr", logged)

    def test_multiple_groups_log_momentum(self):
        model = nn.Linear(4, 2)
        optimizer = torch.optim.Adam(
            [{"params": [model.weight], "lr": 1e-3}, {"params": [model.bias], "lr": 1e-4}],
            betas=(0.85, 0.999),
        )
        trainer = _make_trainer(optimizer=optimizer)
        cb = self.CB(log_momentum=True)
        cb.on_train_epoch_start(trainer)
        logged = trainer.fabric.log_dict.call_args[0][0]
        self.assertIn("momentum_group_0", logged)
        self.assertIn("momentum_group_1", logged)
        self.assertAlmostEqual(logged["momentum_group_0"], 0.85, places=6)

    # _get_momentum static method

    def test_get_momentum_adam(self):
        from deepfont.callbacks import LearningRateMonitorCallback

        pg = {"betas": (0.9, 0.999), "lr": 0.001}
        self.assertAlmostEqual(LearningRateMonitorCallback._get_momentum(pg), 0.9)

    def test_get_momentum_sgd(self):
        from deepfont.callbacks import LearningRateMonitorCallback

        pg = {"momentum": 0.95, "lr": 0.01}
        self.assertAlmostEqual(LearningRateMonitorCallback._get_momentum(pg), 0.95)

    def test_get_momentum_no_key_returns_zero(self):
        from deepfont.callbacks import LearningRateMonitorCallback

        pg = {"lr": 0.01}
        self.assertAlmostEqual(LearningRateMonitorCallback._get_momentum(pg), 0.0)

    # step logged

    def test_logged_step_matches_global_step(self):
        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = _make_trainer(optimizer=optimizer, global_step=42)
        cb = self.CB()
        cb.on_train_epoch_start(trainer)
        _, kwargs = trainer.fabric.log_dict.call_args
        self.assertEqual(kwargs["step"], 42)


# ModelCheckpointCallback


class TestModelCheckpointCallbackConfig(unittest.TestCase):
    def test_defaults(self):
        from deepfont.callbacks import ModelCheckpointCallbackConfig

        cfg = ModelCheckpointCallbackConfig()
        self.assertEqual(cfg._target_, "deepfont.callbacks.ModelCheckpointCallback")
        self.assertEqual(cfg.monitor, "val_loss")
        self.assertEqual(cfg.mode, "min")
        self.assertEqual(cfg.save_top_k, 1)
        self.assertEqual(cfg.filename, "best")
        self.assertTrue(cfg.verbose)


class TestModelCheckpointCallback(unittest.TestCase):
    def setUp(self):
        from deepfont.callbacks import ModelCheckpointCallback

        self.CB = ModelCheckpointCallback
        self._tmp = tempfile.TemporaryDirectory()
        self.ckpt_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _make_ckpt_trainer(
        self,
        *,
        current_epoch: int = 0,
        global_step: int = 0,
        is_global_zero: bool = True,
        model=None,
        scheduler=None,
    ):
        """Create a trainer whose fabric.save actually writes an empty file."""
        trainer = _make_trainer(
            current_epoch=current_epoch,
            global_step=global_step,
            is_global_zero=is_global_zero,
            checkpoint_dir=self.ckpt_dir,
            model=model or nn.Linear(2, 2),
            scheduler=scheduler,
        )

        def _fake_save(path, state):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w").close()

        trainer.fabric.save.side_effect = _fake_save
        return trainer

    def _trigger(self, cb, trainer, score: float, monitor: str = "val_loss"):
        cb.on_validation_epoch_end(trainer, {monitor: _tv(score)})

    # construction

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            self.CB(mode="invalid")

    def test_save_top_k_zero_raises(self):
        with self.assertRaises(ValueError):
            self.CB(save_top_k=0)

    # on_fit_start

    def test_on_fit_start_clears_top_k(self):
        cb = self.CB(save_top_k=3)
        cb._top_k = [(0.5, "p1"), (0.3, "p2")]
        cb.on_fit_start(_make_trainer())
        self.assertEqual(cb._top_k, [])

    # save_top_k=1, mode=min

    def test_first_checkpoint_is_always_saved(self):
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=1)
        trainer = self._make_ckpt_trainer()
        self._trigger(cb, trainer, 0.5)
        trainer.fabric.save.assert_called_once()

    def test_better_score_replaces_checkpoint(self):
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=1)
        trainer = self._make_ckpt_trainer()
        self._trigger(cb, trainer, 0.5)
        self._trigger(cb, trainer, 0.3)  # better (lower)
        self.assertEqual(trainer.fabric.save.call_count, 2)
        self.assertEqual(len(cb._top_k), 1)
        self.assertAlmostEqual(cb._top_k[0][0], 0.3)

    def test_worse_score_is_skipped(self):
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=1)
        trainer = self._make_ckpt_trainer()
        self._trigger(cb, trainer, 0.5)
        self._trigger(cb, trainer, 0.7)  # worse (higher)
        self.assertEqual(trainer.fabric.save.call_count, 1)
        self.assertEqual(len(cb._top_k), 1)
        self.assertAlmostEqual(cb._top_k[0][0], 0.5)

    def test_worse_checkpoint_is_deleted_from_disk(self):
        """When a better checkpoint arrives, the previous one is removed."""
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=1)
        trainer = self._make_ckpt_trainer()
        self._trigger(cb, trainer, 0.5)
        first_path = cb._top_k[0][1]
        self.assertTrue(os.path.exists(first_path))
        self._trigger(cb, trainer, 0.3)  # better → evicts the 0.5 checkpoint
        self.assertFalse(os.path.exists(first_path))

    # save_top_k=1, mode=max

    def test_mode_max_better_score_saved(self):
        cb = self.CB(monitor="val_acc", mode="max", save_top_k=1)
        trainer = self._make_ckpt_trainer()
        self._trigger(cb, trainer, 0.5, "val_acc")
        self._trigger(cb, trainer, 0.7, "val_acc")  # better (higher)
        self.assertEqual(len(cb._top_k), 1)
        self.assertAlmostEqual(cb._top_k[0][0], 0.7)

    def test_mode_max_worse_score_skipped(self):
        cb = self.CB(monitor="val_acc", mode="max", save_top_k=1)
        trainer = self._make_ckpt_trainer()
        self._trigger(cb, trainer, 0.7, "val_acc")
        self._trigger(cb, trainer, 0.4, "val_acc")  # worse (lower)
        self.assertEqual(len(cb._top_k), 1)
        self.assertAlmostEqual(cb._top_k[0][0], 0.7)

    def test_mode_max_worse_checkpoint_deleted(self):
        cb = self.CB(monitor="val_acc", mode="max", save_top_k=1)
        trainer = self._make_ckpt_trainer()
        self._trigger(cb, trainer, 0.5, "val_acc")
        first_path = cb._top_k[0][1]
        self._trigger(cb, trainer, 0.8, "val_acc")  # better → evicts 0.5
        self.assertFalse(os.path.exists(first_path))

    # save_top_k=3, mode=min

    def test_save_top_k_3_fills_without_eviction(self):
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=3)
        trainer = self._make_ckpt_trainer()
        for score in [0.5, 0.4, 0.7]:
            self._trigger(cb, trainer, score)
        self.assertEqual(len(cb._top_k), 3)
        self.assertEqual(trainer.fabric.save.call_count, 3)

    def test_save_top_k_3_evicts_worst_on_overflow(self):
        """The evicted checkpoint is the one with the highest (worst) loss."""
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=3)
        trainer = self._make_ckpt_trainer()
        for score in [0.5, 0.4, 0.7]:
            self._trigger(cb, trainer, score)
        # Identify the worst (highest) checkpoint path before the new entry.
        worst_path = max(cb._top_k, key=lambda x: x[0])[1]
        self.assertTrue(os.path.exists(worst_path))

        self._trigger(cb, trainer, 0.2)  # better than 0.7 → replaces it

        self.assertEqual(len(cb._top_k), 3)
        self.assertFalse(os.path.exists(worst_path))
        top_scores = [s for s, _ in cb._top_k]
        self.assertFalse(any(abs(s - 0.7) < 1e-5 for s in top_scores))
        self.assertTrue(any(abs(s - 0.2) < 1e-5 for s in top_scores))

    def test_save_top_k_3_keeps_k_best_scores(self):
        """After 5 scores only the 3 best (lowest) survive."""
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=3)
        trainer = self._make_ckpt_trainer()
        for score in [0.5, 0.4, 0.7, 0.3, 0.6]:
            self._trigger(cb, trainer, score)
        self.assertEqual(len(cb._top_k), 3)
        top_scores = sorted(s for s, _ in cb._top_k)
        # Best 3 of [0.3, 0.4, 0.5, 0.6, 0.7] are [0.3, 0.4, 0.5].
        for expected, actual in zip([0.3, 0.4, 0.5], top_scores):
            self.assertAlmostEqual(expected, actual, places=5)

    def test_save_top_k_3_mode_max_keeps_k_best_scores(self):
        """For mode=max, after 5 scores only the 3 highest survive."""
        cb = self.CB(monitor="val_acc", mode="max", save_top_k=3)
        trainer = self._make_ckpt_trainer()
        for score in [0.5, 0.4, 0.7, 0.3, 0.6]:
            self._trigger(cb, trainer, score, "val_acc")
        self.assertEqual(len(cb._top_k), 3)
        top_scores = sorted(s for s, _ in cb._top_k)
        # Best 3 of [0.3, 0.4, 0.5, 0.6, 0.7] are [0.5, 0.6, 0.7].
        for expected, actual in zip([0.5, 0.6, 0.7], top_scores):
            self.assertAlmostEqual(expected, actual, places=5)

    # missing monitor key

    def test_missing_monitor_key_skips_save(self):
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=1)
        trainer = self._make_ckpt_trainer()
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.9)})
        trainer.fabric.save.assert_not_called()

    def test_missing_monitor_key_prints_warning(self):
        cb = self.CB(monitor="val_loss", mode="min", verbose=True)
        trainer = self._make_ckpt_trainer()
        cb.on_validation_epoch_end(trainer, {"val_acc": _tv(0.9)})
        trainer.fabric.print.assert_called_once()

    # not global zero

    def test_not_global_zero_does_not_save(self):
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=1)
        trainer = self._make_ckpt_trainer(is_global_zero=False)
        self._trigger(cb, trainer, 0.5)
        trainer.fabric.save.assert_not_called()

    # filename format

    def test_checkpoint_filename_format(self):
        cb = self.CB(monitor="val_loss", mode="min", filename="best")
        trainer = self._make_ckpt_trainer(current_epoch=7)
        self._trigger(cb, trainer, 0.1234)
        self.assertEqual(len(cb._top_k), 1)
        ckpt_name = os.path.basename(cb._top_k[0][1])
        self.assertIn("best-epoch=0007-val_loss=0.1234", ckpt_name)

    # scheduler presence in state dict

    def test_scheduler_included_in_save_when_present(self):
        from torch.optim.lr_scheduler import StepLR

        model = nn.Linear(2, 2)
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = StepLR(optimizer, step_size=1)
        trainer = self._make_ckpt_trainer(model=model, scheduler=scheduler)
        trainer.optimizer = optimizer
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=1)
        self._trigger(cb, trainer, 0.5)
        state_dict = trainer.fabric.save.call_args[0][1]
        self.assertIn("scheduler", state_dict)

    def test_scheduler_excluded_from_save_when_absent(self):
        trainer = self._make_ckpt_trainer(scheduler=None)
        cb = self.CB(monitor="val_loss", mode="min", save_top_k=1)
        self._trigger(cb, trainer, 0.5)
        state_dict = trainer.fabric.save.call_args[0][1]
        self.assertNotIn("scheduler", state_dict)


# ReconstructionVisualizerCallback


class TestReconstructionVisualizerCallbackConfig(unittest.TestCase):
    def test_defaults(self):
        from deepfont.callbacks import ReconstructionVisualizerCallbackConfig

        cfg = ReconstructionVisualizerCallbackConfig()
        self.assertEqual(cfg._target_, "deepfont.callbacks.ReconstructionVisualizerCallback")
        self.assertEqual(cfg.save_every_n_epochs, 5)
        self.assertEqual(cfg.num_samples, 8)
        self.assertEqual(cfg.output_dir, "reconstructions")
        self.assertIsNone(cfg.value_range)


@unittest.skipUnless(_TORCHVISION_AVAILABLE, "torchvision not installed")
class TestReconstructionVisualizerCallback(unittest.TestCase):
    def setUp(self):
        from deepfont.callbacks import ReconstructionVisualizerCallback

        self.CB = ReconstructionVisualizerCallback
        self._tmp = tempfile.TemporaryDirectory()
        self.output_dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _make_rkv_trainer(self, *, current_epoch: int = 0, is_global_zero: bool = True):
        # nn.Identity() passes inputs through unchanged – perfect as a stub
        # reconstruction model.
        return _make_trainer(
            current_epoch=current_epoch,
            is_global_zero=is_global_zero,
            model=nn.Identity(),
        )

    def _batch(self, b: int = 4, h: int = 105, w: int = 105) -> torch.Tensor:
        return torch.rand(b, 1, h, w)

    # construction

    def test_initial_sample_inputs_is_none(self):
        cb = self.CB()
        self.assertIsNone(cb._sample_inputs)

    # _is_save_epoch

    def test_is_save_epoch_multiples_of_period(self):
        cb = self.CB(save_every_n_epochs=5)
        for epoch, expected in [
            (0, True),
            (5, True),
            (10, True),
            (1, False),
            (3, False),
            (4, False),
        ]:
            trainer = self._make_rkv_trainer(current_epoch=epoch)
            self.assertEqual(cb._is_save_epoch(trainer), expected, f"epoch={epoch}")

    # on_validation_batch_start

    def test_skips_non_zero_batch_idx(self):
        cb = self.CB(save_every_n_epochs=1, num_samples=4)
        trainer = self._make_rkv_trainer(current_epoch=0)
        cb.on_validation_batch_start(self._batch(), batch_idx=1, trainer=trainer)
        self.assertIsNone(cb._sample_inputs)

    def test_skips_non_save_epoch(self):
        cb = self.CB(save_every_n_epochs=5, num_samples=4)
        trainer = self._make_rkv_trainer(current_epoch=1)  # 1 % 5 ≠ 0
        cb.on_validation_batch_start(self._batch(), batch_idx=0, trainer=trainer)
        self.assertIsNone(cb._sample_inputs)

    def test_captures_first_batch_on_save_epoch(self):
        cb = self.CB(save_every_n_epochs=1, num_samples=4)
        trainer = self._make_rkv_trainer(current_epoch=0)
        cb.on_validation_batch_start(self._batch(b=8), batch_idx=0, trainer=trainer)
        self.assertIsNotNone(cb._sample_inputs)
        self.assertEqual(cb._sample_inputs.shape, (4, 1, 105, 105))

    def test_captures_correct_number_of_samples(self):
        cb = self.CB(save_every_n_epochs=1, num_samples=2)
        trainer = self._make_rkv_trainer(current_epoch=0)
        cb.on_validation_batch_start(self._batch(b=10), batch_idx=0, trainer=trainer)
        self.assertEqual(cb._sample_inputs.shape[0], 2)

    def test_captures_all_when_batch_smaller_than_num_samples(self):
        cb = self.CB(save_every_n_epochs=1, num_samples=16)
        trainer = self._make_rkv_trainer(current_epoch=0)
        cb.on_validation_batch_start(self._batch(b=4), batch_idx=0, trainer=trainer)
        self.assertEqual(cb._sample_inputs.shape[0], 4)

    # on_validation_epoch_end

    def test_skips_when_no_samples_captured(self):
        cb = self.CB(save_every_n_epochs=1, output_dir=self.output_dir)
        trainer = self._make_rkv_trainer()
        cb._sample_inputs = None
        cb.on_validation_epoch_end(trainer, {})
        self.assertEqual(len(os.listdir(self.output_dir)), 0)

    def test_skips_file_write_when_not_global_zero(self):
        cb = self.CB(save_every_n_epochs=1, output_dir=self.output_dir)
        trainer = self._make_rkv_trainer(is_global_zero=False)
        cb._sample_inputs = self._batch(b=4)
        cb.on_validation_epoch_end(trainer, {})
        self.assertIsNone(cb._sample_inputs)
        self.assertEqual(len(os.listdir(self.output_dir)), 0)

    def test_saves_png_file(self):
        cb = self.CB(save_every_n_epochs=1, num_samples=4, output_dir=self.output_dir)
        trainer = self._make_rkv_trainer(current_epoch=3)
        cb.on_validation_batch_start(self._batch(b=8), batch_idx=0, trainer=trainer)
        cb.on_validation_epoch_end(trainer, {})
        files = os.listdir(self.output_dir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith(".png"))

    def test_saved_filename_contains_epoch(self):
        cb = self.CB(save_every_n_epochs=1, num_samples=4, output_dir=self.output_dir)
        trainer = self._make_rkv_trainer(current_epoch=7)
        cb.on_validation_batch_start(self._batch(b=4), batch_idx=0, trainer=trainer)
        cb.on_validation_epoch_end(trainer, {})
        self.assertIn("epoch-0007.png", os.listdir(self.output_dir))

    def test_sample_inputs_cleared_after_epoch_end(self):
        cb = self.CB(save_every_n_epochs=1, num_samples=4, output_dir=self.output_dir)
        trainer = self._make_rkv_trainer(current_epoch=0)
        cb.on_validation_batch_start(self._batch(b=4), batch_idx=0, trainer=trainer)
        cb.on_validation_epoch_end(trainer, {})
        self.assertIsNone(cb._sample_inputs)

    def test_value_range_clamping_produces_file(self):
        """Out-of-range values should be clamped without error."""
        cb = self.CB(
            save_every_n_epochs=1,
            num_samples=4,
            output_dir=self.output_dir,
            value_range=(0.0, 1.0),
        )
        trainer = self._make_rkv_trainer(current_epoch=0)
        batch = torch.full((4, 1, 105, 105), 2.0)  # all values > 1.0
        cb.on_validation_batch_start(batch, batch_idx=0, trainer=trainer)
        cb.on_validation_epoch_end(trainer, {})
        self.assertEqual(len(os.listdir(self.output_dir)), 1)

    def test_no_file_produced_on_non_save_epoch(self):
        cb = self.CB(save_every_n_epochs=5, num_samples=4, output_dir=self.output_dir)
        trainer = self._make_rkv_trainer(current_epoch=1)  # not a save epoch
        cb.on_validation_batch_start(self._batch(b=4), batch_idx=0, trainer=trainer)
        cb.on_validation_epoch_end(trainer, {})
        self.assertEqual(len(os.listdir(self.output_dir)), 0)


if __name__ == "__main__":
    unittest.main()
