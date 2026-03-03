"""Best-model checkpoint callback for Lightning Fabric trainers."""

from __future__ import annotations

import os
from typing import Any
from dataclasses import dataclass


@dataclass
class ModelCheckpointCallbackConfig:
    """Hydra structured config for :class:`ModelCheckpointCallback`.

    Register with Hydra's ConfigStore via
    :func:`~deepfont.callbacks.config_store.register_callback_configs` to use
    as a config group default.
    """

    _target_: str = "deepfont.callbacks.ModelCheckpointCallback"
    monitor: str = "val_loss"
    mode: str = "min"
    save_top_k: int = 1
    filename: str = "best"
    verbose: bool = True


class ModelCheckpointCallback:
    """Save the best-N checkpoints based on a monitored validation metric.

    At the end of every validation epoch, the callback compares the monitored
    metric against the current top-k scores.  If the new score enters the
    top-k, a checkpoint is saved and the worst checkpoint (if the list exceeds
    ``save_top_k``) is deleted from disk.

    Checkpoints are saved alongside the epoch-based checkpoints produced by
    :meth:`~deepfont.trainer.base.BaseTrainer._save_checkpoint`.  The filename
    template is::

        {filename}-epoch={epoch:04d}-{monitor}={value:.4f}.ckpt

    Saved inside ``trainer.config.checkpoint_dir``.

    Args:
        monitor: Metric key to watch in ``val_metrics``.
        mode: ``"min"`` (lower is better) or ``"max"`` (higher is better).
        save_top_k: Maximum number of best checkpoints to keep on disk.
            When a new best is found and this limit is already reached, the
            worst checkpoint in the tracked set is removed.
        filename: Prefix for the checkpoint filename.
        verbose: Print a message via ``trainer.fabric.print`` when a new best
            checkpoint is saved.

    Example::

        from deepfont.callbacks import ModelCheckpointCallback

        cb = ModelCheckpointCallback(monitor="val_acc", mode="max", save_top_k=3)
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        save_top_k: int = 1,
        filename: str = "best",
        verbose: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        if save_top_k < 1:
            raise ValueError(f"save_top_k must be >= 1, got {save_top_k}")

        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.filename = filename
        self.verbose = verbose

        # List of (score, filepath) kept in ascending order.
        # For "min" mode the worst entry (highest score) is at index -1.
        # For "max" mode the worst entry (lowest score) is at index 0.
        self._top_k: list[tuple[float, str]] = []

    def on_fit_start(self, trainer) -> None:
        """Reset tracked checkpoints at the start of each fit() call."""
        self._top_k = []

    def on_validation_epoch_end(self, trainer, val_metrics) -> None:
        """Save a checkpoint if the current metric enters the top-k."""
        if self.monitor not in val_metrics:
            trainer.fabric.print(
                f"[ModelCheckpoint] monitored key '{self.monitor}' not found in "
                f"val_metrics (available: {list(val_metrics.keys())}). Skipping."
            )
            return

        current: float = val_metrics[self.monitor].item()

        # Determine whether current score beats the worst in the top-k list.
        # _top_k is kept sorted ascending by score.
        # For "min" mode, lower is better → the worst entry has the highest
        # score → it sits at index -1 of the ascending list.
        # For "max" mode, higher is better → the worst entry has the lowest
        # score → it sits at index 0 of the ascending list.
        is_full = len(self._top_k) >= self.save_top_k
        if is_full:
            if self.mode == "min":
                worst_score, _ = self._top_k[-1]  # highest score = worst for "min"
                enters_top_k = current < worst_score
            else:
                worst_score, _ = self._top_k[0]  # lowest score = worst for "max"
                enters_top_k = current > worst_score
        else:
            enters_top_k = True

        if not enters_top_k:
            return

        # Only rank-0 writes files to avoid races in distributed training.
        if not trainer.fabric.is_global_zero:
            return

        os.makedirs(trainer.config.checkpoint_dir, exist_ok=True)

        ckpt_name = (
            f"{self.filename}-epoch={trainer.current_epoch:04d}-{self.monitor}={current:.4f}.ckpt"
        )
        path = os.path.join(trainer.config.checkpoint_dir, ckpt_name)

        # Build state dict from objects stored on the trainer.
        save_state: dict[str, Any] = {
            "model": trainer.model,
            "optimizer": trainer.optimizer,
            "global_step": trainer.global_step,
            "current_epoch": trainer.current_epoch,
        }
        if trainer.scheduler is not None:
            save_state["scheduler"] = trainer.scheduler

        trainer.fabric.save(path, save_state)

        if self.verbose:
            trainer.fabric.print(
                f"[ModelCheckpoint] Saved new best → {path} ({self.monitor}={current:.6f})"
            )

        # Insert new entry and keep list sorted ascending by score.
        self._top_k.append((current, path))
        self._top_k.sort(key=lambda x: x[0])  # ascending
        if self.mode == "max":
            # For "max" mode the *lowest* score at index 0 is the worst.
            pass  # sort ascending is already correct — worst at [0]

        # If we now exceed the top-k limit, evict the worst entry.
        if len(self._top_k) > self.save_top_k:
            if self.mode == "min":
                worst_score, worst_path = self._top_k.pop(-1)  # highest = worst for "min"
            else:
                worst_score, worst_path = self._top_k.pop(0)  # lowest = worst for "max"
            if os.path.exists(worst_path):
                os.remove(worst_path)
                if self.verbose:
                    trainer.fabric.print(f"[ModelCheckpoint] Removed old checkpoint: {worst_path}")
