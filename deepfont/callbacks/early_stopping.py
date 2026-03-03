"""Early-stopping callback for Lightning Fabric trainers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStoppingCallbackConfig:
    """Hydra structured config for :class:`EarlyStoppingCallback`.

    Register with Hydra's ConfigStore via
    :func:`~deepfont.callbacks.config_store.register_callback_configs` to use
    as a config group default.
    """

    _target_: str = "deepfont.callbacks.EarlyStoppingCallback"
    monitor: str = "val_loss"
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"
    verbose: bool = True


class EarlyStoppingCallback:
    """Stop training when a monitored metric stops improving.

    Checks the monitored metric at the end of every validation epoch.  If the
    metric does not improve by at least ``min_delta`` for ``patience``
    consecutive validation epochs, ``trainer.should_stop`` is set to ``True``,
    causing the training loop to exit cleanly at the end of that epoch.

    Args:
        monitor: Key to watch in the ``val_metrics`` dict passed to
            ``on_validation_epoch_end``.  Typically ``"val_loss"`` or
            ``"val_acc"``.
        patience: Number of validation epochs with no improvement before
            training is stopped.
        min_delta: Minimum change in the monitored metric to qualify as
            improvement.  Changes smaller than this threshold are ignored.
        mode: ``"min"`` (lower is better, e.g. loss) or ``"max"`` (higher is
            better, e.g. accuracy).
        verbose: When ``True``, prints a message via ``trainer.fabric.print``
            on each patience increment and when training is stopped.

    Example::

        from deepfont.callbacks import EarlyStoppingCallback
        from deepfont.trainer import PretrainTrainer, PretrainConfig

        trainer = PretrainTrainer(
            config=PretrainConfig(...),
            callbacks=[EarlyStoppingCallback(monitor="val_loss", patience=10)],
        )
        trainer.fit()
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
        verbose: bool = True,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self._best: float = float("inf") if mode == "min" else float("-inf")
        self._wait: int = 0

    def on_fit_start(self, trainer) -> None:
        """Reset internal state at the start of each fit() call."""
        self._best = float("inf") if self.mode == "min" else float("-inf")
        self._wait = 0

    def on_validation_epoch_end(self, trainer, val_metrics) -> None:
        """Check for improvement and update the patience counter."""
        if self.monitor not in val_metrics:
            trainer.fabric.print(
                f"[EarlyStopping] monitored key '{self.monitor}' not found in "
                f"val_metrics (available: {list(val_metrics.keys())}). Skipping."
            )
            return

        current: float = val_metrics[self.monitor].item()
        improved = (
            current < self._best - self.min_delta
            if self.mode == "min"
            else current > self._best + self.min_delta
        )

        if improved:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1
            if self.verbose:
                trainer.fabric.print(
                    f"[EarlyStopping] No improvement in '{self.monitor}' "
                    f"({self._wait}/{self.patience}). Best: {self._best:.6f}"
                )
            if self._wait >= self.patience:
                if self.verbose:
                    trainer.fabric.print(
                        f"[EarlyStopping] Stopping training. '{self.monitor}' "
                        f"has not improved for {self.patience} epochs."
                    )
                trainer.should_stop = True
