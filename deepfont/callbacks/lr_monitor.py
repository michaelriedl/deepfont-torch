"""Learning-rate monitor callback for Lightning Fabric trainers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LearningRateMonitorCallbackConfig:
    """Hydra structured config for :class:`LearningRateMonitorCallback`.

    Register with Hydra's ConfigStore via
    :func:`~deepfont.callbacks.config_store.register_callback_configs` to use
    as a config group default.
    """

    _target_: str = "deepfont.callbacks.LearningRateMonitorCallback"
    log_momentum: bool = False


class LearningRateMonitorCallback:
    """Log the current learning rate(s) to the Fabric logger each epoch.

    Reads LR values from the optimizer's ``param_groups`` at the start of
    every training epoch and forwards them to ``trainer.fabric.log_dict()``.
    If the optimizer has a single parameter group the key is ``"lr"``; for
    multiple groups the keys are ``"lr_group_0"``, ``"lr_group_1"``, etc.

    Args:
        log_momentum: When ``True``, also log the momentum (``betas[0]`` for
            Adam-family optimizers, ``momentum`` for SGD) under keys
            ``"momentum"`` / ``"momentum_group_{i}"``.

    Example::

        from deepfont.callbacks import LearningRateMonitorCallback

        cb = LearningRateMonitorCallback(log_momentum=True)
    """

    def __init__(self, log_momentum: bool = False) -> None:
        self.log_momentum = log_momentum

    def on_train_epoch_start(self, trainer) -> None:
        """Read LR from the optimizer and log it."""
        optimizer = trainer.optimizer
        if optimizer is None:
            return

        param_groups = optimizer.param_groups
        log_dict: dict[str, float] = {}

        if len(param_groups) == 1:
            log_dict["lr"] = param_groups[0]["lr"]
            if self.log_momentum:
                log_dict["momentum"] = self._get_momentum(param_groups[0])
        else:
            for i, pg in enumerate(param_groups):
                log_dict[f"lr_group_{i}"] = pg["lr"]
                if self.log_momentum:
                    log_dict[f"momentum_group_{i}"] = self._get_momentum(pg)

        trainer.fabric.log_dict(log_dict, step=trainer.global_step)

    @staticmethod
    def _get_momentum(param_group: dict) -> float:
        """Extract momentum from a param group, handling Adam and SGD."""
        if "betas" in param_group:
            # Adam, AdamW, etc.
            return float(param_group["betas"][0])
        return float(param_group.get("momentum", 0.0))
