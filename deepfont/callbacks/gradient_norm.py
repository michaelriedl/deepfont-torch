"""Gradient-norm monitor callback for Lightning Fabric trainers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GradientNormMonitorCallbackConfig:
    """Hydra structured config for :class:`GradientNormMonitorCallback`.

    Register with Hydra's ConfigStore via
    :func:`~deepfont.callbacks.config_store.register_callback_configs` to use
    as a config group default.
    """

    _target_: str = "deepfont.callbacks.GradientNormMonitorCallback"
    norm_type: float = 2.0
    log_every_n_steps: int = 10


class GradientNormMonitorCallback:
    """Compute and log the gradient norm before each optimizer step.

    The hook fires **after** gradient clipping (if ``gradient_clip_val`` is set
    in the trainer config) and **before** ``optimizer.step()``, so the logged
    norm reflects the *post-clip* gradient norm when clipping is active.

    The norm is logged under the key ``"grad_norm"`` via
    ``trainer.fabric.log_dict()``.

    Args:
        norm_type: The type of norm to compute (e.g. ``2.0`` for the L2 norm,
            ``float("inf")`` for the max-absolute-value norm).
        log_every_n_steps: Log every this many optimizer steps.  Logging every
            step can be expensive for large models; the default of 10 reduces
            overhead while still giving a useful signal.

    Note:
        When ``gradient_clip_val`` is set in :class:`TrainerConfig`, the norm
        is measured *after* clipping and will be at most ``gradient_clip_val``.
        To observe the raw (pre-clip) norm, disable gradient clipping in the
        config and apply clipping manually in a callback instead.

    Example::

        from deepfont.callbacks import GradientNormMonitorCallback

        cb = GradientNormMonitorCallback(norm_type=2.0, log_every_n_steps=50)
    """

    def __init__(
        self,
        norm_type: float = 2.0,
        log_every_n_steps: int = 10,
    ) -> None:
        self.norm_type = norm_type
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(self, trainer, optimizer) -> None:
        """Compute the gradient norm and log it."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        if trainer.model is None:
            return

        total_norm = sum(
            p.grad.data.norm(self.norm_type).item() ** self.norm_type
            for p in trainer.model.parameters()
            if p.grad is not None
        ) ** (1.0 / self.norm_type)

        trainer.fabric.log_dict({"grad_norm": total_norm}, step=trainer.global_step)
