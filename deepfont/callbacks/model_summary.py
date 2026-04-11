"""Model summary callback for Lightning Fabric trainers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelSummaryCallbackConfig:
    """Hydra structured config for ModelSummaryCallback.

    Register with Hydra's ConfigStore via
    register_callback_configs() to use as a config group default.
    """

    _target_: str = "deepfont.callbacks.ModelSummaryCallback"
    max_depth: int = 1


class ModelSummaryCallback:
    """Print a model architecture summary at the start of training.

    Uses torchinfo to generate a summary of the model architecture
    including parameter counts, layer shapes, and memory estimates.
    The summary is printed via trainer.fabric.print so it respects
    rank-zero-only output in distributed training.

    Args:
        max_depth: Maximum depth of nested layers to display. Use -1
            for unlimited depth.

    Example:
        >>> from deepfont.callbacks import ModelSummaryCallback
        >>> cb = ModelSummaryCallback(max_depth=2)
    """

    def __init__(self, max_depth: int = 1) -> None:
        self.max_depth = max_depth

    def on_fit_start(self, trainer) -> None:
        """Print the model summary."""
        if trainer.model is None:
            return

        from torchinfo import summary

        model_info = summary(
            trainer.model,
            depth=self.max_depth,
            verbose=0,
        )
        if trainer.fabric.is_global_zero:
            logger.info("%s", model_info)
