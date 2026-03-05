"""Callback classes for the deepfont training loop.

All callbacks integrate with Lightning Fabric's ``fabric.call()`` mechanism
and are compatible with Hydra instantiation via ``hydra.utils.instantiate()``.

Callbacks:
    EarlyStoppingCallback: Stop training when a monitored validation metric plateaus.
    ModelCheckpointCallback: Keep the best-N checkpoints based on a monitored validation metric.
    LearningRateMonitorCallback: Log the current learning rate(s) to the Fabric logger each epoch.
    GradientNormMonitorCallback: Compute and log gradient norms before each optimizer step.
    ReconstructionVisualizerCallback: Save input / reconstruction image grids to disk
    (pretrain stage only).

Hydra integration:
    Each callback ships with a matching ``*Config`` dataclass that can be
    registered with Hydra's ConfigStore::

        from deepfont.callbacks.config_store import register_callback_configs
        register_callback_configs()

    See :mod:`deepfont.callbacks.config_store` for details.
"""

from .lr_monitor import LearningRateMonitorCallback, LearningRateMonitorCallbackConfig
from .gradient_norm import GradientNormMonitorCallback, GradientNormMonitorCallbackConfig
from .early_stopping import EarlyStoppingCallback, EarlyStoppingCallbackConfig
from .model_checkpoint import ModelCheckpointCallback, ModelCheckpointCallbackConfig
from .reconstruction_visualizer import (
    ReconstructionVisualizerCallback,
    ReconstructionVisualizerCallbackConfig,
)

__all__ = [
    "EarlyStoppingCallback",
    "EarlyStoppingCallbackConfig",
    "GradientNormMonitorCallback",
    "GradientNormMonitorCallbackConfig",
    "LearningRateMonitorCallback",
    "LearningRateMonitorCallbackConfig",
    "ModelCheckpointCallback",
    "ModelCheckpointCallbackConfig",
    "ReconstructionVisualizerCallback",
    "ReconstructionVisualizerCallbackConfig",
]
