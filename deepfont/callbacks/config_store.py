"""Hydra ConfigStore registration for all callback structured configs."""

from __future__ import annotations


def register_callback_configs() -> None:
    """Register all callback config dataclasses with Hydra's ConfigStore.

    Call this function once at the start of your training entry-point script,
    before @hydra.main processes any config.  This makes each callback
    available as a Hydra config group under the "callbacks" group so that
    configs can be selected via defaults lists:

        # conf/train/pretrain.yaml
        defaults:
          - callbacks/early_stopping
          - callbacks/lr_monitor
          - _self_

    You can then instantiate the callbacks in your script with:

        from hydra.utils import instantiate
        callbacks = [instantiate(cb) for cb in cfg.callbacks.values()]

    Example entry-point usage:

        from deepfont.callbacks.config_store import register_callback_configs
        register_callback_configs()

        import hydra
        from omegaconf import DictConfig

        @hydra.main(config_path="../../conf", config_name="train/pretrain")
        def main(cfg: DictConfig) -> None:
            ...
    """
    from hydra.core.config_store import ConfigStore

    from .lr_monitor import LearningRateMonitorCallbackConfig
    from .gradient_norm import GradientNormMonitorCallbackConfig
    from .early_stopping import EarlyStoppingCallbackConfig
    from .model_checkpoint import ModelCheckpointCallbackConfig
    from .reconstruction_visualizer import ReconstructionVisualizerCallbackConfig

    cs = ConfigStore.instance()
    cs.store(group="callbacks", name="early_stopping", node=EarlyStoppingCallbackConfig)
    cs.store(group="callbacks", name="model_checkpoint", node=ModelCheckpointCallbackConfig)
    cs.store(group="callbacks", name="lr_monitor", node=LearningRateMonitorCallbackConfig)
    cs.store(group="callbacks", name="gradient_norm", node=GradientNormMonitorCallbackConfig)
    cs.store(
        group="callbacks",
        name="reconstruction_visualizer",
        node=ReconstructionVisualizerCallbackConfig,
    )
