"""Entry-point script for DeepFont autoencoder pretraining.

Trains the DeepFontAE autoencoder using Lightning Fabric and Hydra for
configuration management. After training, encoder weights can be extracted
from checkpoints for transfer to the DeepFont classifier.

Usage:
    python scripts/pretrain.py
    python scripts/pretrain.py experiment=pretrain_smoke_test
    python scripts/pretrain.py trainer.max_epochs=200 trainer.batch_size=128
"""

import os
import logging
from typing import Any
from pathlib import Path

import hydra
from omegaconf import OmegaConf, DictConfig

# Auto-detect project root from script location (scripts/ -> parent).
# An explicit PROJECT_ROOT env var takes precedence if set.
os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))

from deepfont.trainer.pretrain import PretrainTrainer

logger = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig | None) -> list[Any]:
    """Instantiate all callbacks from a Hydra callbacks config group.

    Args:
        callbacks_cfg: The cfg.callbacks DictConfig, where each key holds
            a sub-config with a _target_ field. None means no callbacks.

    Returns:
        A list of instantiated callback objects.
    """
    callbacks: list[Any] = []
    if callbacks_cfg is None:
        return callbacks
    for _, cb_cfg in callbacks_cfg.items():
        if isinstance(cb_cfg, DictConfig) and "_target_" in cb_cfg:
            callbacks.append(hydra.utils.instantiate(cb_cfg))
    return callbacks


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Any]:
    """Instantiate all loggers from a Hydra logger config group.

    Args:
        logger_cfg: The cfg.logger DictConfig, where each key holds a
            sub-config with a _target_ field. None means no loggers.

    Returns:
        A list of instantiated logger objects.
    """
    loggers: list[Any] = []
    if logger_cfg is None:
        return loggers
    if isinstance(logger_cfg, DictConfig):
        for _, lg_cfg in logger_cfg.items():
            if isinstance(lg_cfg, DictConfig) and "_target_" in lg_cfg:
                loggers.append(hydra.utils.instantiate(lg_cfg))
    return loggers


@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run DeepFont autoencoder pretraining."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Instantiate Pydantic configs via Hydra _target_
    data_config = hydra.utils.instantiate(cfg.data)
    model_config = hydra.utils.instantiate(cfg.model)
    trainer_config = hydra.utils.instantiate(cfg.trainer)

    # Instantiate callbacks and loggers
    callbacks = instantiate_callbacks(cfg.get("callbacks"))
    loggers = instantiate_loggers(cfg.get("logger"))

    trainer = PretrainTrainer(
        config=trainer_config,
        model_config=model_config,
        data_config=data_config,
        loggers=loggers or None,
        callbacks=callbacks or None,
    )

    if cfg.get("train", True):
        trainer.fit(ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    main()
