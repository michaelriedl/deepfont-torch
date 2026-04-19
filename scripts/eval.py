"""Entry-point script for DeepFont TTA evaluation on the real test set.

Loads a finetuned DeepFont checkpoint and runs test-time augmentation (TTA)
evaluation over the real held-out test set (VFR_real_test).

Usage:
    python scripts/eval.py
    python scripts/eval.py ckpt_path=/path/to/checkpoint.ckpt
    python scripts/eval.py eval_data.num_image_crops=30
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

from deepfont.trainer.finetune import FinetuneTrainer

logger = logging.getLogger(__name__)


def instantiate_loggers(logger_cfg: DictConfig | None) -> list[Any]:
    """Instantiate all loggers from a Hydra logger config group."""
    loggers: list[Any] = []
    if logger_cfg is None:
        return loggers
    if isinstance(logger_cfg, DictConfig):
        for _, lg_cfg in logger_cfg.items():
            if isinstance(lg_cfg, DictConfig) and "_target_" in lg_cfg:
                loggers.append(hydra.utils.instantiate(lg_cfg))
    return loggers


@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run TTA evaluation on the real held-out test set."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    eval_data_config = hydra.utils.instantiate(cfg.eval_data)
    model_config = hydra.utils.instantiate(cfg.model)
    trainer_config = hydra.utils.instantiate(cfg.trainer)

    loggers = instantiate_loggers(cfg.get("logger"))

    trainer = FinetuneTrainer(
        config=trainer_config,
        model_config=model_config,
        eval_data_config=eval_data_config,
        loggers=loggers or None,
    )

    results = trainer.evaluate(ckpt_path=cfg.ckpt_path)
    logger.info(
        "Top-1: %.4f (%d / %d)  |  Top-5: %.4f (%d / %d)",
        results["top1_accuracy"], results["correct"], results["total"],
        results["top5_accuracy"], results["top5_correct"], results["total"],
    )


if __name__ == "__main__":
    main()
