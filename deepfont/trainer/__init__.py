"""DeepFont trainer package.

Provides a Lightning Fabric-based training framework with a shared base class
and two concrete implementations:

- :class:`PretrainTrainer` — trains :class:`~deepfont.models.deepfont.DeepFontAE`
  for unsupervised reconstruction pretraining.
- :class:`FinetuneTrainer` — trains :class:`~deepfont.models.deepfont.DeepFont`
  for supervised font classification, with optional transfer from a pretrained
  encoder and TTA evaluation.

Typical workflow::

    from deepfont.trainer import PretrainTrainer, PretrainConfig
    from deepfont.trainer import FinetuneTrainer, FinetuneConfig

    # 1. Pretrain the autoencoder
    pre_config = PretrainConfig(bcf_store_file="...", ...)
    PretrainTrainer(pre_config).fit()
    PretrainTrainer(pre_config).save_encoder_weights(
        ckpt_path="checkpoints/epoch-0050.ckpt",
        output_path="checkpoints/encoder_weights.pt",
    )

    # 2. Fine-tune the classifier
    ft_config = FinetuneConfig(
        bcf_store_file="...",
        label_file="...",
        encoder_weights_path="checkpoints/encoder_weights.pt",
        ...
    )
    trainer = FinetuneTrainer(ft_config)
    trainer.fit()
    trainer.evaluate(ckpt_path="checkpoints/epoch-0030.ckpt")
"""

from .base import BaseTrainer
from .config import TrainerConfig, FinetuneConfig, PretrainConfig
from .finetune import FinetuneTrainer
from .pretrain import PretrainTrainer

__all__ = [
    "BaseTrainer",
    "TrainerConfig",
    "PretrainConfig",
    "FinetuneConfig",
    "PretrainTrainer",
    "FinetuneTrainer",
]
