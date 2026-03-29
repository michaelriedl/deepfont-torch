"""DeepFont trainer package.

Provides a Lightning Fabric-based training framework with a shared base class
and two concrete implementations:

- PretrainTrainer -- trains DeepFontAE for unsupervised reconstruction
  pretraining.
- FinetuneTrainer -- trains DeepFont for supervised font classification,
  with optional transfer from a pretrained encoder and TTA evaluation.

Typical workflow:

    from deepfont.trainer import PretrainTrainer, PretrainConfig
    from deepfont.trainer import FinetuneTrainer, FinetuneConfig
    from deepfont.data.config import PretrainDataConfig, FinetuneDataConfig, EvalDataConfig
    from deepfont.models.config import DeepFontAEConfig, DeepFontConfig

    # 1. Pretrain the autoencoder
    pre_config = PretrainConfig(learning_rate=1e-3, max_epochs=50)
    model_config = DeepFontAEConfig()
    data_config = PretrainDataConfig(synthetic_bcf_file="...", real_image_dir="...")
    trainer = PretrainTrainer(pre_config, model_config, data_config)
    trainer.fit()
    trainer.save_encoder_weights(
        ckpt_path="checkpoints/epoch-0050.ckpt",
        output_path="checkpoints/encoder_weights.pt",
    )

    # 2. Fine-tune the classifier
    ft_config = FinetuneConfig(
        encoder_weights_path="checkpoints/encoder_weights.pt",
        learning_rate=1e-4,
        max_epochs=30,
    )
    ft_model_config = DeepFontConfig(num_classes=2383)
    ft_data_config = FinetuneDataConfig(
        synthetic_bcf_file="...",
        label_file="...",
    )
    eval_data_config = EvalDataConfig(
        synthetic_bcf_file="...",
        label_file="...",
    )
    trainer = FinetuneTrainer(ft_config, ft_model_config, ft_data_config, eval_data_config)
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
