"""Trainer for the DeepFont supervised fine-tuning stage."""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from deepfont.data.config import EvalDataConfig, FinetuneDataConfig
from deepfont.data.datasets import EvalData, FinetuneData, bcf_worker_init_fn
from deepfont.models.config import DeepFontConfig
from deepfont.models.deepfont import DeepFont

from .base import BaseTrainer
from .config import FinetuneConfig

logger = logging.getLogger(__name__)


class FinetuneTrainer(BaseTrainer):
    """Trainer for the DeepFont classification stage.

    Trains a font classifier with cross-entropy loss and optional transfer of
    pretrained encoder weights from a PretrainTrainer run.

    Beyond the standard fit() loop, evaluate() runs a separate
    test-time augmentation (TTA) pass over an EvalData
    dataset, averaging logits across multiple crops of each image to produce a
    more robust top-1 accuracy estimate.

    Example:
        >>> from deepfont.trainer import FinetuneTrainer, FinetuneConfig
        >>> from deepfont.data.config import FinetuneDataConfig, EvalDataConfig
        >>> from deepfont.models.config import DeepFontConfig
        >>> config = FinetuneConfig(
        ...     encoder_weights_path="checkpoints/encoder_weights.pt",
        ...     learning_rate=1e-4,
        ...     max_epochs=30,
        ... )
        >>> model_config = DeepFontConfig(num_classes=2383)
        >>> data_config = FinetuneDataConfig(
        ...     synthetic_bcf_file="data/finetune.bcf",
        ...     label_file="data/finetune.labels",
        ... )
        >>> eval_data_config = EvalDataConfig(
        ...     synthetic_bcf_file="data/test.bcf",
        ...     label_file="data/test.labels",
        ... )
        >>> trainer = FinetuneTrainer(config, model_config, data_config, eval_data_config)
        >>> trainer.fit()
        >>> results = trainer.evaluate(ckpt_path="checkpoints/epoch-0030.ckpt")
        >>> print(results["accuracy"])
    """

    def __init__(
        self,
        config: FinetuneConfig,
        model_config: DeepFontConfig,
        data_config: FinetuneDataConfig | None = None,
        eval_data_config: EvalDataConfig | None = None,
        loggers=None,
        callbacks=None,
    ) -> None:
        super().__init__(config, loggers=loggers, callbacks=callbacks)
        self.config: FinetuneConfig = config
        self.model_config = model_config
        self.data_config = data_config
        self.eval_data_config = (
            eval_data_config if eval_data_config is not None else EvalDataConfig()
        )

    # BaseTrainer abstract interface

    def create_model(self) -> nn.Module:
        """Return a DeepFont instance.

        If config.encoder_weights_path is set, the pretrained AE encoder
        weights are loaded and frozen before training begins.
        """
        model = DeepFont(self.model_config)
        if self.config.encoder_weights_path is not None:
            model.load_encoder_weights(self.config.encoder_weights_path)
        return model

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build FinetuneData and split it.

        The validation split uses aug_prob=0.0 (no augmentation) to give a
        consistent, repeatable loss estimate across epochs.

        Returns:
            (train_loader, val_loader) ready for fit().

        Raises:
            ValueError: If data_config was not provided (eval-only usage).
        """
        if self.data_config is None:
            raise ValueError(
                "data_config must be provided to call fit(). "
                "Pass a FinetuneDataConfig or use evaluate() for inference only."
            )
        dataset = FinetuneData(self.data_config)
        train_set, val_set = dataset.split_data_random(train_ratio=self.config.train_ratio)

        # Disable augmentation for the validation split
        val_set.aug_prob = 0.0

        if self.config.num_images_to_cache > 0:
            train_set.cache_images(self.config.num_images_to_cache)
            val_set.cache_images(self.config.num_images_to_cache)

        train_loader = DataLoader(
            train_set,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=bcf_worker_init_fn,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            worker_init_fn=bcf_worker_init_fn,
        )
        return train_loader, val_loader

    def create_optimizer(
        self,
        model: nn.Module,
    ) -> tuple[Optimizer, LRScheduler | None]:
        """Return the configured optimizer over trainable parameters only.

        Parameters with requires_grad=False (i.e. frozen encoder layers
        when encoder_weights_path is set) are excluded automatically.
        """
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = self._build_optimizer(
            trainable,
            optimizer_type=self.config.optimizer_type,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            optimizer_kwargs=self.config.optimizer_kwargs,
        )
        scheduler = self._build_scheduler(
            optimizer,
            self.config.scheduler_type,
            self.config.scheduler_kwargs,
        )
        return optimizer, scheduler

    def training_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Classification forward pass for one training batch.

        Args:
            model: Fabric-wrapped DeepFont.
            batch: (images, labels) tuple with shapes
                (B, 1, H, W) and (B,).
            batch_idx: Unused; present for interface compatibility.

        Returns:
            {"loss": cross_entropy, "acc": top1_accuracy}
        """
        images, labels = batch
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return {"loss": loss, "acc": acc}

    def validation_step(
        self,
        model: nn.Module,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Classification forward pass for one validation batch.

        Args:
            model: Fabric-wrapped model in eval mode.
            batch: (images, labels) tuple.
            batch_idx: Unused; present for interface compatibility.

        Returns:
            {"loss": cross_entropy, "acc": top1_accuracy}
        """
        images, labels = batch
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return {"loss": loss, "acc": acc}

    # TTA evaluation

    def evaluate(self, ckpt_path: str | None = None) -> dict[str, float]:
        """Run test-time augmentation (TTA) evaluation over the eval dataset.

        For each image, EvalData generates num_image_crops augmented crops.
        The model produces a logit vector per crop; these are averaged before
        taking the argmax.  This ensemble typically yields higher accuracy than
        single-crop evaluation.

        This method is self-contained: it builds a fresh model, optionally
        loads from a checkpoint, and runs evaluation without affecting any
        state set by a prior fit() call.

        Args:
            ckpt_path: Optional path to a Fabric checkpoint to load weights
                from.  If None the model is evaluated with its initialized
                weights (only useful if fit() was called beforehand and
                you are using this method to evaluate mid-run state).

        Returns:
            A dict with keys:

            - "accuracy" -- top-1 accuracy as a float in [0, 1]
            - "correct" -- number of correctly classified images
            - "total" -- total number of images evaluated

        Raises:
            ValueError: If eval_data_config.synthetic_bcf_file or
                eval_data_config.label_file is not set.
        """
        if not self.eval_data_config.synthetic_bcf_file or not self.eval_data_config.label_file:
            raise ValueError(
                "Both synthetic_bcf_file and label_file must be set in "
                "EvalDataConfig before calling evaluate()."
            )

        # Resolve which checkpoint to load: explicit arg → best from training → warn and skip.
        if ckpt_path is None and self.best_checkpoint_path is not None:
            ckpt_path = self.best_checkpoint_path
            logger.info("evaluate(): auto-using best checkpoint from training: %s", ckpt_path)
        elif ckpt_path is None:
            logger.warning(
                "evaluate(): no checkpoint path provided and no best checkpoint recorded "
                "from a prior fit() call. Evaluating with randomly initialized weights "
                "(encoder weights only). Results will not reflect trained performance."
            )

        self._ensure_launched()

        with self.fabric.init_module():
            model = self.create_model()

        # setup() without an optimizer; Fabric still handles device placement
        model = self.fabric.setup(model)

        if ckpt_path is not None:
            state: dict = {"model": model}
            self.fabric.load(ckpt_path, state)

        eval_dataset = EvalData(self.eval_data_config)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
            worker_init_fn=bcf_worker_init_fn,
        )
        eval_loader = self.fabric.setup_dataloaders(eval_loader)

        model.eval()
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        limit = self.config.limit_eval_batches
        num_batches = len(eval_loader)
        if limit is not None:
            num_batches = min(num_batches, limit)

        with torch.no_grad():
            iterable = self._progbar(
                eval_loader,
                total=num_batches,
                desc="Evaluate [TTA]",
            )
            for batch_idx, (crops, labels) in enumerate(iterable):
                if limit is not None and batch_idx >= limit:
                    break
                # crops:  (B, N, 1, H, W)  — N augmented views per image
                # labels: (B,)
                b, n, c, h, w = crops.shape
                crops_flat = crops.view(b * n, c, h, w)

                logits = model(crops_flat)  # (B*N, num_classes)

                # Average logits across crops then classify
                avg_logits = logits.view(b, n, -1).mean(dim=1)  # (B, num_classes)

                top5 = avg_logits.topk(5, dim=1).indices  # (B, 5)
                correct_top1 += (top5[:, 0] == labels).sum().item()
                correct_top5 += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += b

                if isinstance(iterable, tqdm):
                    iterable.set_postfix(
                        {
                            "top1": f"{correct_top1 / total:.4f}",
                            "top5": f"{correct_top5 / total:.4f}",
                        }
                    )

        top1 = correct_top1 / total
        top5 = correct_top5 / total
        if self.fabric.is_global_zero:
            logger.info(
                "TTA Evaluation: top1=%.4f (%d/%d)  top5=%.4f (%d/%d)",
                top1,
                correct_top1,
                total,
                top5,
                correct_top5,
                total,
            )
        return {
            "accuracy": top1,
            "top1_accuracy": top1,
            "top5_accuracy": top5,
            "correct": correct_top1,
            "top5_correct": correct_top5,
            "total": total,
        }
