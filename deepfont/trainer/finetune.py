"""Trainer for the DeepFont supervised fine-tuning stage."""

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from tqdm import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from deepfont.data.datasets import EvalData, FinetuneData
from deepfont.models.deepfont import DeepFont

from .base import BaseTrainer
from .config import FinetuneConfig


class FinetuneTrainer(BaseTrainer):
    """Trainer for the :class:`~deepfont.models.deepfont.DeepFont` classification stage.

    Trains a font classifier with cross-entropy loss and optional transfer of
    pretrained encoder weights from a :class:`PretrainTrainer` run.

    Beyond the standard :meth:`fit` loop, :meth:`evaluate` runs a separate
    test-time augmentation (TTA) pass over an :class:`~deepfont.data.datasets.EvalData`
    dataset, averaging logits across multiple crops of each image to produce a
    more robust top-1 accuracy estimate.

    Example::

        from deepfont.trainer import FinetuneTrainer, FinetuneConfig

        config = FinetuneConfig(
            bcf_store_file="data/finetune.bcf",
            label_file="data/finetune.labels",
            eval_bcf_store_file="data/test.bcf",
            eval_label_file="data/test.labels",
            encoder_weights_path="checkpoints/encoder_weights.pt",
            num_classes=2383,
            learning_rate=1e-4,
            max_epochs=30,
        )
        trainer = FinetuneTrainer(config)
        trainer.fit()
        results = trainer.evaluate(ckpt_path="checkpoints/epoch-0030.ckpt")
        print(results["accuracy"])
    """

    def __init__(
        self,
        config: FinetuneConfig,
        loggers=None,
        callbacks=None,
    ) -> None:
        super().__init__(config, loggers=loggers, callbacks=callbacks)
        self.config: FinetuneConfig = config

    # -------------------------------------------------------------------------
    # BaseTrainer abstract interface
    # -------------------------------------------------------------------------

    def create_model(self) -> nn.Module:
        """Return a :class:`~deepfont.models.deepfont.DeepFont` instance.

        If ``config.encoder_weights_path`` is set, the pretrained AE encoder
        weights are loaded and frozen before training begins.
        """
        model = DeepFont(num_out=self.config.num_classes)
        if self.config.encoder_weights_path is not None:
            model.load_encoder_weights(self.config.encoder_weights_path)
        return model

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build :class:`~deepfont.data.datasets.FinetuneData` and split it.

        The validation split uses ``aug_prob=0.0`` (no augmentation) to give a
        consistent, repeatable loss estimate across epochs.

        Returns:
            ``(train_loader, val_loader)`` ready for :meth:`fit`.
        """
        dataset = FinetuneData(
            bcf_store_file=self.config.bcf_store_file,
            label_file=self.config.label_file,
            aug_prob=self.config.aug_prob,
            image_normalization=self.config.image_normalization,
        )
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
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    def create_optimizer(
        self,
        model: nn.Module,
    ) -> tuple[Optimizer, LRScheduler | None]:
        """Return an Adam optimiser over trainable parameters only.

        Parameters with ``requires_grad=False`` (i.e. frozen encoder layers
        when ``encoder_weights_path`` is set) are excluded automatically.
        """
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(
            trainable,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
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
            model: Fabric-wrapped :class:`~deepfont.models.deepfont.DeepFont`.
            batch: ``(images, labels)`` tuple with shapes
                ``(B, 1, H, W)`` and ``(B,)``.
            batch_idx: Unused; present for interface compatibility.

        Returns:
            ``{"loss": cross_entropy, "acc": top1_accuracy}``
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
            batch: ``(images, labels)`` tuple.
            batch_idx: Unused; present for interface compatibility.

        Returns:
            ``{"loss": cross_entropy, "acc": top1_accuracy}``
        """
        images, labels = batch
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=1) == labels).float().mean()
        return {"loss": loss, "acc": acc}

    # -------------------------------------------------------------------------
    # TTA evaluation
    # -------------------------------------------------------------------------

    def evaluate(self, ckpt_path: str | None = None) -> dict[str, float]:
        """Run test-time augmentation (TTA) evaluation over the eval dataset.

        For each image, :class:`~deepfont.data.datasets.EvalData` generates
        ``num_image_crops`` augmented crops.  The model produces a logit
        vector per crop; these are averaged before taking the argmax.  This
        ensemble typically yields higher accuracy than single-crop evaluation.

        This method is self-contained: it builds a fresh model, optionally
        loads from a checkpoint, and runs evaluation without affecting any
        state set by a prior :meth:`fit` call.

        Args:
            ckpt_path: Optional path to a Fabric checkpoint to load weights
                from.  If ``None`` the model is evaluated with its initialised
                weights (only useful if :meth:`fit` was called beforehand and
                you are using this method to evaluate mid-run state).

        Returns:
            A dict with keys:

            - ``"accuracy"`` — top-1 accuracy as a float in ``[0, 1]``
            - ``"correct"`` — number of correctly classified images
            - ``"total"`` — total number of images evaluated

        Raises:
            ValueError: If ``eval_bcf_store_file`` or ``eval_label_file`` is
                not set in the config.
        """
        if not self.config.eval_bcf_store_file or not self.config.eval_label_file:
            raise ValueError(
                "Both eval_bcf_store_file and eval_label_file must be set in "
                "FinetuneConfig before calling evaluate()."
            )

        self._ensure_launched()

        with self.fabric.init_module():
            model = self.create_model()

        # setup() without an optimizer; Fabric still handles device placement
        model = self.fabric.setup(model)

        if ckpt_path is not None:
            state: dict = {"model": model}
            self.fabric.load(ckpt_path, state)

        eval_dataset = EvalData(
            bcf_store_file=self.config.eval_bcf_store_file,
            label_file=self.config.eval_label_file,
            image_normalization=self.config.image_normalization,
            num_image_crops=self.config.num_image_crops,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.config.val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
        eval_loader = self.fabric.setup_dataloaders(eval_loader)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            iterable = self._progbar(
                eval_loader,
                total=len(eval_loader),
                desc="Evaluate [TTA]",
            )
            for crops, labels in iterable:
                # crops:  (B, N, 1, H, W)  — N augmented views per image
                # labels: (B,)
                b, n, c, h, w = crops.shape
                crops_flat = crops.view(b * n, c, h, w)

                logits = model(crops_flat)  # (B*N, num_classes)

                # Average logits across crops then classify
                avg_logits = logits.view(b, n, -1).mean(dim=1)  # (B, num_classes)
                preds = avg_logits.argmax(dim=1)  # (B,)

                correct += (preds == labels).sum().item()
                total += b

                if isinstance(iterable, tqdm):
                    iterable.set_postfix({"acc": f"{correct / total:.4f}"})

        accuracy = correct / total
        self.fabric.print(f"TTA Evaluation: accuracy={accuracy:.4f} ({correct}/{total})")
        return {"accuracy": accuracy, "correct": correct, "total": total}
