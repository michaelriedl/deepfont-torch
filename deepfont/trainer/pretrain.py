"""Trainer for the DeepFont autoencoder pretraining stage."""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from deepfont.data.datasets import PretrainData
from deepfont.models.deepfont import DeepFontAE

from .base import BaseTrainer
from .config import PretrainConfig


class PretrainTrainer(BaseTrainer):
    """Trainer for the :class:`~deepfont.models.deepfont.DeepFontAE` pretraining stage.

    Trains the autoencoder to reconstruct font images using either MSE or L1
    reconstruction loss.  After training, call :meth:`save_encoder_weights`
    to export the encoder in a format compatible with
    :meth:`~deepfont.models.deepfont.DeepFont.load_encoder_weights`.

    Example::

        from deepfont.trainer import PretrainTrainer, PretrainConfig

        config = PretrainConfig(
            bcf_store_file="data/train.bcf",
            data_folder_name="data/real_images",
            learning_rate=1e-3,
            max_epochs=50,
            batch_size=64,
        )
        trainer = PretrainTrainer(config)
        trainer.fit()
        trainer.save_encoder_weights(
            ckpt_path="checkpoints/epoch-0050.ckpt",
            output_path="checkpoints/encoder_weights.pt",
        )
    """

    def __init__(
        self,
        config: PretrainConfig,
        loggers=None,
        callbacks=None,
    ) -> None:
        super().__init__(config, loggers=loggers, callbacks=callbacks)
        # Narrow the type so subclass code has access to PretrainConfig fields
        self.config: PretrainConfig = config

    # -------------------------------------------------------------------------
    # BaseTrainer abstract interface
    # -------------------------------------------------------------------------

    def create_model(self) -> nn.Module:
        """Return a :class:`~deepfont.models.deepfont.DeepFontAE` instance."""
        return DeepFontAE(output_activation=self.config.output_activation)

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        """Build :class:`~deepfont.data.datasets.PretrainData` and split it.

        Returns:
            ``(train_loader, val_loader)`` ready for :meth:`fit`.
        """
        dataset = PretrainData(
            bcf_store_file=self.config.bcf_store_file,
            data_folder_name=self.config.data_folder_name,
            aug_prob=self.config.aug_prob,
            image_normalization=self.config.image_normalization,
        )
        train_set, val_set = dataset.split_data_random(train_ratio=self.config.train_ratio)

        # Balance real ↔ synthetic in the training split
        if self.config.upsample_real_images:
            train_set.upsample_real_images()

        # Pre-load a portion of the dataset into RAM for faster iteration
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
        """Return an Adam optimiser and an optional LR scheduler."""
        optimizer = torch.optim.Adam(
            model.parameters(),
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
        batch: torch.Tensor,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Forward + reconstruction loss for a batch of images.

        Args:
            model: Fabric-wrapped :class:`~deepfont.models.deepfont.DeepFontAE`.
            batch: Image tensor of shape ``(B, 1, H, W)``.
            batch_idx: Unused; present for interface compatibility.

        Returns:
            ``{"loss": scalar_tensor}``
        """
        reconstructed = model(batch)
        return {"loss": self._reconstruction_loss(reconstructed, batch)}

    def validation_step(
        self,
        model: nn.Module,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Reconstruction loss on a validation batch.

        Args:
            model: Fabric-wrapped model in eval mode.
            batch: Image tensor of shape ``(B, 1, H, W)``.
            batch_idx: Unused; present for interface compatibility.

        Returns:
            ``{"loss": scalar_tensor}``
        """
        reconstructed = model(batch)
        return {"loss": self._reconstruction_loss(reconstructed, batch)}

    # -------------------------------------------------------------------------
    # Extra utility
    # -------------------------------------------------------------------------

    def save_encoder_weights(self, ckpt_path: str, output_path: str) -> None:
        """Extract encoder weights from a checkpoint and save them in raw format.

        The output file contains the full ``DeepFontAE`` state dict (not the
        Fabric checkpoint envelope), which is the format expected by
        :meth:`~deepfont.models.deepfont.DeepFont.load_encoder_weights`.

        Args:
            ckpt_path: Path to a Fabric checkpoint saved by :meth:`fit`.
            output_path: Destination path for the weights file (e.g.
                ``"checkpoints/encoder_weights.pt"``).
        """
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_state = checkpoint["model"]  # plain state dict saved by fabric.save

        if self.fabric.is_global_zero:
            out_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model_state, output_path)

        self.fabric.print(f"Saved encoder weights → {output_path}")

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.reconstruction_loss == "mse":
            return F.mse_loss(pred, target)
        if self.config.reconstruction_loss == "l1":
            return F.l1_loss(pred, target)
        raise ValueError(
            f"Unknown reconstruction_loss '{self.config.reconstruction_loss}'. Use 'mse' or 'l1'."
        )
