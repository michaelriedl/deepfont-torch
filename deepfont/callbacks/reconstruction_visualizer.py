"""Reconstruction visualizer callback for the pretrain stage."""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class ReconstructionVisualizerCallbackConfig:
    """Hydra structured config for ReconstructionVisualizerCallback.

    Register with Hydra's ConfigStore via
    register_callback_configs() to use as a config group default.
    """

    _target_: str = "deepfont.callbacks.ReconstructionVisualizerCallback"
    save_every_n_epochs: int = 5
    num_samples: int = 8
    output_dir: str = "reconstructions"
    value_range: tuple | None = None


class ReconstructionVisualizerCallback:
    """Save input / reconstruction image grids to disk during pretraining.

    At the end of qualifying validation epochs (every save_every_n_epochs),
    this callback runs the stored sample inputs through the model and saves a
    side-by-side grid of originals and reconstructions as a PNG file.

    When batches carry an is_real flag (as emitted by PretrainTrainer), the
    callback collects num_samples // 2 real and num_samples // 2 synthetic
    images by scanning across validation batches.  This guarantees both
    image types appear in the grid even when the first batch is homogeneous.
    If one type is absent from the entire validation set, the grid is filled
    with whatever is available.

    Args:
        save_every_n_epochs: Save a grid every this many epochs.  Epoch 0 is
            always included.
        num_samples: Total number of images in the grid.  When is_real flags
            are present, half are real and half are synthetic.
        output_dir: Directory where PNG files are written.  Created
            automatically if it does not exist.
        value_range: Passed directly to torchvision.utils.make_grid as
            the value_range argument (a (min, max) tuple).  Set to
            (-1, 1) when using image_normalization="-1to1" in the
            trainer config.  None triggers auto-normalization (min/max
            over the displayed batch), which is safe for both normalizations.

    Requires:
        torchvision must be installed.  It is an indirect dependency of
        this project; add it as an explicit dependency if needed.

    Example:
        >>> from deepfont.callbacks import ReconstructionVisualizerCallback
        >>> cb = ReconstructionVisualizerCallback(
        ...     save_every_n_epochs=10,
        ...     num_samples=16,
        ...     output_dir="runs/pretrain/reconstructions",
        ...     value_range=(0.0, 1.0),
        ... )
    """

    def __init__(
        self,
        save_every_n_epochs: int = 5,
        num_samples: int = 8,
        output_dir: str = "reconstructions",
        value_range: tuple | None = None,
    ) -> None:
        self.save_every_n_epochs = save_every_n_epochs
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.value_range = value_range

        self._fixed_samples: torch.Tensor | None = None
        self._pending_real: list[torch.Tensor] = []
        self._pending_syn: list[torch.Tensor] = []

    def _is_save_epoch(self, trainer) -> bool:
        return trainer.current_epoch % self.save_every_n_epochs == 0

    def _pending_count(self, pending: list[torch.Tensor]) -> int:
        return sum(t.shape[0] for t in pending)

    def on_validation_batch_start(self, batch, batch_idx, trainer) -> None:
        """Collect real and synthetic samples across batches until quotas are met.

        The samples are frozen on the first qualifying epoch so that the
        same inputs are visualized across epochs, making it easy to see
        how reconstruction quality evolves over time.

        When the batch carries an is_real flag, num_samples // 2 images of
        each type are collected by scanning as many batches as needed.
        """
        if not self._is_save_epoch(trainer):
            return
        if self._fixed_samples is not None:
            return  # already locked in for this run

        # Reset pending buffers at the start of each save-epoch's val loop.
        if batch_idx == 0:
            self._pending_real = []
            self._pending_syn = []

        if not isinstance(batch, (tuple, list)):
            # Plain tensor batch (no type info): capture first batch and lock in.
            if batch_idx == 0:
                self._fixed_samples = batch[: self.num_samples].detach().cpu()
            return

        images = batch[0].detach().cpu()
        is_real = batch[1].bool().cpu()

        n_real_target = self.num_samples // 2
        n_syn_target = self.num_samples - n_real_target

        need_real = n_real_target - self._pending_count(self._pending_real)
        need_syn = n_syn_target - self._pending_count(self._pending_syn)

        if need_real > 0:
            real_imgs = images[is_real][:need_real]
            if real_imgs.shape[0] > 0:
                self._pending_real.append(real_imgs)

        if need_syn > 0:
            syn_imgs = images[~is_real][:need_syn]
            if syn_imgs.shape[0] > 0:
                self._pending_syn.append(syn_imgs)

        # Lock in as soon as both quotas are satisfied.
        if (
            self._pending_count(self._pending_real) >= n_real_target
            and self._pending_count(self._pending_syn) >= n_syn_target
        ):
            self._fixed_samples = torch.cat(
                [torch.cat(self._pending_real), torch.cat(self._pending_syn)]
            )

    def on_validation_epoch_end(self, trainer, val_metrics) -> None:
        """Finalize pending samples (if any) then run the model and save the grid."""
        # If the val loop ended before both quotas were filled (e.g. one image
        # type is absent from the val set), lock in whatever was collected.
        if self._fixed_samples is None and self._is_save_epoch(trainer):
            parts = [
                torch.cat(self._pending_real) if self._pending_real else None,
                torch.cat(self._pending_syn) if self._pending_syn else None,
            ]
            parts = [p for p in parts if p is not None]
            if parts:
                self._fixed_samples = torch.cat(parts)

        if self._fixed_samples is None or not self._is_save_epoch(trainer):
            return

        # Only write files on rank 0.
        if not trainer.fabric.is_global_zero:
            return

        try:
            from torchvision.utils import make_grid, save_image
        except ImportError as exc:
            raise ImportError(
                "ReconstructionVisualizerCallback requires torchvision. "
                "Install it with: pip install torchvision"
            ) from exc

        inputs = self._fixed_samples  # (N, 1, H, W) on CPU

        device = trainer.fabric.device
        with torch.no_grad():
            reconstructed = trainer.model(inputs.to(device)).detach().cpu()

        # Clamp to avoid artifacts from out-of-range activations.
        if self.value_range is not None:
            lo, hi = self.value_range
            inputs = inputs.clamp(lo, hi)
            reconstructed = reconstructed.clamp(lo, hi)

        # Interleave: row 0 = inputs, row 1 = reconstructions (column-wise).
        n = inputs.shape[0]
        # Stack so that pairs appear next to each other in the grid.
        interleaved = torch.stack([inputs, reconstructed], dim=1).view(2 * n, *inputs.shape[1:])

        grid = make_grid(
            interleaved,
            nrow=n,
            normalize=True,
            value_range=self.value_range,
        )

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, f"epoch-{trainer.current_epoch:04d}.png")
        save_image(grid, out_path)
        logger.info("[ReconstructionVisualizer] Saved grid → %s", out_path)
