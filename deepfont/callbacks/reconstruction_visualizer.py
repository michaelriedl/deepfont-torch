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

    Warning:
        This callback is designed for use with PretrainTrainer only.  It
        expects batch in on_validation_batch_start to be a plain image tensor
        of shape (B, 1, H, W).  Attaching it to FinetuneTrainer (where batches
        are (images, labels) tuples) will raise a TypeError.

    Args:
        save_every_n_epochs: Save a grid every this many epochs.  Epoch 0 is
            always included.
        num_samples: Number of images to include in the grid (taken from the
            start of the first validation batch).
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

    def _is_save_epoch(self, trainer) -> bool:
        return trainer.current_epoch % self.save_every_n_epochs == 0

    def on_validation_batch_start(self, batch, batch_idx, trainer) -> None:
        """Capture the first num_samples images once and reuse them every epoch.

        The samples are frozen on the first qualifying epoch so that the
        same inputs are visualized across epochs, making it easy to see
        how reconstruction quality evolves over time.
        """
        if batch_idx != 0 or not self._is_save_epoch(trainer):
            return
        # Only capture once; reuse the same images for every subsequent epoch.
        if self._fixed_samples is None:
            # PretrainTrainer emits (images, is_real) tuples; extract just images.
            images = batch[0] if isinstance(batch, tuple) else batch
            self._fixed_samples = images[: self.num_samples].detach().cpu()

    def on_validation_epoch_end(self, trainer, val_metrics) -> None:
        """Run the fixed sample inputs through the model and save the grid."""
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
