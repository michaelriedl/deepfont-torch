"""Tune the synthetic-image augmentation pipeline against a pretrained feature space.

Runs an Optuna study (TPE sampler) over the hyperparameters of
SyntheticAugmentationConfig, minimizing the Maximum Mean Discrepancy (MMD) distance between real and
augmented-synthetic image features extracted from a frozen torchvision
backbone (e.g. AlexNet).

Usage:
    python scripts/tune_augmentations.py
    python scripts/tune_augmentations.py n_trials=20 feature_extractor.model_name=resnet50
"""

import io
import os
import json
import logging
from typing import ClassVar
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
import optuna
import pandas as pd
import torch.nn.functional as functional
from PIL import Image, ImageFile, PngImagePlugin
from omegaconf import OmegaConf, DictConfig
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

# Match the relaxations applied in deepfont.data.datasets so real-image PNGs
# with oversized iCCP chunks or truncated streams still decode.
PngImagePlugin.MAX_TEXT_CHUNK = 1048576 * 10  # ty: ignore[invalid-assignment]
ImageFile.LOAD_TRUNCATED_IMAGES = True  # ty: ignore[invalid-assignment]

# Auto-detect project root from script location (scripts/ -> parent).
os.environ.setdefault("PROJECT_ROOT", str(Path(__file__).resolve().parent.parent))

from deepfont.data.bcf import BCFStoreFile
from deepfont.data.augmentations import (
    EvalAugmentationConfig,
    EvalAugmentationPipeline,
    SyntheticAugmentationConfig,
    SyntheticAugmentationPipeline,
)

logger = logging.getLogger(__name__)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# Image loading


def load_raw_images(
    manifest_file: str,
    n_real: int,
    n_synthetic: int,
    seed: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Sample raw uint8 grayscale arrays for real and synthetic images.

    Args:
        manifest_file: Path to the parquet manifest.
        n_real: Number of real images to draw.
        n_synthetic: Number of synthetic images to draw.
        seed: Seed for the RNG used to pick the sample.

    Returns:
        A tuple (real_images, synthetic_images), each a list of 2D uint8 arrays.
    """
    rng = np.random.default_rng(seed)
    manifest_dir = Path(manifest_file).resolve().parent
    df = pd.read_parquet(manifest_file)

    syn_df = df[df["image_type"] == "synthetic"]
    real_df = df[df["image_type"] == "real"]

    if len(syn_df) == 0 or len(real_df) == 0:
        raise ValueError(
            f"Manifest must contain both synthetic and real entries; got "
            f"{len(syn_df)} synthetic and {len(real_df)} real."
        )

    # Synthetic from BCF store. Index by the original bcf_index, not by row
    # order, since we are sampling a non-contiguous subset.
    syn_sample = syn_df.sample(n=min(n_synthetic, len(syn_df)), random_state=seed)
    bcf_path = str(manifest_dir / syn_sample["bcf_file"].iloc[0])
    bcf_store = BCFStoreFile(bcf_path)
    syn_indices = syn_sample["bcf_index"].to_numpy(np.int64)

    synthetic_images: list[np.ndarray] = []
    for idx in syn_indices:
        img = Image.open(io.BytesIO(bcf_store.get(int(idx)))).convert("L")
        synthetic_images.append(np.array(img, dtype=np.uint8))

    # Real from filesystem
    real_paths = [str(manifest_dir / p) for p in real_df["filepath"].dropna().tolist()]
    real_idx = rng.choice(len(real_paths), size=min(n_real, len(real_paths)), replace=False)
    real_images: list[np.ndarray] = []
    for i in real_idx:
        img = Image.open(real_paths[int(i)]).convert("L")
        if 0 in img.size or 1 in img.size:
            continue
        real_images.append(np.array(img, dtype=np.uint8))

    logger.info(
        "Loaded %d real and %d synthetic raw images.",
        len(real_images),
        len(synthetic_images),
    )
    return real_images, synthetic_images


# Feature extractor


class FrozenFeatureExtractor:
    """Frozen pretrained backbone that maps grayscale crops to a feature vector.

    Replicates the input gray channel three times, resizes to ``input_size``,
    applies ImageNet normalization, and taps the requested layer. 4D feature
    maps are global-average-pooled to a 2D (N, D) tensor.

    Args:
        model_name: One of "alexnet", "vgg16", "resnet50".
        layer: Named module to extract from (forwarded to
            torchvision's create_feature_extractor).
        input_size: Spatial resolution to resize inputs to.
        device: Torch device for the model and features.
    """

    _BUILDERS: ClassVar[dict] = {
        "alexnet": (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1),
        "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
        "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    }

    def __init__(self, model_name: str, layer: str, input_size: int, device: torch.device):
        if model_name not in self._BUILDERS:
            raise ValueError(
                f"Unsupported model '{model_name}'. Choose from {list(self._BUILDERS)}."
            )
        builder, weights = self._BUILDERS[model_name]
        backbone = builder(weights=weights)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        self.model = create_feature_extractor(backbone, return_nodes={layer: "feat"}).to(device)
        self.layer = layer
        self.input_size = input_size
        self.device = device
        self._mean = IMAGENET_MEAN.to(device)
        self._std = IMAGENET_STD.to(device)

    @torch.no_grad()
    def __call__(self, images: np.ndarray, batch_size: int) -> torch.Tensor:
        """Extract features for a stack of grayscale uint8 images.

        Args:
            images: (N, H, W) uint8 array. All images must share H, W.
            batch_size: Mini-batch size for the forward pass.

        Returns:
            A (N, D) float32 tensor of features on the configured device.
        """
        feats: list[torch.Tensor] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            x = torch.from_numpy(chunk).float().unsqueeze(1).to(self.device) / 255.0
            x = x.repeat(1, 3, 1, 1)
            x = functional.interpolate(
                x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False
            )
            x = (x - self._mean) / self._std
            out = self.model(x)["feat"]
            if out.ndim == 4:
                out = out.mean(dim=(2, 3))
            elif out.ndim > 2:
                out = out.flatten(1)
            feats.append(out)
        return torch.cat(feats, dim=0)


# Maximum Mean Discrepancy (MMD)


def median_heuristic_sigma(features: torch.Tensor, max_pairs: int = 5000) -> float:
    """Median pairwise distance among ``features`` rows, used as RBF bandwidth."""
    n = features.shape[0]
    if n > max_pairs:
        idx = torch.randperm(n, device=features.device)[:max_pairs]
        features = features[idx]
    with torch.no_grad():
        d = torch.pdist(features)
        return float(d.median().item()) or 1.0


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, sigma: float) -> float:
    """Biased MMD^2 with an RBF kernel between two feature sets."""
    with torch.no_grad():
        gamma = 1.0 / (2.0 * sigma * sigma)
        xx = torch.cdist(x, x).pow_(2).mul_(-gamma).exp_().mean()
        yy = torch.cdist(y, y).pow_(2).mul_(-gamma).exp_().mean()
        xy = torch.cdist(x, y).pow_(2).mul_(-gamma).exp_().mean()
        return float((xx + yy - 2.0 * xy).item())


# Search space


def _ordered_pair(low: float, high: float) -> tuple[float, float]:
    """Ensure (low, high) is ordered without ever inverting the range."""
    return (min(low, high), max(low, high))


SEARCH_SPACE: dict[str, tuple[float, float]] = {
    # Per-step probabilities. RandomRotate90/HorizontalFlip/VerticalFlip and
    # InvertImg are destructive for text; pin them at 0 via fixed_params unless
    # you have a text-aware feature space judging the search.
    "rot_flip_prob": (0.0, 1.0),
    "invert_prob": (0.0, 1.0),
    "affine_prob": (0.0, 1.0),
    "blur_prob": (0.0, 1.0),
    "brightness_contrast_prob": (0.0, 1.0),
    "noise_prob": (0.0, 1.0),
    "gradient_prob": (0.0, 1.0),
    # Geometric
    "aspect_ratio_low": (1.0, 3.5),
    "aspect_ratio_width": (0.0, 2.0),
    "rotate_abs": (0.0, 5.0),
    "shear_abs": (0.0, 10.0),
    # Blur
    "blur_low": (0.1, 2.0),
    "blur_width": (0.0, 3.0),
    # Noise
    "noise_std_low": (0.0, 0.05),
    "noise_std_width": (0.0, 0.05),
    # Gradient overlay. The foreground low end is parameterized as an offset
    # above the background high end so that fg_low > bg_high is structurally
    # guaranteed; otherwise the optimizer can invert the fg/bg mapping (which
    # the add_grayscale_gradient formula treats as image inversion) and so
    # work around a pinned invert_prob=0.
    "grad_fg_above_bg": (0.0, 100.0),
    "grad_fg_width": (0.0, 80.0),
    "grad_bg_low": (0.0, 130.0),
    "grad_bg_width": (0.0, 100.0),
    "grad_a_low": (0.0, 1.0),
    "grad_a_width": (0.0, 0.6),
}


def _suggest(
    trial: optuna.Trial,
    name: str,
    fixed: dict[str, float] | None,
) -> float:
    """Suggest a float for ``name``, honoring any pin in ``fixed``.

    Pinned values are validated against the search-space bounds so config
    typos do not silently produce out-of-range parameters.
    """
    low, high = SEARCH_SPACE[name]
    if fixed is not None and name in fixed:
        value = float(fixed[name])
        if not (low <= value <= high):
            raise ValueError(
                f"fixed_params[{name}]={value} is outside the search range [{low}, {high}]."
            )
        return value
    return trial.suggest_float(name, low, high)


def sample_synthetic_config(
    trial: optuna.Trial,
    fixed_params: dict[str, float] | None = None,
) -> SyntheticAugmentationConfig:
    """Draw a SyntheticAugmentationConfig from an Optuna trial's search space.

    Args:
        trial: Optuna trial used to draw values for non-pinned parameters.
        fixed_params: Optional mapping from search-space parameter name to a
            constant value. Pinned parameters are not added to the trial so
            they do not pollute the study's parameter space.

    Returns:
        A SyntheticAugmentationConfig assembled from the drawn or pinned values.
    """
    drawn = {name: _suggest(trial, name, fixed_params) for name in SEARCH_SPACE}

    ar_low = drawn["aspect_ratio_low"]
    ar_high = ar_low + drawn["aspect_ratio_width"]
    rotate_abs = drawn["rotate_abs"]
    shear_abs = drawn["shear_abs"]
    blur_low, blur_width = drawn["blur_low"], drawn["blur_width"]
    noise_low, noise_width = drawn["noise_std_low"], drawn["noise_std_width"]
    bg_low, bg_width = drawn["grad_bg_low"], drawn["grad_bg_width"]
    fg_low = bg_low + bg_width + drawn["grad_fg_above_bg"]
    fg_width = drawn["grad_fg_width"]
    a_low, a_width = drawn["grad_a_low"], drawn["grad_a_width"]

    return SyntheticAugmentationConfig(
        aspect_ratio_low=ar_low,
        aspect_ratio_high=ar_high,
        rotate_bounds=_ordered_pair(-rotate_abs, rotate_abs),
        shear_bounds=_ordered_pair(-shear_abs, shear_abs),
        blur_limit=_ordered_pair(blur_low, blur_low + blur_width),
        noise_std_range=_ordered_pair(noise_low, noise_low + noise_width),
        rot_flip_prob=drawn["rot_flip_prob"],
        invert_prob=drawn["invert_prob"],
        affine_prob=drawn["affine_prob"],
        blur_prob=drawn["blur_prob"],
        brightness_contrast_prob=drawn["brightness_contrast_prob"],
        noise_prob=drawn["noise_prob"],
        gradient_prob=drawn["gradient_prob"],
        gradient_fg_range=_ordered_pair(fg_low, fg_low + fg_width),
        gradient_bg_range=_ordered_pair(bg_low, bg_low + bg_width),
        gradient_a_range=_ordered_pair(a_low, a_low + a_width),
    )


# Augmentation rendering


def render_augmented_synthetic(
    raw_images: list[np.ndarray],
    config: SyntheticAugmentationConfig,
    augmentations_per_image: int,
) -> np.ndarray:
    """Apply the synthetic augmentation pipeline to every raw image.

    Args:
        raw_images: List of 2D uint8 grayscale arrays (variable size).
        config: Synthetic augmentation config to instantiate the pipeline.
        augmentations_per_image: How many independent augmented copies to draw
            per source image.

    Returns:
        A (N, H, W) uint8 array of augmented images, all at the configured
        crop size.
    """
    pipeline = SyntheticAugmentationPipeline(config)
    out: list[np.ndarray] = []
    for img in raw_images:
        for _ in range(augmentations_per_image):
            out.append(pipeline(img))
    return np.stack(out, axis=0)


def render_real(raw_images: list[np.ndarray], image_size: int) -> np.ndarray:
    """Frame real images with the eval pipeline (one geometric crop per image)."""
    pipeline = EvalAugmentationPipeline(EvalAugmentationConfig(image_size=image_size))
    return np.stack([pipeline(img, num_image_crops=1)[0] for img in raw_images], axis=0)


# Optuna study


def resolve_device(spec: str) -> torch.device:
    """Map a config string ("auto", "cuda", "cpu", ...) to a torch.device."""
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def write_best_yaml(path: str, config: SyntheticAugmentationConfig) -> None:
    """Persist a SyntheticAugmentationConfig as a Hydra-loadable YAML."""
    payload = {
        "_target_": "deepfont.data.config.SyntheticAugmentationConfig",
        **{k: list(v) if isinstance(v, tuple) else v for k, v in config.model_dump().items()},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(payload), path)


@hydra.main(config_path="../configs", config_name="tune_augmentations", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Run a TPE search over augmentation hyperparameters."""
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    cv2.setNumThreads(1)
    torch.set_num_threads(max(1, os.cpu_count() // 2))

    device = resolve_device(cfg.device)
    logger.info("Using device: %s", device)

    real_raw, syn_raw = load_raw_images(
        cfg.manifest_file,
        cfg.n_real_samples,
        cfg.n_synthetic_samples,
        cfg.sample_seed,
    )

    # Frame real images once; their features are independent of the trial.
    extractor = FrozenFeatureExtractor(
        cfg.feature_extractor.model_name,
        cfg.feature_extractor.layer,
        cfg.feature_extractor.input_size,
        device,
    )
    image_size = SyntheticAugmentationConfig().image_size
    real_framed = render_real(real_raw, image_size)
    real_features = extractor(real_framed, cfg.feature_batch_size)

    if cfg.mmd_sigma == "median":
        sigma = median_heuristic_sigma(real_features)
    else:
        sigma = float(cfg.mmd_sigma)
    logger.info("RBF sigma = %.4f (real features: %s)", sigma, tuple(real_features.shape))

    fixed_params = OmegaConf.to_container(cfg.fixed_params, resolve=True) or {}
    unknown = set(fixed_params) - set(SEARCH_SPACE)
    if unknown:
        raise ValueError(
            f"fixed_params contains unknown keys: {sorted(unknown)}. "
            f"Valid keys: {sorted(SEARCH_SPACE)}"
        )
    if fixed_params:
        logger.info("Pinning %d parameter(s): %s", len(fixed_params), fixed_params)

    def objective(trial: optuna.Trial) -> float:
        config = sample_synthetic_config(trial, fixed_params)
        syn_aug = render_augmented_synthetic(syn_raw, config, cfg.augmentations_per_image)
        syn_features = extractor(syn_aug, cfg.feature_batch_size)
        return mmd_rbf(real_features, syn_features, sigma)

    storage = cfg.get("storage")
    if storage:
        Path(storage.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        direction="minimize",
        study_name=cfg.study_name,
        storage=storage,
        load_if_exists=cfg.load_if_exists,
        sampler=optuna.samplers.TPESampler(seed=cfg.sample_seed),
    )
    study.optimize(objective, n_trials=cfg.n_trials)

    logger.info("Best MMD (Maximum Mean Discrepancy): %.6f", study.best_value)
    logger.info("Best params:\n%s", json.dumps(study.best_params, indent=2))

    # Re-materialize the best config (FixedTrial needs every search-space key,
    # so merge the trial's drawn params with the pinned ones).
    fixed_trial_params = {**fixed_params, **study.best_trial.params}
    best_config = sample_synthetic_config(optuna.trial.FixedTrial(fixed_trial_params), fixed_params)
    write_best_yaml(cfg.output_yaml, best_config)
    logger.info("Wrote best config to %s", cfg.output_yaml)


if __name__ == "__main__":
    main()
