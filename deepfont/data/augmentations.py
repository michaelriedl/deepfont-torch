from typing import Any, Literal

import cv2
import numpy as np
import albumentations as A  # noqa: N812
from pydantic import Field, BaseModel, ConfigDict, model_validator
from albumentations.core.type_definitions import Targets
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.geometric.functional import resize

IMAGE_SIZE = 105
# Training-time aspect ratio bounds. The DeepFont paper combines a constant
# 2.5x squeeze (Section 3.1) with a per-image jitter drawn from Uniform[5/6, 7/6]
# (Section 2.3, augmentation 6), giving a post-squeeze width-to-height ratio
# of Uniform[2.5*5/6, 2.5*7/6] = Uniform[2.083, 2.917].
ASPECT_RATIO_LOW_TRAIN = 2.5 * 5 / 6
ASPECT_RATIO_HIGH_TRAIN = 2.5 * 7 / 6
# Eval-time aspect ratio bounds (Section 3.1, "Testing Details": squeeze
# ratios drawn from Uniform[1.5, 3.5]).
ASPECT_RATIO_LOW_EVAL = 1.5
ASPECT_RATIO_HIGH_EVAL = 3.5
ROTATE_BOUNDS = (-3, 3)
SHEAR_BOUNDS = (-3, 3)
BLUR_LIMIT = (0.5, 1.0)
NOISE_MEAN_RANGE = (0.0, 0.0)
NOISE_STD_RANGE = (0.008, 0.016)
ROT_FLIP_PROB = 0.0
INVERT_PROB = 0.0
AFFINE_PROB = 1.0
BLUR_PROB = 1.0
BRIGHTNESS_CONTRAST_PROB = 0.0
NOISE_PROB = 1.0
GRADIENT_PROB = 1.0
GRADIENT_FG_RANGE = (140, 220)
GRADIENT_BG_RANGE = (20, 100)
GRADIENT_A_RANGE = (0.4, 0.6)


class _AugmentationConfigBase(BaseModel):
    """Common helpers for augmentation configs.

    Provides ``with_stochastic_disabled`` so callers can derive a copy with
    every probability field zeroed (used by validation splits to disable
    stochastic augmentation without rebuilding the dataset).
    """

    model_config = ConfigDict(frozen=True)

    def with_stochastic_disabled(self):
        """Return a copy of this config with every *_prob field set to 0.

        The base ``model_copy`` keeps non-probability fields (image size,
        aspect-ratio bounds, etc.) unchanged.
        """
        updates = {name: 0.0 for name in self.__class__.model_fields if name.endswith("_prob")}
        return self.model_copy(update=updates)

    @model_validator(mode="after")
    def _validate_aspect_ratio_bounds(self):
        """Ensure aspect_ratio_low is at least 1.0 and not greater than _high.

        ``aspect_ratio_low >= 1.0`` is required because the downstream
        ``RandomCrop`` step assumes the post-resize width is at least the
        target height; values below 1 would force a clamp and silently
        break the configured distribution.
        """
        low = getattr(self, "aspect_ratio_low", None)
        high = getattr(self, "aspect_ratio_high", None)
        if low is None or high is None:
            return self
        if low < 1.0:
            raise ValueError(
                f"aspect_ratio_low={low} must be >= 1.0; values below 1 would "
                f"clamp at the target height and silently flatten the distribution."
            )
        if low > high:
            raise ValueError(f"aspect_ratio_low={low} must be <= aspect_ratio_high={high}.")
        return self


class SyntheticAugmentationConfig(_AugmentationConfigBase):
    """Hyperparameters for the synthetic-image augmentation pipeline.

    Mirrors the kwargs of SyntheticAugmentationPipeline so the same values
    can be supplied via Hydra configs and validated by pydantic. Defaults
    match the module-level constants in this module.
    """

    image_size: int = Field(
        default=IMAGE_SIZE,
        gt=0,
        description="Target output crop size (square) in pixels.",
    )
    aspect_ratio_low: float = Field(
        default=ASPECT_RATIO_LOW_TRAIN,
        ge=1.0,
        description=(
            "Inclusive lower bound on the post-resize width/height ratio drawn "
            "uniformly per image. Must be >= 1.0."
        ),
    )
    aspect_ratio_high: float = Field(
        default=ASPECT_RATIO_HIGH_TRAIN,
        ge=1.0,
        description=(
            "Inclusive upper bound on the post-resize width/height ratio drawn "
            "uniformly per image. Must be >= aspect_ratio_low."
        ),
    )
    rotate_bounds: tuple[float, float] = Field(
        default=ROTATE_BOUNDS,
        description="(min, max) rotation angles in degrees for the affine step.",
    )
    shear_bounds: tuple[float, float] = Field(
        default=SHEAR_BOUNDS,
        description="(min, max) shear angles in degrees for the affine step.",
    )
    blur_limit: tuple[float, float] = Field(
        default=BLUR_LIMIT,
        description="(min, max) sigma range for GaussianBlur.",
    )
    noise_mean_range: tuple[float, float] = Field(
        default=NOISE_MEAN_RANGE,
        description="(min, max) mean range for GaussNoise.",
    )
    noise_std_range: tuple[float, float] = Field(
        default=NOISE_STD_RANGE,
        description="(min, max) standard-deviation range for GaussNoise.",
    )
    rot_flip_prob: float = Field(
        default=ROT_FLIP_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for RandomRotate90, HorizontalFlip, VerticalFlip.",
    )
    invert_prob: float = Field(
        default=INVERT_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for InvertImg.",
    )
    affine_prob: float = Field(
        default=AFFINE_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for the Affine (rotate + shear) step.",
    )
    blur_prob: float = Field(
        default=BLUR_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for GaussianBlur.",
    )
    brightness_contrast_prob: float = Field(
        default=BRIGHTNESS_CONTRAST_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for RandomBrightnessContrast.",
    )
    noise_prob: float = Field(
        default=NOISE_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for GaussNoise.",
    )
    gradient_prob: float = Field(
        default=GRADIENT_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for the grayscale gradient overlay applied in __call__.",
    )
    gradient_fg_range: tuple[float, float] = Field(
        default=GRADIENT_FG_RANGE,
        description="(min, max) foreground intensity for the gradient overlay.",
    )
    gradient_bg_range: tuple[float, float] = Field(
        default=GRADIENT_BG_RANGE,
        description="(min, max) background intensity for the gradient overlay.",
    )
    gradient_a_range: tuple[float, float] = Field(
        default=GRADIENT_A_RANGE,
        description="(min, max) gradient amplitude factor.",
    )


class RealAugmentationConfig(_AugmentationConfigBase):
    """Hyperparameters for the real-image augmentation pipeline.

    Mirrors the kwargs of RealAugmentationPipeline. Defaults match the
    module-level constants in this module.
    """

    image_size: int = Field(
        default=IMAGE_SIZE,
        gt=0,
        description="Target output crop size (square) in pixels.",
    )
    aspect_ratio_low: float = Field(
        default=ASPECT_RATIO_LOW_TRAIN,
        ge=1.0,
        description=(
            "Inclusive lower bound on the post-resize width/height ratio drawn "
            "uniformly per image. Must be >= 1.0."
        ),
    )
    aspect_ratio_high: float = Field(
        default=ASPECT_RATIO_HIGH_TRAIN,
        ge=1.0,
        description=(
            "Inclusive upper bound on the post-resize width/height ratio drawn "
            "uniformly per image. Must be >= aspect_ratio_low."
        ),
    )
    rotate_bounds: tuple[float, float] = Field(
        default=ROTATE_BOUNDS,
        description="(min, max) rotation angles in degrees for the affine step.",
    )
    shear_bounds: tuple[float, float] = Field(
        default=SHEAR_BOUNDS,
        description="(min, max) shear angles in degrees for the affine step.",
    )
    rot_flip_prob: float = Field(
        default=ROT_FLIP_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for RandomRotate90, HorizontalFlip, VerticalFlip.",
    )
    invert_prob: float = Field(
        default=INVERT_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for InvertImg.",
    )
    affine_prob: float = Field(
        default=AFFINE_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for the Affine (rotate + shear) step.",
    )
    brightness_contrast_prob: float = Field(
        default=BRIGHTNESS_CONTRAST_PROB,
        ge=0.0,
        le=1.0,
        description="Probability for RandomBrightnessContrast.",
    )


class EvalAugmentationConfig(_AugmentationConfigBase):
    """Hyperparameters for the eval (TTA) augmentation pipeline.

    Mirrors the kwargs of EvalAugmentationPipeline. All transforms run with
    p=1.0, so this config only contains the geometric parameters.
    """

    image_size: int = Field(
        default=IMAGE_SIZE,
        gt=0,
        description="Target output crop size (square) in pixels.",
    )
    aspect_ratio_low: float = Field(
        default=ASPECT_RATIO_LOW_EVAL,
        ge=1.0,
        description=(
            "Inclusive lower bound on the post-resize width/height ratio drawn "
            "uniformly per crop. Must be >= 1.0."
        ),
    )
    aspect_ratio_high: float = Field(
        default=ASPECT_RATIO_HIGH_EVAL,
        ge=1.0,
        description=(
            "Inclusive upper bound on the post-resize width/height ratio drawn "
            "uniformly per crop. Must be >= aspect_ratio_low."
        ),
    )


def add_grayscale_gradient(
    image: np.ndarray,
    fg_range: tuple = GRADIENT_FG_RANGE,
    bg_range: tuple = GRADIENT_BG_RANGE,
    a_range: tuple = GRADIENT_A_RANGE,
) -> np.ndarray:
    """Applies directional gradient shading that faithfully replicates affine2d.apply2.

    The legacy DeepFont C extension (affine2d.so) applies gradient shading in two steps:
      1. Intensity remap: pixel values are linearly mapped from [0, 255] to [bg, fg],
         so dark pixels get the background tone and bright pixels get the foreground tone.
      2. Multiplicative spatial gradient: a linear scale centered on the image is applied
         in a random direction theta, with amplitude controlled by `a`.

    Combined formula per pixel at (col, row):
        normalized = pixel * (fg - bg) / 255 + bg
        grad_pos   = (col - w/2) * cos(theta) + (row - h/2) * sin(theta)
        scale      = 1.0 + a * grad_pos / min(h, w)
        output     = normalized * scale

    Args:
        image: A 2D NumPy array representing a grayscale image with values in [0, 255].
            The array dtype will be preserved in the output.
        fg_range: (min, max) for the foreground (bright) intensity level.
            Default matches the legacy code: (140, 220).
        bg_range: (min, max) for the background (dark) intensity level.
            Default matches the legacy code: (20, 100).
        a_range: (min, max) for the gradient amplitude factor.
            Default matches the legacy code: (0.4, 0.6).

    Returns:
        A 2D NumPy array of the same shape and dtype as the input, with gradient
        shading applied and pixel values clipped to [0, 255].
    """
    original_dtype = image.dtype
    h, w = image.shape[:2]

    fg = np.random.uniform(fg_range[0], fg_range[1])
    bg = np.random.uniform(bg_range[0], bg_range[1])
    theta = np.random.uniform(0, 2 * np.pi)
    a = np.random.uniform(a_range[0], a_range[1])

    normalized = image.astype(float) * (fg - bg) / 255.0 + bg

    cols = np.arange(w) - w / 2.0
    rows = np.arange(h) - h / 2.0
    col_grid, row_grid = np.meshgrid(cols, rows)
    projection = col_grid * np.cos(theta) + row_grid * np.sin(theta)
    scale = 1.0 + a * projection / min(h, w)

    return np.clip(normalized * scale, 0, 255).astype(original_dtype)


class TargetAspectRatioResize(DualTransform):
    """Resize an image so its post-resize width/height ratio is a target value.

    The output height is fixed to ``height``; the output width is drawn per
    application as ``round(height * r)`` where ``r ~ Uniform[low, high]``.
    This makes the post-resize aspect ratio independent of the input image's
    dimensions and matches the DeepFont paper's test-time procedure
    (Section 3.1: "squeezed in width by three different random ratios, all
    drawn from a uniform distribution between 1.5 and 3.5") and training-time
    procedure (the constant 2.5 squeeze combined with the [5/6, 7/6] jitter
    of augmentation 6, giving Uniform[2.083, 2.917]).

    The earlier two-step pipeline (ResizeHeightSqueezeWidth then
    RandomWidthScale) inadvertently made the post-resize aspect ratio depend
    on the input aspect ratio, so the configured distribution only matched
    the paper for one specific input aspect ratio.

    Args:
        height: Target output height in pixels.
        aspect_ratio_low: Inclusive lower bound on the post-resize aspect ratio.
        aspect_ratio_high: Inclusive upper bound on the post-resize aspect ratio.
        interpolation: OpenCV interpolation flag.
        p: Probability of applying the transform.

    Targets:
        image
    """

    _targets = Targets.IMAGE

    def __init__(
        self,
        height: int,
        aspect_ratio_low: float,
        aspect_ratio_high: float,
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1,
    ):
        super().__init__(p)
        if aspect_ratio_low < 1.0:
            raise ValueError(
                f"aspect_ratio_low={aspect_ratio_low} must be >= 1.0; values below 1 "
                f"would clamp at the target height."
            )
        if aspect_ratio_low > aspect_ratio_high:
            raise ValueError(
                f"aspect_ratio_low={aspect_ratio_low} must be <= "
                f"aspect_ratio_high={aspect_ratio_high}."
            )
        self.height = height
        self.aspect_ratio_low = aspect_ratio_low
        self.aspect_ratio_high = aspect_ratio_high
        self.interpolation = interpolation

    def get_params(self) -> dict[str, Any]:
        return {
            "aspect_ratio": float(np.random.uniform(self.aspect_ratio_low, self.aspect_ratio_high))
        }

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        interpolation = params.get("interpolation", self.interpolation)
        aspect_ratio = float(params.get("aspect_ratio", self.aspect_ratio_low))
        new_width = max(round(self.height * aspect_ratio), self.height)
        return resize(img, (self.height, new_width), interpolation=interpolation)

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "height", "aspect_ratio_low", "aspect_ratio_high", "interpolation"


def _resize_step(cfg) -> TargetAspectRatioResize:
    """Build the configured TargetAspectRatioResize for any augmentation config."""
    return TargetAspectRatioResize(
        height=cfg.image_size,
        aspect_ratio_low=cfg.aspect_ratio_low,
        aspect_ratio_high=cfg.aspect_ratio_high,
        p=1.0,
    )


class SyntheticAugmentationPipeline:
    """Builds the synthetic augmentation pipeline once and reuses it across calls.

    Creating an ``A.Compose`` object is expensive relative to applying it.
    This class pays the construction cost once in ``__init__`` and reuses the
    compiled pipeline on every ``__call__``, giving a 4-7x speedup when the
    same pipeline is applied to many images (e.g. inside a Dataset).

    Per-augmentation probabilities live on the config object. Assigning a new
    config (e.g. via ``pipeline.config = pipeline.config.with_stochastic_disabled()``)
    rebuilds the underlying Compose so validation splits can disable
    stochastic augmentation without rebuilding the dataset.

    Args:
        config: Hyperparameters for the transform list. When None, a default
            SyntheticAugmentationConfig is used (matching the module-level
            constants).
    """

    def __init__(self, config: SyntheticAugmentationConfig | None = None) -> None:
        self._config = config if config is not None else SyntheticAugmentationConfig()
        self._compose = self._build()

    def _build(self) -> A.Compose:
        cfg = self._config
        return A.Compose(
            [
                _resize_step(cfg),
                A.InvertImg(p=cfg.invert_prob),
                A.Affine(
                    rotate=cfg.rotate_bounds,
                    shear=cfg.shear_bounds,
                    border_mode=cv2.BORDER_REFLECT,
                    p=cfg.affine_prob,
                ),
                A.RandomCrop(cfg.image_size, cfg.image_size, p=1.0),
                A.GaussianBlur(blur_limit=0, sigma_limit=cfg.blur_limit, p=cfg.blur_prob),
                A.RandomBrightnessContrast(p=cfg.brightness_contrast_prob),
                A.GaussNoise(
                    std_range=cfg.noise_std_range,
                    mean_range=cfg.noise_mean_range,
                    p=cfg.noise_prob,
                ),
                A.RandomRotate90(p=cfg.rot_flip_prob),
                A.HorizontalFlip(p=cfg.rot_flip_prob),
                A.VerticalFlip(p=cfg.rot_flip_prob),
            ]
        )

    @property
    def config(self) -> SyntheticAugmentationConfig:
        return self._config

    @config.setter
    def config(self, value: SyntheticAugmentationConfig) -> None:
        self._config = value
        self._compose = self._build()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = self._compose(image=image)["image"]
        if np.random.rand() < self._config.gradient_prob:
            image = add_grayscale_gradient(
                image,
                fg_range=self._config.gradient_fg_range,
                bg_range=self._config.gradient_bg_range,
                a_range=self._config.gradient_a_range,
            )
        return image


class RealAugmentationPipeline:
    """Builds the real-image augmentation pipeline once and reuses it across calls.

    Mirrors ``SyntheticAugmentationPipeline`` but uses the gentler real-image
    transform list (no gradient overlay, no blur, no noise).

    Args:
        config: Hyperparameters for the transform list. When None, a default
            RealAugmentationConfig is used (matching the module-level constants).
    """

    def __init__(self, config: RealAugmentationConfig | None = None) -> None:
        self._config = config if config is not None else RealAugmentationConfig()
        self._compose = self._build()

    def _build(self) -> A.Compose:
        cfg = self._config
        return A.Compose(
            [
                _resize_step(cfg),
                A.InvertImg(p=cfg.invert_prob),
                A.Affine(
                    rotate=cfg.rotate_bounds,
                    shear=cfg.shear_bounds,
                    border_mode=cv2.BORDER_REFLECT,
                    p=cfg.affine_prob,
                ),
                A.RandomCrop(cfg.image_size, cfg.image_size, p=1.0),
                A.RandomBrightnessContrast(p=cfg.brightness_contrast_prob),
                A.RandomRotate90(p=cfg.rot_flip_prob),
                A.HorizontalFlip(p=cfg.rot_flip_prob),
                A.VerticalFlip(p=cfg.rot_flip_prob),
            ]
        )

    @property
    def config(self) -> RealAugmentationConfig:
        return self._config

    @config.setter
    def config(self, value: RealAugmentationConfig) -> None:
        self._config = value
        self._compose = self._build()

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self._compose(image=image)["image"]


class EvalAugmentationPipeline:
    """Builds the eval TTA pipeline once and reuses it across calls.

    All transforms run with ``p=1.0`` so there are no per-augmentation
    probabilities. Multiple calls with the same image still produce different
    crops because the aspect-ratio resize and the random crop are both
    stochastic.

    The pipeline is constructed once in ``__init__`` and applied inside a loop
    in ``__call__``, so construction cost is paid at most once per dataset
    lifetime rather than once per image.

    Args:
        config: Hyperparameters for the transform list. When None, a default
            EvalAugmentationConfig is used (matching the module-level constants).
    """

    def __init__(self, config: EvalAugmentationConfig | None = None) -> None:
        self.config = config if config is not None else EvalAugmentationConfig()
        self._compose = A.Compose(
            [
                _resize_step(self.config),
                A.RandomCrop(self.config.image_size, self.config.image_size, p=1.0),
            ]
        )

    def __call__(self, image: np.ndarray, num_image_crops: int) -> np.ndarray:
        crops = [self._compose(image=image)["image"] for _ in range(num_image_crops)]
        return np.array(crops)


def augmentation_pipeline(
    image: np.ndarray, image_type: Literal["synthetic", "real"]
) -> np.ndarray:
    """Applies the appropriate augmentation pipeline based on image type.

    Routes images to either the synthetic or real image augmentation pipeline
    depending on the image_type parameter. Synthetic images (e.g., rendered
    text) undergo different augmentations than real images (e.g., photographs)
    to better simulate their respective real-world variations.

    The synthetic pipeline includes gradient addition, blur, and noise to
    simulate printing and scanning artifacts. The real pipeline focuses on
    geometric and color transformations without synthetic artifacts.

    The instantiated pipeline uses default per-augmentation probabilities; for
    fine-grained control build the pipeline class directly with a custom
    ``SyntheticAugmentationConfig`` or ``RealAugmentationConfig``.

    Args:
        image: Input image as a NumPy array. Should be a grayscale image with values
            in the range [0, 255].
        image_type: The type of image determining which pipeline to use. Must be
            either "synthetic" for rendered/generated images or "real" for
            photographs or scanned images.

    Returns:
        The augmented image as a NumPy array. The output will always be
        (105, 105) due to the cropping step in both pipelines.

    Raises:
        ValueError: If image_type is not "synthetic" or "real".
    """
    if image_type == "synthetic":
        return SyntheticAugmentationPipeline()(image)
    elif image_type == "real":
        return RealAugmentationPipeline()(image)
    else:
        raise ValueError("The image type must be either 'synthetic' or 'real'.")


def eval_pipeline(image: np.ndarray, num_image_crops: int) -> np.ndarray:
    """Creates multiple augmented crops for test-time augmentation during evaluation.

    Each crop is generated by sampling a target aspect ratio uniformly from
    the configured [aspect_ratio_low, aspect_ratio_high] range, resizing the
    image so the post-resize width/height matches that ratio, then taking a
    random 105x105 crop. This mirrors the test-time procedure described in
    Section 3.1 of the DeepFont paper.

    Args:
        image: Input image as a NumPy array. Should be a grayscale image with
            values in the range [0, 255].
        num_image_crops: The number of different augmented crops to generate from
            the input image. More crops provide better coverage but increase
            computational cost. Typical values range from 5 to 20.

    Returns:
        A NumPy array of shape (num_image_crops, 105, 105) containing all the
        augmented crops stacked along the first dimension.

    Example:
        >>> image = cv2.imread('font_sample.png', cv2.IMREAD_GRAYSCALE)
        >>> crops = eval_pipeline(image, num_image_crops=10)
        >>> crops.shape
        (10, 105, 105)
    """
    return EvalAugmentationPipeline()(image, num_image_crops)
