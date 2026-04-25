from typing import Any, Literal

import cv2
import numpy as np
import albumentations as A  # noqa: N812
from pydantic import Field, BaseModel, ConfigDict
from albumentations import RandomScale
from albumentations.core.type_definitions import Targets
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.geometric.functional import resize

IMAGE_SIZE = 105
SQUEEZE_RATIO = 1 / 2.5
SCALE_LIMIT = 0.4
ROTATE_BOUNDS = (-3, 3)
SHEAR_BOUNDS = (-3, 3)
BLUR_LIMIT = (0.5, 1.0)
NOISE_MEAN_RANGE = (0.0, 0.0)
NOISE_STD_RANGE = (0.008, 0.016)
ROT_FLIP_PROB = 0.5
INVERT_PROB = 0.5
AFFINE_PROB = 1.0
BLUR_PROB = 1.0
BRIGHTNESS_CONTRAST_PROB = 1.0
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
        scale limits, etc.) unchanged.
        """
        updates = {name: 0.0 for name in self.__class__.model_fields if name.endswith("_prob")}
        return self.model_copy(update=updates)


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
    squeeze_ratio: float = Field(
        default=SQUEEZE_RATIO,
        gt=0.0,
        description="Width-squeeze factor used by ResizeHeightSqueezeWidth.",
    )
    scale_limit: float = Field(
        default=SCALE_LIMIT,
        ge=0.0,
        description="Random width-scale limit used by RandomWidthScale.",
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
    squeeze_ratio: float = Field(
        default=SQUEEZE_RATIO,
        gt=0.0,
        description="Width-squeeze factor used by ResizeHeightSqueezeWidth.",
    )
    scale_limit: float = Field(
        default=SCALE_LIMIT,
        ge=0.0,
        description="Random width-scale limit used by RandomWidthScale.",
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
    squeeze_ratio: float = Field(
        default=SQUEEZE_RATIO,
        gt=0.0,
        description="Width-squeeze factor used by ResizeHeightSqueezeWidth.",
    )
    scale_limit: float = Field(
        default=SCALE_LIMIT,
        ge=0.0,
        description="Random width-scale limit used by RandomWidthScale.",
    )


def add_grayscale_gradient(
    image: np.ndarray,
    fg_range: tuple = GRADIENT_FG_RANGE,
    bg_range: tuple = GRADIENT_BG_RANGE,
    a_range: tuple = GRADIENT_A_RANGE,
) -> np.ndarray:
    """Applies directional gradient shading that faithfully replicates affine2d.apply2.

    The legacy DeepFont C extension (affine2d.so) applies gradient shading in two steps:
      1. Intensity remap: pixel values are linearly mapped from [0, 255] → [bg, fg],
         so dark pixels get the background tone and bright pixels get the foreground tone.
      2. Multiplicative spatial gradient: a linear scale centered on the image is applied
         in a random direction θ, with amplitude controlled by `a`.

    Combined formula per pixel at (col, row):
        normalized = pixel * (fg - bg) / 255 + bg
        grad_pos   = (col - w/2) * cos(θ) + (row - h/2) * sin(θ)
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


class RandomWidthScale(RandomScale):
    """Randomly scales only the width of an image while preserving height.

    This augmentation inherits from albumentations' RandomScale but modifies the behavior
    to only scale the image width, keeping the height constant. This is useful for
    simulating variations in character spacing or aspect ratios while maintaining a
    consistent vertical dimension.

    The width is scaled by a random factor, but will never be smaller than the height
    to avoid overly compressed images.

    Inherits all parameters from albumentations.RandomScale, including:
        - scale_limit: The range for random scaling (e.g., 0.15 means ±15%)
        - interpolation: OpenCV interpolation method
        - p: Probability of applying the transform

    Note:
        This class only overrides the apply method to implement width-only scaling.
    """

    def apply(
        self,
        img: np.ndarray,
        scale: float,
        **params: Any,
    ) -> np.ndarray:
        interpolation = params.get("interpolation", cv2.INTER_LINEAR)
        height, width = img.shape[:2]
        new_size = int(height), max(int(width * scale), int(height))
        return resize(img, new_size, interpolation)


class ResizeHeightSqueezeWidth(DualTransform):
    """Resizes an image to a specified height while applying a scaling factor to the width.

    This transform is particularly useful for text and font images where maintaining a
    consistent height is important, but the width may need to be compressed or expanded.
    The transform first resizes the image to the target height, then applies a width
    scaling factor to create the final dimensions.

    The width is never allowed to become smaller than the height, preventing overly
    compressed aspect ratios. This ensures that characters remain readable even after
    aggressive width squeezing.

    This is a DualTransform, meaning it can be applied to both images and masks in
    segmentation tasks.

    Args:
        height: The desired height of the output image in pixels. The output will
            always have exactly this height.
        width_scale: The scaling factor to apply to the width. For example, 0.5 will
            compress the width to half of what it would be if only height scaling
            was applied. Values < 1 squeeze the width, values > 1 expand it.
        interpolation: OpenCV interpolation flag specifying the resampling algorithm.
            Should be one of: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC,
            cv2.INTER_AREA, or cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.
        p: Probability of applying the transform. Only used if always_apply is False.
            Default: 1.

    Targets:
        image

    Image types:
        uint8, float32

    Example:
        >>> transform = ResizeHeightSqueezeWidth(height=105, width_scale=0.4)
        >>> # Image of shape (200, 300) becomes approximately (105, 126)
        >>> # Width would be ~157 with only height scaling, but 0.4 factor makes it 126
    """

    _targets = Targets.IMAGE

    def __init__(
        self,
        height: int,
        width_scale: float,
        interpolation: int = cv2.INTER_LINEAR,
        p: float = 1,
    ):
        super().__init__(p)
        self.height = height
        self.width_scale = width_scale
        self.interpolation = interpolation

    def apply(self, img: np.ndarray, *args: Any, **params: Any) -> np.ndarray:
        interpolation = params.get("interpolation", self.interpolation)
        height, width = img.shape[:2]
        height_scale = self.height / height
        # Don't allow the width to be squeezed below the height
        new_width = max(int(height_scale * self.width_scale * width), self.height)
        return resize(
            img,
            (self.height, new_width),
            interpolation=interpolation,
        )

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return "height", "width_scale", "interpolation"


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
                ResizeHeightSqueezeWidth(cfg.image_size, cfg.squeeze_ratio, p=1.0),
                RandomWidthScale(scale_limit=cfg.scale_limit, p=1.0),
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
                ResizeHeightSqueezeWidth(cfg.image_size, cfg.squeeze_ratio, p=1.0),
                RandomWidthScale(scale_limit=cfg.scale_limit, p=1.0),
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
    random crops because ``RandomWidthScale`` and ``RandomCrop`` are
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
                ResizeHeightSqueezeWidth(self.config.image_size, self.config.squeeze_ratio, p=1.0),
                RandomWidthScale(scale_limit=self.config.scale_limit, p=1.0),
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

    This pipeline is designed for model evaluation and inference, where test-time
    augmentation (TTA) can improve prediction robustness. Unlike training pipelines,
    it uses only geometric augmentations (no color or blur) to create multiple views
    of the same image. The model predictions on all crops can be averaged or ensembled
    for more reliable results.

    The augmentation sequence for each crop:
    1. Height resize with width squeezing (2.5x squeeze factor)
    2. Random width scaling (±40% variation, more aggressive than training)
    3. Random cropping to 105x105

    All augmentations are applied with always_apply=True, meaning each crop is
    guaranteed to be different. No rotation, flip, or photometric augmentations
    are applied to preserve the image's semantic content.

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
        >>> # Feed all crops to model and average predictions
    """
    return EvalAugmentationPipeline()(image, num_image_crops)
