from typing import Any

import cv2
import numpy as np
import albumentations as A  # noqa: N812
from albumentations import RandomScale
from albumentations.core.type_definitions import Targets
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.geometric.functional import resize

IMAGE_SIZE = 105
SQUEEZE_RATIO = 1 / 2.5
SCALE_LIMIT = 0.15
EVAL_SCALE_LIMIT = 0.4
ROTATE_BOUNDS = (-45, 45)
SHEAR_BOUNDS = (-15, 15)
BLUR_LIMIT = (2.5, 3.5)
NOISE_MEAN_RANGE = (0.0, 0.0)
NOISE_STD_RANGE = (0.05, 0.15)
ROT_FLIP_PROB = 0.5


def add_greyscale_gradient(
    image: np.ndarray, gradient_min: tuple = (20, 100), gradient_max: tuple = (140, 220)
) -> np.ndarray:
    """Adds a random linear gradient overlay to a greyscale image.

    This function generates a 2D linear gradient in a random direction and subtracts
    it from the input image to simulate lighting variations. The gradient direction is
    determined by a random 2D vector, and its intensity range is randomly sampled from
    the specified bounds. This augmentation helps models become more robust to varying
    lighting conditions and background gradients.

    The gradient is normalized to [0, 1] and then scaled to the range [gradient_min, gradient_max]
    before being subtracted from the image. The result is clipped to [0, 255] to ensure
    valid pixel values.

    Args:
        image: A 2D NumPy array representing a greyscale image with values in [0, 255].
            The array dtype will be preserved in the output.
        gradient_min: A tuple of two floats (min_bound, max_bound) defining the range
            from which the minimum gradient intensity is randomly sampled. Default is
            (20, 100), meaning the gradient minimum will be between 20 and 100.
        gradient_max: A tuple of two floats (min_bound, max_bound) defining the range
            from which the maximum gradient intensity is randomly sampled. Default is
            (140, 220), meaning the gradient maximum will be between 140 and 220.

    Returns:
        A 2D NumPy array of the same shape and dtype as the input, representing the
        image with the gradient overlay applied.

    Note:
        The gradient is subtracted from the image, so higher gradient values result in
        darker regions. The final result is clipped to ensure valid pixel values.
    """
    # Store the original image dtype
    original_dtype = image.dtype
    # Convert the image to a float
    image = image.astype(float)
    # Draw a random 2D vector
    random_vector = np.random.randn(2, 1)
    # Create the x,y grid for the image
    x = np.linspace(0, 1, image.shape[1])
    y = np.linspace(0, 1, image.shape[0])
    x, y = np.meshgrid(x, y)
    # Create the gradient image
    gradient_image = np.dot(random_vector.T, np.array([x.flatten(), y.flatten()])).reshape(
        image.shape
    ) / np.linalg.norm(random_vector)
    # Normalize the gradient image
    gradient_image = (gradient_image - np.min(gradient_image)) / (
        np.max(gradient_image) - np.min(gradient_image)
    )
    # Choose a random value for the gradient bounds
    gradient_min = np.random.uniform(gradient_min[0], gradient_min[1])
    gradient_max = np.random.uniform(gradient_max[0], gradient_max[1])
    # Scale the gradient image
    gradient_image = gradient_min + (gradient_max - gradient_min) * gradient_image
    # Add the gradient to the image
    image = image - gradient_image
    # Clip the image and convert it back to the original dtype
    image = np.clip(image, 0, 255).astype(original_dtype)

    return image


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
        interpolation: int,
        **params: Any,
    ) -> np.ndarray:
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

    def apply(self, img: np.ndarray, interpolation: int, **params: Any) -> np.ndarray:
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


def augmentation_pipeline(image: np.ndarray, image_type: str, aug_prob: float) -> np.ndarray:
    """Applies the appropriate augmentation pipeline based on image type.

    This function routes images to either the synthetic or real image augmentation
    pipeline depending on the image_type parameter. Synthetic images (e.g., rendered
    text) undergo different augmentations than real images (e.g., photographs) to
    better simulate their respective real-world variations.

    The synthetic pipeline includes gradient addition, stronger blur, and noise to
    simulate printing and scanning artifacts. The real pipeline focuses on geometric
    and color transformations without synthetic artifacts.

    Args:
        image: Input image as a NumPy array. Should be a greyscale image with values
            in the range [0, 255].
        image_type: The type of image determining which pipeline to use. Must be
            either "synthetic" for rendered/generated images or "real" for
            photographs or scanned images.
        aug_prob: The probability (0.0 to 1.0) of applying each individual
            augmentation in the pipeline. Higher values result in more aggressive
            augmentation. A value of 1.0 always applies all augmentations.

    Returns:
        The augmented image as a NumPy array with the same shape as the input.
        The output will always be (105, 105) due to the cropping step in both pipelines.

    Raises:
        ValueError: If image_type is not "synthetic" or "real".

    Note:
        Both pipelines include resizing, random scaling, cropping, and various
        geometric and photometric transformations. See _synthetic_image_pipeline
        and _real_image_pipeline for details.
    """
    # Run the correct pipeline based on the image type
    if image_type == "synthetic":
        return _synthetic_image_pipeline(image, aug_prob)
    elif image_type == "real":
        return _real_image_pipeline(image, aug_prob)
    else:
        raise ValueError("The image type must be either 'synthetic' or 'real'.")


def _synthetic_image_pipeline(image: np.ndarray, aug_prob: float) -> np.ndarray:
    """Applies augmentations specifically designed for synthetic/rendered images.

    This pipeline is tailored for synthetically generated images (e.g., rendered text,
    computer-generated graphics) to make them appear more realistic by simulating
    real-world artifacts and variations. The augmentation sequence includes:

    1. Greyscale gradient (simulates lighting and background variations)
    2. Height resize with width squeezing (consistent dimensions with aspect variation)
    3. Random width scaling (±15% variation)
    4. Image inversion (simulates light-on-dark vs dark-on-light)
    5. Affine transformations (rotation ±45°, shear ±15°)
    6. Random cropping to 105x105
    7. Gaussian blur (simulates camera/scan blur)
    8. Brightness/contrast adjustments
    9. Gaussian noise (simulates sensor noise and compression artifacts)
    10. 90° rotations and flips (orientation variations)

    This pipeline is more aggressive than the real image pipeline, particularly
    with blur and noise, to bridge the domain gap between synthetic and real images.

    Args:
        image: Input synthetic image as a NumPy array. Should be a greyscale image
            with values in the range [0, 255].
        aug_prob: The probability (0.0 to 1.0) of applying each probabilistic
            augmentation. A value of 0.0 applies only always_apply transforms,
            while 1.0 applies all augmentations.

    Returns:
        The augmented image as a NumPy array with shape (105, 105), ready for
        model input.

    Note:
        The greyscale gradient is applied first with probability aug_prob, before
        the albumentations pipeline. All geometric parameters are defined as module-level
        constants (IMAGE_SIZE, SQUEEZE_RATIO, SCALE_LIMIT, etc.).
    """
    # Add the greyscale gradient
    if np.random.rand() < aug_prob:
        image = add_greyscale_gradient(image)
    # Create the augmentation pipeline
    augmentations = A.Compose(
        [
            ResizeHeightSqueezeWidth(IMAGE_SIZE, SQUEEZE_RATIO, p=1.0),
            RandomWidthScale(scale_limit=SCALE_LIMIT, p=1.0),
            A.InvertImg(p=aug_prob),
            A.Affine(
                rotate=ROTATE_BOUNDS,
                shear=SHEAR_BOUNDS,
                border_mode=cv2.BORDER_REFLECT,
                p=aug_prob,
            ),
            A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
            A.GaussianBlur(blur_limit=0, sigma_limit=BLUR_LIMIT, p=aug_prob),
            A.RandomBrightnessContrast(p=aug_prob),
            A.GaussNoise(std_range=NOISE_STD_RANGE, mean_range=NOISE_MEAN_RANGE, p=aug_prob),
            A.RandomRotate90(p=ROT_FLIP_PROB),
            A.HorizontalFlip(p=ROT_FLIP_PROB),
            A.VerticalFlip(p=ROT_FLIP_PROB),
        ]
    )

    return augmentations(image=image)["image"]


def _real_image_pipeline(image: np.ndarray, aug_prob: float) -> np.ndarray:
    """Applies augmentations specifically designed for real photographs and scans.

    This pipeline is optimized for real-world images (photographs, scans, camera captures)
    and focuses on geometric and photometric variations without adding synthetic artifacts
    like blur or noise that are already present in real images. The augmentation sequence
    includes:

    1. Height resize with width squeezing (consistent dimensions with aspect variation)
    2. Random width scaling (±15% variation)
    3. Image inversion (simulates light-on-dark vs dark-on-light)
    4. Affine transformations (rotation ±45°, shear ±15°)
    5. Random cropping to 105x105
    6. Brightness/contrast adjustments (compensates for lighting variations)
    7. 90° rotations and flips (orientation variations)

    Compared to the synthetic pipeline, this omits Gaussian blur and noise since real
    images already contain these artifacts naturally. This prevents over-processing that
    could degrade image quality.

    Args:
        image: Input real image as a NumPy array. Should be a greyscale image
            with values in the range [0, 255].
        aug_prob: The probability (0.0 to 1.0) of applying each probabilistic
            augmentation. A value of 0.0 applies only always_apply transforms,
            while 1.0 applies all augmentations.

    Returns:
        The augmented image as a NumPy array with shape (105, 105), ready for
        model input.

    Note:
        Unlike the synthetic pipeline, no greyscale gradient or noise is added.
        All geometric parameters are defined as module-level constants (IMAGE_SIZE,
        SQUEEZE_RATIO, SCALE_LIMIT, etc.).
    """
    # Create the real image pipeline
    real_augmentations = A.Compose(
        [
            ResizeHeightSqueezeWidth(IMAGE_SIZE, SQUEEZE_RATIO, p=1.0),
            RandomWidthScale(scale_limit=SCALE_LIMIT, p=1.0),
            A.InvertImg(p=aug_prob),
            A.Affine(
                rotate=ROTATE_BOUNDS,
                shear=SHEAR_BOUNDS,
                border_mode=cv2.BORDER_REFLECT,
                p=aug_prob,
            ),
            A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
            A.RandomBrightnessContrast(p=aug_prob),
            A.RandomRotate90(p=ROT_FLIP_PROB),
            A.HorizontalFlip(p=ROT_FLIP_PROB),
            A.VerticalFlip(p=ROT_FLIP_PROB),
        ]
    )

    return real_augmentations(image=image)["image"]


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
        image: Input image as a NumPy array. Should be a greyscale image with
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

    Note:
        The scale_limit of 0.4 (±40%) is higher than the training scale_limit of
        0.15 (±15%) to ensure good coverage during evaluation.
    """
    # Create the augmentations
    augmentations = A.Compose(
        [
            ResizeHeightSqueezeWidth(IMAGE_SIZE, SQUEEZE_RATIO, p=1.0),
            RandomWidthScale(scale_limit=EVAL_SCALE_LIMIT, p=1.0),
            A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0),
        ]
    )
    # Create the image crops
    image_crops = []
    for _ in range(num_image_crops):
        # Apply the augmentations
        image_crops.append(augmentations(image=image)["image"])
    # Convert to numpy array
    image_crops = np.array(image_crops)

    return image_crops
