"""Tests for the augmentation pipelines in deepfont.data.augmentations.

These tests verify that all augmentation pipelines execute correctly and produce
outputs with the expected shapes, dtypes, and value ranges. They are deliberately
explicit about the albumentations API surface being used so that a version upgrade
that renames or removes a parameter produces a clear, descriptive failure rather than
a silent regression.

Test classes:
    TestConstants                       -- module-level hyper-parameter values
    TestAddGrayscaleGradient            -- standalone NumPy gradient function
    TestTargetAspectRatioResize         -- aspect-ratio-driven resize transform
    TestAugmentationPipelineDispatch    -- dispatcher routing and error handling
    TestSyntheticPipeline               -- end-to-end synthetic image pipeline
    TestRealPipeline                    -- end-to-end real image pipeline
    TestEvalPipeline                    -- test-time augmentation (TTA) eval pipeline
    TestAlbumentationsAPIContract       -- explicit tests for albumentations parameter names
                                           and import paths that have historically changed
"""

import cv2
import numpy as np
import pytest
import albumentations as A  # noqa: N812
from albumentations.core.type_definitions import Targets
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.geometric.functional import resize

from deepfont.data.augmentations import (
    BLUR_LIMIT,
    IMAGE_SIZE,
    SHEAR_BOUNDS,
    ROT_FLIP_PROB,
    ROTATE_BOUNDS,
    NOISE_STD_RANGE,
    GRADIENT_A_RANGE,
    NOISE_MEAN_RANGE,
    GRADIENT_BG_RANGE,
    GRADIENT_FG_RANGE,
    ASPECT_RATIO_LOW_EVAL,
    ASPECT_RATIO_HIGH_EVAL,
    ASPECT_RATIO_LOW_TRAIN,
    ASPECT_RATIO_HIGH_TRAIN,
    TargetAspectRatioResize,
    eval_pipeline,
    augmentation_pipeline,
    add_grayscale_gradient,
)

# Shared fixtures


@pytest.fixture
def wide_image() -> np.ndarray:
    """A wide grayscale text patch (landscape orientation)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(80, 400), dtype=np.uint8)


@pytest.fixture
def tall_image() -> np.ndarray:
    """A tall, narrow image that triggers the width clamping code paths."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(400, 80), dtype=np.uint8)


@pytest.fixture
def square_image() -> np.ndarray:
    """A square grayscale image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(200, 200), dtype=np.uint8)


# Module-level constants


class TestConstants:
    """Verify that all module-level hyperparameters have the expected values.

    Pinning these values means any accidental edit to the augmentations module
    immediately causes a test failure, giving the same protection as locking a
    config file.
    """

    def test_image_size(self):
        assert IMAGE_SIZE == 105

    def test_aspect_ratio_low_train(self):
        # 2.5 * 5/6, matching the paper's training-time post-squeeze AR.
        assert ASPECT_RATIO_LOW_TRAIN == pytest.approx(2.5 * 5 / 6)

    def test_aspect_ratio_high_train(self):
        # 2.5 * 7/6, matching the paper's training-time post-squeeze AR.
        assert ASPECT_RATIO_HIGH_TRAIN == pytest.approx(2.5 * 7 / 6)

    def test_aspect_ratio_low_eval(self):
        assert ASPECT_RATIO_LOW_EVAL == pytest.approx(1.5)

    def test_aspect_ratio_high_eval(self):
        assert ASPECT_RATIO_HIGH_EVAL == pytest.approx(3.5)

    def test_rotate_bounds(self):
        assert ROTATE_BOUNDS == (-3, 3)

    def test_shear_bounds(self):
        assert SHEAR_BOUNDS == (-3, 3)

    def test_blur_limit(self):
        assert BLUR_LIMIT == (0.5, 1.0)

    def test_noise_mean_range(self):
        assert NOISE_MEAN_RANGE == (0.0, 0.0)

    def test_noise_std_range(self):
        assert NOISE_STD_RANGE == (0.008, 0.016)

    def test_rot_flip_prob(self):
        assert ROT_FLIP_PROB == pytest.approx(0.0)

    def test_gradient_fg_range(self):
        assert GRADIENT_FG_RANGE == (140, 220)

    def test_gradient_bg_range(self):
        assert GRADIENT_BG_RANGE == (20, 100)

    def test_gradient_a_range(self):
        assert GRADIENT_A_RANGE == (0.4, 0.6)


# add_grayscale_gradient


class TestAddGrayscaleGradient:
    """Tests for the standalone gradient overlay function."""

    def test_output_shape_preserved(self, wide_image):
        result = add_grayscale_gradient(wide_image)
        assert result.shape == wide_image.shape

    def test_output_dtype_preserved_uint8(self, wide_image):
        result = add_grayscale_gradient(wide_image)
        assert result.dtype == np.uint8

    def test_output_dtype_preserved_float32(self):
        img = (np.random.default_rng(0).random((100, 200)) * 255).astype(np.float32)
        result = add_grayscale_gradient(img)
        assert result.dtype == np.float32

    def test_output_values_clipped_to_valid_range(self, wide_image):
        result = add_grayscale_gradient(wide_image)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_gradient_changes_pixel_values(self):
        # A uniformly bright image is guaranteed to change after gradient subtraction.
        bright = np.full((100, 200), 200, dtype=np.uint8)
        result = add_grayscale_gradient(bright)
        assert not np.array_equal(result, bright)

    def test_custom_a_range_produces_valid_output(self, wide_image):
        result = add_grayscale_gradient(wide_image, a_range=(0.1, 0.2))
        assert result.shape == wide_image.shape
        assert result.dtype == wide_image.dtype
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_square_image(self):
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, size=(150, 150), dtype=np.uint8)
        result = add_grayscale_gradient(img)
        assert result.shape == img.shape


# TargetAspectRatioResize


class TestTargetAspectRatioResize:
    """Tests for the aspect-ratio-driven resize DualTransform."""

    def test_is_subclass_of_dual_transform(self):
        transform = TargetAspectRatioResize(
            height=IMAGE_SIZE,
            aspect_ratio_low=ASPECT_RATIO_LOW_TRAIN,
            aspect_ratio_high=ASPECT_RATIO_HIGH_TRAIN,
        )
        assert isinstance(transform, DualTransform)

    def test_targets_attribute_is_image(self):
        assert TargetAspectRatioResize._targets == Targets.IMAGE

    def test_get_transform_init_args_names(self):
        transform = TargetAspectRatioResize(
            height=IMAGE_SIZE,
            aspect_ratio_low=ASPECT_RATIO_LOW_TRAIN,
            aspect_ratio_high=ASPECT_RATIO_HIGH_TRAIN,
        )
        assert transform.get_transform_init_args_names() == (
            "height",
            "aspect_ratio_low",
            "aspect_ratio_high",
            "interpolation",
        )

    def test_rejects_aspect_ratio_low_below_one(self):
        with pytest.raises(ValueError, match="aspect_ratio_low"):
            TargetAspectRatioResize(
                height=IMAGE_SIZE, aspect_ratio_low=0.5, aspect_ratio_high=2.0
            )

    def test_rejects_low_greater_than_high(self):
        with pytest.raises(ValueError, match="aspect_ratio_low"):
            TargetAspectRatioResize(
                height=IMAGE_SIZE, aspect_ratio_low=3.0, aspect_ratio_high=2.0
            )

    def test_output_height_equals_target(self, wide_image):
        transform = TargetAspectRatioResize(
            height=IMAGE_SIZE,
            aspect_ratio_low=ASPECT_RATIO_LOW_EVAL,
            aspect_ratio_high=ASPECT_RATIO_HIGH_EVAL,
            p=1.0,
        )
        result = transform(image=wide_image)["image"]
        assert result.shape[0] == IMAGE_SIZE

    @pytest.mark.parametrize(
        "shape",
        [(80, 400), (50, 500), (200, 200), (400, 80), (105, 105), (30, 800)],
    )
    def test_output_aspect_ratio_independent_of_input(self, shape):
        # Regardless of the input aspect ratio the post-resize AR must land
        # inside the configured [low, high] band - that is the whole point of
        # this transform.
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=shape, dtype=np.uint8)
        transform = TargetAspectRatioResize(
            height=IMAGE_SIZE,
            aspect_ratio_low=ASPECT_RATIO_LOW_EVAL,
            aspect_ratio_high=ASPECT_RATIO_HIGH_EVAL,
            p=1.0,
        )
        for _ in range(20):
            out = transform(image=img)["image"]
            ar = out.shape[1] / out.shape[0]
            # Allow 1 pixel of rounding slack at each end.
            assert ar >= ASPECT_RATIO_LOW_EVAL - 1.0 / IMAGE_SIZE
            assert ar <= ASPECT_RATIO_HIGH_EVAL + 1.0 / IMAGE_SIZE

    def test_aspect_ratio_distribution_is_uniform(self):
        # Drawing many crops from a square input, the realized aspect ratios
        # should cover most of the configured band rather than collapsing to
        # one value (regression guard against a silent fixed-ratio resize).
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(200, 200), dtype=np.uint8)
        transform = TargetAspectRatioResize(
            height=IMAGE_SIZE,
            aspect_ratio_low=ASPECT_RATIO_LOW_EVAL,
            aspect_ratio_high=ASPECT_RATIO_HIGH_EVAL,
            p=1.0,
        )
        ars = []
        for _ in range(200):
            out = transform(image=img)["image"]
            ars.append(out.shape[1] / out.shape[0])
        ars = np.array(ars)
        band = ASPECT_RATIO_HIGH_EVAL - ASPECT_RATIO_LOW_EVAL
        # At least 70% of the configured band should be observed.
        observed_band = ars.max() - ars.min()
        assert observed_band >= 0.7 * band

    def test_works_inside_compose(self, wide_image):
        pipeline = A.Compose(
            [
                TargetAspectRatioResize(
                    height=IMAGE_SIZE,
                    aspect_ratio_low=ASPECT_RATIO_LOW_TRAIN,
                    aspect_ratio_high=ASPECT_RATIO_HIGH_TRAIN,
                    p=1.0,
                )
            ]
        )
        result = pipeline(image=wide_image)["image"]
        assert result.shape[0] == IMAGE_SIZE
        ar = result.shape[1] / result.shape[0]
        assert ar >= ASPECT_RATIO_LOW_TRAIN - 1.0 / IMAGE_SIZE
        assert ar <= ASPECT_RATIO_HIGH_TRAIN + 1.0 / IMAGE_SIZE


# augmentation_pipeline dispatcher


class TestAugmentationPipelineDispatch:
    """Tests for the public dispatcher that routes to the correct subpipeline."""

    def test_raises_value_error_for_unknown_image_type(self, wide_image):
        with pytest.raises(ValueError, match="synthetic.*real"):
            augmentation_pipeline(wide_image, "unknown")

    def test_raises_value_error_for_empty_string_type(self, wide_image):
        with pytest.raises(ValueError):
            augmentation_pipeline(wide_image, "")

    def test_synthetic_type_returns_correct_shape(self, wide_image):
        result = augmentation_pipeline(wide_image, "synthetic")
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_real_type_returns_correct_shape(self, wide_image):
        result = augmentation_pipeline(wide_image, "real")
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)


# Synthetic pipeline


class TestSyntheticPipeline:
    """End to end tests for the synthetic image augmentation pipeline."""

    def test_output_shape(self, wide_image):
        result = augmentation_pipeline(wide_image, "synthetic")
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_is_uint8(self, wide_image):
        result = augmentation_pipeline(wide_image, "synthetic")
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self, wide_image):
        result = augmentation_pipeline(wide_image, "synthetic")
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    @pytest.mark.parametrize(
        "shape",
        [(50, 500), (200, 200), (400, 80), (105, 105), (30, 800)],
    )
    def test_works_with_various_input_shapes(self, shape):
        img = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
        result = augmentation_pipeline(img, "synthetic")
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE), f"Failed for input shape {shape}"


# Real pipeline


class TestRealPipeline:
    """End to end tests for the real image augmentation pipeline."""

    def test_output_shape(self, wide_image):
        result = augmentation_pipeline(wide_image, "real")
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_is_uint8(self, wide_image):
        result = augmentation_pipeline(wide_image, "real")
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self, wide_image):
        result = augmentation_pipeline(wide_image, "real")
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    @pytest.mark.parametrize(
        "shape",
        [(50, 500), (200, 200), (400, 80), (105, 105), (30, 800)],
    )
    def test_works_with_various_input_shapes(self, shape):
        img = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
        result = augmentation_pipeline(img, "real")
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE), f"Failed for input shape {shape}"


# Eval pipeline


class TestEvalPipeline:
    """End to end tests for the test time augmentation (TTA) eval pipeline."""

    @pytest.mark.parametrize("num_crops", [1, 5, 10])
    def test_output_shape(self, wide_image, num_crops):
        result = eval_pipeline(wide_image, num_crops)
        assert result.shape == (num_crops, IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_is_uint8(self, wide_image):
        result = eval_pipeline(wide_image, 3)
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self, wide_image):
        result = eval_pipeline(wide_image, 3)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_crops_are_stochastic(self, wide_image):
        # 10 crops from a non-trivial image should not all be identical.
        result = eval_pipeline(wide_image, 10)
        all_same = all(np.array_equal(result[0], result[i]) for i in range(1, 10))
        assert not all_same

    def test_returns_numpy_array(self, wide_image):
        result = eval_pipeline(wide_image, 3)
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize("shape", [(50, 500), (200, 200), (400, 80)])
    def test_works_with_various_input_shapes(self, shape):
        img = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
        result = eval_pipeline(img, 3)
        assert result.shape == (3, IMAGE_SIZE, IMAGE_SIZE), f"Failed for input shape {shape}"


# Albumentations API contract


class TestAlbumentationsAPIContract:
    """Explicitly test the albumentations parameter names and import paths used.

    These tests document the exact API surface consumed from albumentations so that
    a version upgrade that renames or removes a parameter or moves a class to a
    different module produces a descriptive failure pointing to the exact breakage,
    rather than an obscure error buried inside a pipeline call.

    Each test corresponds to a change observed between albumentations 1.x and 2.x.
    """

    # Import paths

    def test_dual_transform_importable_from_core(self):
        """albumentations.core.transforms_interface.DualTransform must exist."""
        assert DualTransform is not None

    def test_resize_importable_from_geometric_functional(self):
        """albumentations.augmentations.geometric.functional.resize must exist."""
        assert resize is not None

    def test_targets_importable_from_core_type_definitions(self):
        """albumentations.core.type_definitions.Targets must exist."""
        assert Targets is not None

    # Transform availability

    def test_invert_img_available(self):
        """A.InvertImg must exist and produce a same-shape result."""
        t = A.InvertImg(p=1.0)
        img = np.full((105, 105), 100, dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.shape == (105, 105)

    def test_invert_img_correctness(self):
        """A.InvertImg should compute 255 - pixel for uint8 images."""
        t = A.InvertImg(p=1.0)
        img = np.full((105, 105), 100, dtype=np.uint8)
        result = t(image=img)["image"]
        assert np.all(result == 155)  # 255 - 100 = 155

    def test_random_brightness_contrast_available(self):
        """A.RandomBrightnessContrast must exist and produce a same-shape result."""
        t = A.RandomBrightnessContrast(p=1.0)
        img = np.full((105, 105), 128, dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.shape == (105, 105)

    def test_random_rotate90_available(self):
        """A.RandomRotate90 must exist and produce a 105x105 result."""
        t = A.RandomRotate90(p=1.0)
        img = np.zeros((105, 105), dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.shape == (105, 105)

    def test_horizontal_flip_available(self):
        """A.HorizontalFlip must exist and preserve image shape."""
        t = A.HorizontalFlip(p=1.0)
        img = np.zeros((105, 105), dtype=np.uint8)
        assert t(image=img)["image"].shape == (105, 105)

    def test_vertical_flip_available(self):
        """A.VerticalFlip must exist and preserve image shape."""
        t = A.VerticalFlip(p=1.0)
        img = np.zeros((105, 105), dtype=np.uint8)
        assert t(image=img)["image"].shape == (105, 105)

    def test_compose_callable_and_returns_image_key(self):
        """A.Compose must return a dict with an 'image' key."""
        pipeline = A.Compose([A.HorizontalFlip(p=0.5)])
        result = pipeline(image=np.zeros((105, 105), dtype=np.uint8))
        assert "image" in result

    # Parameter names that changed between albumentations 1.x and 2.x

    def test_gaussnoise_accepts_std_range_parameter(self):
        """albumentations>=2.0 renamed var_limit to std_range for GaussNoise."""
        t = A.GaussNoise(std_range=NOISE_STD_RANGE, p=1.0)
        img = np.full((105, 105), 128, dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.shape == (105, 105)

    def test_gaussnoise_accepts_mean_range_parameter(self):
        """albumentations>=2.0 added mean_range to GaussNoise."""
        t = A.GaussNoise(std_range=NOISE_STD_RANGE, mean_range=NOISE_MEAN_RANGE, p=1.0)
        img = np.full((105, 105), 128, dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.shape == (105, 105)

    def test_gaussianblur_accepts_sigma_limit_parameter(self):
        """albumentations>=2.0 uses sigma_limit to specify the blur sigma range."""
        t = A.GaussianBlur(blur_limit=0, sigma_limit=BLUR_LIMIT, p=1.0)
        img = np.full((105, 105), 128, dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.shape == (105, 105)

    def test_affine_accepts_border_mode_parameter(self):
        """albumentations>=2.0 uses border_mode (OpenCV flag) in A.Affine."""
        t = A.Affine(
            rotate=ROTATE_BOUNDS,
            shear=SHEAR_BOUNDS,
            border_mode=cv2.BORDER_REFLECT,
            p=1.0,
        )
        img = np.zeros((200, 200), dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.shape == (200, 200)

    def test_random_crop_output_shape(self):
        """A.RandomCrop must produce the exact requested (height, width) size."""
        t = A.RandomCrop(IMAGE_SIZE, IMAGE_SIZE, p=1.0)
        img = np.zeros((200, 300), dtype=np.uint8)
        result = t(image=img)["image"]
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)
