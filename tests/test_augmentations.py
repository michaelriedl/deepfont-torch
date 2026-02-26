"""Tests for the augmentation pipelines in deepfont.data.augmentations.

These tests verify that all augmentation pipelines execute correctly and produce
outputs with the expected shapes, dtypes, and value ranges. They are deliberately
explicit about the albumentations API surface being used so that a version upgrade
that renames or removes a parameter produces a clear, descriptive failure rather than
a silent regression.

Test classes:
    TestConstants                 -- module-level hyper-parameter values
    TestAddGreyscaleGradient      -- standalone NumPy gradient function
    TestRandomWidthScale          -- custom albumentations width-only scaling transform
    TestResizeHeightSqueezeWidth  -- custom DualTransform for height + width resize
    TestAugmentationPipelineDispatch -- dispatcher routing and error handling
    TestSyntheticPipeline         -- end-to-end synthetic image pipeline
    TestRealPipeline              -- end-to-end real image pipeline
    TestEvalPipeline              -- test-time augmentation (TTA) eval pipeline
    TestAlbumentationsAPIContract -- explicit tests for albumentations parameter names
                                     and import paths that have historically changed
"""

import inspect

import cv2
import numpy as np
import pytest
import albumentations as A  # noqa: N812
from albumentations import RandomScale
from albumentations.core.type_definitions import Targets
from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.geometric.functional import resize

from deepfont.data.augmentations import (
    BLUR_LIMIT,
    IMAGE_SIZE,
    SCALE_LIMIT,
    SHEAR_BOUNDS,
    ROT_FLIP_PROB,
    ROTATE_BOUNDS,
    SQUEEZE_RATIO,
    NOISE_STD_RANGE,
    EVAL_SCALE_LIMIT,
    NOISE_MEAN_RANGE,
    RandomWidthScale,
    ResizeHeightSqueezeWidth,
    eval_pipeline,
    augmentation_pipeline,
    add_greyscale_gradient,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def wide_image() -> np.ndarray:
    """A wide greyscale text patch (landscape orientation)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(80, 400), dtype=np.uint8)


@pytest.fixture
def tall_image() -> np.ndarray:
    """A tall, narrow image that triggers the width-clamping code paths."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(400, 80), dtype=np.uint8)


@pytest.fixture
def square_image() -> np.ndarray:
    """A square greyscale image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(200, 200), dtype=np.uint8)


# ── Module-level constants ─────────────────────────────────────────────────────


class TestConstants:
    """Verify that all module-level hyper-parameters have the expected values.

    Pinning these values means any accidental edit to the augmentations module
    immediately causes a test failure, giving the same protection as locking a
    config file.
    """

    def test_image_size(self):
        assert IMAGE_SIZE == 105

    def test_squeeze_ratio(self):
        assert SQUEEZE_RATIO == pytest.approx(1 / 2.5)

    def test_scale_limit(self):
        assert SCALE_LIMIT == pytest.approx(0.15)

    def test_eval_scale_limit(self):
        assert EVAL_SCALE_LIMIT == pytest.approx(0.4)

    def test_rotate_bounds(self):
        assert ROTATE_BOUNDS == (-45, 45)

    def test_shear_bounds(self):
        assert SHEAR_BOUNDS == (-15, 15)

    def test_blur_limit(self):
        assert BLUR_LIMIT == (2.5, 3.5)

    def test_noise_mean_range(self):
        assert NOISE_MEAN_RANGE == (0.0, 0.0)

    def test_noise_std_range(self):
        assert NOISE_STD_RANGE == (0.05, 0.15)

    def test_rot_flip_prob(self):
        assert ROT_FLIP_PROB == pytest.approx(0.5)


# ── add_greyscale_gradient ─────────────────────────────────────────────────────


class TestAddGreyscaleGradient:
    """Tests for the standalone gradient-overlay function."""

    def test_output_shape_preserved(self, wide_image):
        result = add_greyscale_gradient(wide_image)
        assert result.shape == wide_image.shape

    def test_output_dtype_preserved_uint8(self, wide_image):
        result = add_greyscale_gradient(wide_image)
        assert result.dtype == np.uint8

    def test_output_dtype_preserved_float32(self):
        img = (np.random.default_rng(0).random((100, 200)) * 255).astype(np.float32)
        result = add_greyscale_gradient(img)
        assert result.dtype == np.float32

    def test_output_values_clipped_to_valid_range(self, wide_image):
        result = add_greyscale_gradient(wide_image)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_gradient_changes_pixel_values(self):
        # A uniformly bright image is guaranteed to change after gradient subtraction.
        bright = np.full((100, 200), 200, dtype=np.uint8)
        result = add_greyscale_gradient(bright)
        assert not np.array_equal(result, bright)

    def test_custom_gradient_bounds_produces_valid_output(self, wide_image):
        result = add_greyscale_gradient(wide_image, gradient_min=(0, 10), gradient_max=(10, 20))
        assert result.shape == wide_image.shape
        assert result.dtype == wide_image.dtype
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_square_image(self):
        rng = np.random.default_rng(7)
        img = rng.integers(0, 256, size=(150, 150), dtype=np.uint8)
        result = add_greyscale_gradient(img)
        assert result.shape == img.shape


# ── RandomWidthScale ───────────────────────────────────────────────────────────


class TestRandomWidthScale:
    """Tests for the width-only scaling transform."""

    def test_is_subclass_of_random_scale(self):
        transform = RandomWidthScale(scale_limit=SCALE_LIMIT, p=1.0)
        assert isinstance(transform, RandomScale)

    def test_height_is_always_preserved(self, wide_image):
        transform = RandomWidthScale(scale_limit=SCALE_LIMIT, p=1.0)
        result = transform.apply(wide_image, scale=1.0, interpolation=cv2.INTER_LINEAR)
        assert result.shape[0] == wide_image.shape[0]

    def test_height_preserved_with_upscaling(self, wide_image):
        transform = RandomWidthScale(scale_limit=SCALE_LIMIT, p=1.0)
        result = transform.apply(wide_image, scale=1.15, interpolation=cv2.INTER_LINEAR)
        assert result.shape[0] == wide_image.shape[0]

    def test_height_preserved_with_downscaling(self, wide_image):
        transform = RandomWidthScale(scale_limit=SCALE_LIMIT, p=1.0)
        result = transform.apply(wide_image, scale=0.85, interpolation=cv2.INTER_LINEAR)
        assert result.shape[0] == wide_image.shape[0]

    def test_width_scales_proportionally(self, wide_image):
        transform = RandomWidthScale(scale_limit=SCALE_LIMIT, p=1.0)
        result = transform.apply(wide_image, scale=1.1, interpolation=cv2.INTER_LINEAR)
        expected_width = int(wide_image.shape[1] * 1.1)
        assert result.shape[1] == expected_width

    def test_width_never_below_height(self):
        # Height=80, Width=50 -- downscaling would produce width < height.
        narrow = np.zeros((80, 50), dtype=np.uint8)
        transform = RandomWidthScale(scale_limit=0.5, p=1.0)
        result = transform.apply(narrow, scale=0.5, interpolation=cv2.INTER_LINEAR)
        assert result.shape[1] >= result.shape[0]

    def test_works_inside_compose(self, wide_image):
        pipeline = A.Compose([RandomWidthScale(scale_limit=SCALE_LIMIT, p=1.0)])
        result = pipeline(image=wide_image)["image"]
        assert result.shape[0] == wide_image.shape[0]
        assert result.shape[1] >= result.shape[0]


# ── ResizeHeightSqueezeWidth ───────────────────────────────────────────────────


class TestResizeHeightSqueezeWidth:
    """Tests for the combined height-resize / width-squeeze DualTransform."""

    def test_is_subclass_of_dual_transform(self):
        transform = ResizeHeightSqueezeWidth(height=IMAGE_SIZE, width_scale=SQUEEZE_RATIO)
        assert isinstance(transform, DualTransform)

    def test_output_height_equals_target(self, wide_image):
        transform = ResizeHeightSqueezeWidth(height=IMAGE_SIZE, width_scale=SQUEEZE_RATIO, p=1.0)
        result = transform.apply(wide_image, interpolation=cv2.INTER_LINEAR)
        assert result.shape[0] == IMAGE_SIZE

    def test_width_never_below_target_height(self):
        # (200, 100): squeezed width = int((105/200)*0.4*100) = 21 < 105, clamped.
        narrow = np.zeros((200, 100), dtype=np.uint8)
        transform = ResizeHeightSqueezeWidth(height=IMAGE_SIZE, width_scale=SQUEEZE_RATIO, p=1.0)
        result = transform.apply(narrow, interpolation=cv2.INTER_LINEAR)
        assert result.shape[1] >= IMAGE_SIZE

    def test_width_calculation_wide_image(self):
        # (50, 600): height_scale=105/50=2.1, new_width=int(2.1*0.4*600)=504.
        img = np.zeros((50, 600), dtype=np.uint8)
        transform = ResizeHeightSqueezeWidth(height=105, width_scale=0.4, p=1.0)
        result = transform.apply(img, interpolation=cv2.INTER_LINEAR)
        expected_width = max(int((105 / 50) * 0.4 * 600), 105)
        assert result.shape == (105, expected_width)

    def test_get_transform_init_args_names(self):
        transform = ResizeHeightSqueezeWidth(height=IMAGE_SIZE, width_scale=SQUEEZE_RATIO)
        assert transform.get_transform_init_args_names() == (
            "height",
            "width_scale",
            "interpolation",
        )

    def test_targets_attribute_is_image(self):
        assert ResizeHeightSqueezeWidth._targets == Targets.IMAGE

    def test_works_inside_compose(self, wide_image):
        pipeline = A.Compose([ResizeHeightSqueezeWidth(IMAGE_SIZE, SQUEEZE_RATIO, p=1.0)])
        result = pipeline(image=wide_image)["image"]
        assert result.shape[0] == IMAGE_SIZE
        assert result.shape[1] >= IMAGE_SIZE


# ── augmentation_pipeline (dispatcher) ────────────────────────────────────────


class TestAugmentationPipelineDispatch:
    """Tests for the public dispatcher that routes to the correct sub-pipeline."""

    def test_raises_value_error_for_unknown_image_type(self, wide_image):
        with pytest.raises(ValueError, match="synthetic.*real"):
            augmentation_pipeline(wide_image, "unknown", aug_prob=1.0)

    def test_raises_value_error_for_empty_string_type(self, wide_image):
        with pytest.raises(ValueError):
            augmentation_pipeline(wide_image, "", aug_prob=1.0)

    def test_synthetic_type_returns_correct_shape(self, wide_image):
        result = augmentation_pipeline(wide_image, "synthetic", aug_prob=1.0)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_real_type_returns_correct_shape(self, wide_image):
        result = augmentation_pipeline(wide_image, "real", aug_prob=1.0)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)


# ── Synthetic pipeline ─────────────────────────────────────────────────────────


class TestSyntheticPipeline:
    """End-to-end tests for the synthetic image augmentation pipeline."""

    @pytest.mark.parametrize("aug_prob", [0.0, 0.5, 1.0])
    def test_output_shape(self, wide_image, aug_prob):
        result = augmentation_pipeline(wide_image, "synthetic", aug_prob=aug_prob)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_is_uint8(self, wide_image):
        result = augmentation_pipeline(wide_image, "synthetic", aug_prob=1.0)
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self, wide_image):
        result = augmentation_pipeline(wide_image, "synthetic", aug_prob=1.0)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    @pytest.mark.parametrize(
        "shape",
        [(50, 500), (200, 200), (400, 80), (105, 105), (30, 800)],
    )
    def test_works_with_various_input_shapes(self, shape):
        img = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
        result = augmentation_pipeline(img, "synthetic", aug_prob=0.0)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE), f"Failed for input shape {shape}"

    def test_zero_aug_prob_still_produces_correct_shape(self, wide_image):
        # aug_prob=0 disables stochastic transforms; only p=1.0 steps run.
        result = augmentation_pipeline(wide_image, "synthetic", aug_prob=0.0)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)


# ── Real pipeline ──────────────────────────────────────────────────────────────


class TestRealPipeline:
    """End-to-end tests for the real image augmentation pipeline."""

    @pytest.mark.parametrize("aug_prob", [0.0, 0.5, 1.0])
    def test_output_shape(self, wide_image, aug_prob):
        result = augmentation_pipeline(wide_image, "real", aug_prob=aug_prob)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_is_uint8(self, wide_image):
        result = augmentation_pipeline(wide_image, "real", aug_prob=1.0)
        assert result.dtype == np.uint8

    def test_output_values_in_valid_range(self, wide_image):
        result = augmentation_pipeline(wide_image, "real", aug_prob=1.0)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    @pytest.mark.parametrize(
        "shape",
        [(50, 500), (200, 200), (400, 80), (105, 105), (30, 800)],
    )
    def test_works_with_various_input_shapes(self, shape):
        img = np.random.default_rng(0).integers(0, 256, size=shape, dtype=np.uint8)
        result = augmentation_pipeline(img, "real", aug_prob=0.0)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE), f"Failed for input shape {shape}"

    def test_zero_aug_prob_still_produces_correct_shape(self, wide_image):
        result = augmentation_pipeline(wide_image, "real", aug_prob=0.0)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)


# ── Eval pipeline ──────────────────────────────────────────────────────────────


class TestEvalPipeline:
    """End-to-end tests for the test-time augmentation (TTA) eval pipeline."""

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
        # At ±40% width scaling, 10 crops of a natural image should not all be identical.
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


# ── Albumentations API contract ────────────────────────────────────────────────


class TestAlbumentationsAPIContract:
    """Explicitly test the albumentations parameter names and import paths used.

    These tests document the exact API surface consumed from albumentations so that
    a version upgrade that renames or removes a parameter or moves a class to a
    different module produces a descriptive failure pointing to the exact breakage,
    rather than an obscure error buried inside a pipeline call.

    Each test corresponds to a change observed between albumentations 1.x and 2.x.
    """

    # ── Import paths ──────────────────────────────────────────────────────────

    def test_dual_transform_importable_from_core(self):
        """albumentations.core.transforms_interface.DualTransform must exist."""
        assert DualTransform is not None

    def test_resize_importable_from_geometric_functional(self):
        """albumentations.augmentations.geometric.functional.resize must exist."""
        assert resize is not None

    def test_targets_importable_from_core_type_definitions(self):
        """albumentations.core.type_definitions.Targets must exist."""
        assert Targets is not None

    def test_random_scale_importable_from_albumentations(self):
        """albumentations.RandomScale must be directly importable."""
        assert RandomScale is not None

    # ── Transform availability ────────────────────────────────────────────────

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
        """A.RandomRotate90 must exist and produce a 105×105 result."""
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

    # ── Parameter names that changed between albumentations 1.x and 2.x ──────

    def test_gaussnoise_accepts_std_range_parameter(self):
        """albumentations>=2.0 renamed var_limit -> std_range for GaussNoise."""
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

    # ── RandomScale.apply() signature ────────────────────────────────────────

    def test_random_scale_apply_signature_has_scale_param(self):
        """RandomScale.apply() must accept a 'scale' parameter.

        RandomWidthScale overrides apply() and relies on receiving 'scale' as a
        keyword argument from the albumentations dispatch machinery. If the parent
        class renames this parameter the override will silently stop working.
        """
        sig = inspect.signature(RandomScale.apply)
        assert "scale" in sig.parameters

    def test_random_scale_apply_passes_interpolation_via_kwargs(self):
        """RandomScale.apply() passes interpolation through **params in albumentations>=2.x.

        The parent passes 'interpolation' as a keyword argument via **params, and our
        RandomWidthScale.apply() override captures it as an explicit named parameter.
        This test verifies both sides of that contract so a future signature change is
        caught immediately.
        """
        # Parent must accept **params so that interpolation can flow through.
        parent_sig = inspect.signature(RandomScale.apply)
        assert "params" in parent_sig.parameters

        # Our override must explicitly name 'interpolation' to receive it.
        override_sig = inspect.signature(RandomWidthScale.apply)
        assert "interpolation" in override_sig.parameters
