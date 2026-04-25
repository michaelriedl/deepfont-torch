"""Tests for the persistent augmentation pipeline classes.

Verifies that SyntheticAugmentationPipeline, RealAugmentationPipeline, and
EvalAugmentationPipeline produce correct outputs, that assigning a new config
to the pipeline rebuilds the internal Compose correctly, and that deep-copying
a dataset produces independent pipeline objects (so val_set.disable_augmentation()
does not affect the train set).

Test classes:
    TestSyntheticAugmentationPipeline  -- output shape/dtype, config property
    TestRealAugmentationPipeline       -- output shape/dtype, config property
    TestEvalAugmentationPipeline       -- output shape, stochasticity
    TestDatasetDisableAugmentation     -- disable_augmentation behavior
"""

import copy
import struct
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from deepfont.data.config import FinetuneDataConfig, PretrainDataConfig
from deepfont.data.datasets import FinetuneData, PretrainData
from deepfont.data.augmentations import (
    IMAGE_SIZE,
    RealAugmentationConfig,
    EvalAugmentationPipeline,
    RealAugmentationPipeline,
    SyntheticAugmentationConfig,
    SyntheticAugmentationPipeline,
)

# Shared fixtures


@pytest.fixture
def wide_image() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(80, 400), dtype=np.uint8)


def _make_png_bytes(width: int, height: int, seed: int) -> bytes:
    rng = np.random.RandomState(seed)
    pixels = rng.randint(0, 256, (height, width), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_bcf(path: str, png_list: list[bytes]) -> None:
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(png_list)))
        for png in png_list:
            f.write(struct.pack("<Q", len(png)))
        for png in png_list:
            f.write(png)


def _write_labels(path: str, labels: list[int]) -> None:
    with open(path, "wb") as f:
        for label in labels:
            f.write(struct.pack("<I", label))


@pytest.fixture
def pretrain_dataset(tmp_path):
    pngs = [_make_png_bytes(width=50 + i * 10, height=110, seed=i) for i in range(20)]
    bcf_path = str(tmp_path / "train.bcf")
    _write_bcf(bcf_path, pngs)
    config = PretrainDataConfig(
        synthetic_bcf_file=bcf_path,
        real_image_dir=None,
        image_normalization="0to1",
    )
    return PretrainData(config)


@pytest.fixture
def finetune_dataset(tmp_path):
    pngs = [_make_png_bytes(width=50 + i * 10, height=110, seed=i + 100) for i in range(20)]
    bcf_path = str(tmp_path / "train.bcf")
    label_path = str(tmp_path / "train.label")
    _write_bcf(bcf_path, pngs)
    _write_labels(label_path, [i % 5 for i in range(20)])
    config = FinetuneDataConfig(
        synthetic_bcf_file=bcf_path,
        label_file=label_path,
        image_normalization="0to1",
    )
    return FinetuneData(config)


# SyntheticAugmentationPipeline


class TestSyntheticAugmentationPipeline:
    def test_output_shape(self, wide_image):
        pipeline = SyntheticAugmentationPipeline()
        result = pipeline(wide_image)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_is_uint8(self, wide_image):
        pipeline = SyntheticAugmentationPipeline()
        assert pipeline(wide_image).dtype == np.uint8

    def test_output_values_in_valid_range(self, wide_image):
        pipeline = SyntheticAugmentationPipeline()
        result = pipeline(wide_image)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_default_config_uses_default_values(self):
        pipeline = SyntheticAugmentationPipeline()
        assert pipeline.config == SyntheticAugmentationConfig()

    def test_config_setter_updates_getter(self):
        pipeline = SyntheticAugmentationPipeline()
        new_config = pipeline.config.with_stochastic_disabled()
        pipeline.config = new_config
        assert pipeline.config == new_config

    def test_config_setter_rebuilds_compose(self):
        pipeline = SyntheticAugmentationPipeline()
        original_compose = pipeline._compose
        pipeline.config = pipeline.config.with_stochastic_disabled()
        assert pipeline._compose is not original_compose

    def test_disabled_config_disables_gradient(self, wide_image):
        """With every *_prob=0, add_grayscale_gradient must never be called."""
        pipeline = SyntheticAugmentationPipeline(
            SyntheticAugmentationConfig().with_stochastic_disabled()
        )
        with patch("deepfont.data.augmentations.add_grayscale_gradient") as mock_gradient:
            for _ in range(20):
                pipeline(wide_image)
        mock_gradient.assert_not_called()

    def test_deepcopy_produces_independent_pipeline(self, wide_image):
        original = SyntheticAugmentationPipeline()
        cloned = copy.deepcopy(original)
        cloned.config = cloned.config.with_stochastic_disabled()
        assert original.config.gradient_prob == pytest.approx(1.0)
        assert cloned.config.gradient_prob == pytest.approx(0.0)


# RealAugmentationPipeline


class TestRealAugmentationPipeline:
    def test_output_shape(self, wide_image):
        pipeline = RealAugmentationPipeline()
        assert pipeline(wide_image).shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_is_uint8(self, wide_image):
        assert RealAugmentationPipeline()(wide_image).dtype == np.uint8

    def test_output_values_in_valid_range(self, wide_image):
        result = RealAugmentationPipeline()(wide_image)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_default_config_uses_default_values(self):
        pipeline = RealAugmentationPipeline()
        assert pipeline.config == RealAugmentationConfig()

    def test_config_setter_updates_getter(self):
        pipeline = RealAugmentationPipeline()
        new_config = pipeline.config.with_stochastic_disabled()
        pipeline.config = new_config
        assert pipeline.config == new_config

    def test_config_setter_rebuilds_compose(self):
        pipeline = RealAugmentationPipeline()
        original_compose = pipeline._compose
        pipeline.config = pipeline.config.with_stochastic_disabled()
        assert pipeline._compose is not original_compose

    def test_deepcopy_produces_independent_pipeline(self):
        original = RealAugmentationPipeline()
        cloned = copy.deepcopy(original)
        cloned.config = cloned.config.with_stochastic_disabled()
        assert original.config.affine_prob == pytest.approx(1.0)
        assert cloned.config.affine_prob == pytest.approx(0.0)


# EvalAugmentationPipeline


class TestEvalAugmentationPipeline:
    @pytest.mark.parametrize("num_crops", [1, 5, 10])
    def test_output_shape(self, wide_image, num_crops):
        pipeline = EvalAugmentationPipeline()
        result = pipeline(wide_image, num_crops)
        assert result.shape == (num_crops, IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_is_uint8(self, wide_image):
        assert EvalAugmentationPipeline()(wide_image, 3).dtype == np.uint8

    def test_output_values_in_valid_range(self, wide_image):
        result = EvalAugmentationPipeline()(wide_image, 3)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_crops_are_stochastic(self, wide_image):
        result = EvalAugmentationPipeline()(wide_image, 10)
        all_same = all(np.array_equal(result[0], result[i]) for i in range(1, 10))
        assert not all_same

    def test_returns_numpy_array(self, wide_image):
        assert isinstance(EvalAugmentationPipeline()(wide_image, 3), np.ndarray)

    def test_reuse_across_calls_produces_correct_shape(self, wide_image):
        pipeline = EvalAugmentationPipeline()
        for _ in range(5):
            result = pipeline(wide_image, 4)
            assert result.shape == (4, IMAGE_SIZE, IMAGE_SIZE)


# Dataset disable_augmentation


class TestDatasetDisableAugmentation:
    def test_pretrain_disable_zeros_pipeline_probs(self, pretrain_dataset):
        pretrain_dataset.disable_augmentation()
        syn_cfg = pretrain_dataset._synthetic_pipeline.config
        real_cfg = pretrain_dataset._real_pipeline.config
        assert syn_cfg.affine_prob == pytest.approx(0.0)
        assert syn_cfg.blur_prob == pytest.approx(0.0)
        assert syn_cfg.brightness_contrast_prob == pytest.approx(0.0)
        assert syn_cfg.noise_prob == pytest.approx(0.0)
        assert syn_cfg.gradient_prob == pytest.approx(0.0)
        assert syn_cfg.invert_prob == pytest.approx(0.0)
        assert syn_cfg.rot_flip_prob == pytest.approx(0.0)
        assert real_cfg.affine_prob == pytest.approx(0.0)
        assert real_cfg.brightness_contrast_prob == pytest.approx(0.0)
        assert real_cfg.invert_prob == pytest.approx(0.0)
        assert real_cfg.rot_flip_prob == pytest.approx(0.0)

    def test_finetune_disable_zeros_pipeline_probs(self, finetune_dataset):
        finetune_dataset.disable_augmentation()
        syn_cfg = finetune_dataset._synthetic_pipeline.config
        assert syn_cfg.affine_prob == pytest.approx(0.0)
        assert syn_cfg.gradient_prob == pytest.approx(0.0)
        assert syn_cfg.invert_prob == pytest.approx(0.0)
        assert syn_cfg.rot_flip_prob == pytest.approx(0.0)

    def test_val_set_disable_does_not_affect_train_set(self, finetune_dataset):
        """Mirrors the finetune.py pattern: val_set.disable_augmentation()."""
        train_set, val_set = finetune_dataset.split_data_random(train_ratio=0.8)
        original_train_gradient = train_set._synthetic_pipeline.config.gradient_prob

        val_set.disable_augmentation()

        assert val_set._synthetic_pipeline.config.gradient_prob == pytest.approx(0.0)
        assert train_set._synthetic_pipeline.config.gradient_prob == pytest.approx(
            original_train_gradient
        )

    def test_pretrain_val_set_disable_independent(self, pretrain_dataset):
        train_set, val_set = pretrain_dataset.split_data_random(train_ratio=0.8)
        val_set.disable_augmentation()

        assert val_set._synthetic_pipeline.config.gradient_prob == pytest.approx(0.0)
        assert val_set._real_pipeline.config.affine_prob == pytest.approx(0.0)
        assert train_set._synthetic_pipeline.config.gradient_prob == pytest.approx(1.0)
        assert train_set._real_pipeline.config.affine_prob == pytest.approx(1.0)

    def test_deepcopy_of_dataset_produces_independent_pipelines(self, finetune_dataset):
        cloned = copy.deepcopy(finetune_dataset)
        cloned.disable_augmentation()
        assert finetune_dataset._synthetic_pipeline.config.gradient_prob == pytest.approx(1.0)
        assert cloned._synthetic_pipeline.config.gradient_prob == pytest.approx(0.0)
