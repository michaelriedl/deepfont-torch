"""Tests for the persistent augmentation pipeline classes.

Verifies that SyntheticAugmentationPipeline, RealAugmentationPipeline, and
EvalAugmentationPipeline produce correct outputs, that their aug_prob property
setter rebuilds the internal Compose correctly, and that deep-copying a dataset
produces independent pipeline objects (so val_set.aug_prob = 0.0 does not affect
the train set).

Test classes:
    TestSyntheticAugmentationPipeline  -- output shape/dtype, aug_prob property
    TestRealAugmentationPipeline       -- output shape/dtype, aug_prob property
    TestEvalAugmentationPipeline       -- output shape, stochasticity
    TestDatasetAugProbProperty         -- property behaviour on PretrainData and FinetuneData
"""

import copy
import struct
from io import BytesIO
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from deepfont.data.augmentations import (
    IMAGE_SIZE,
    EvalAugmentationPipeline,
    RealAugmentationPipeline,
    SyntheticAugmentationPipeline,
)
from deepfont.data.config import FinetuneDataConfig, PretrainDataConfig
from deepfont.data.datasets import FinetuneData, PretrainData


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


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
        aug_prob=0.5,
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
        aug_prob=0.5,
        image_normalization="0to1",
    )
    return FinetuneData(config)


# ---------------------------------------------------------------------------
# SyntheticAugmentationPipeline
# ---------------------------------------------------------------------------


class TestSyntheticAugmentationPipeline:
    def test_output_shape(self, wide_image):
        pipeline = SyntheticAugmentationPipeline(aug_prob=0.5)
        result = pipeline(wide_image)
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_is_uint8(self, wide_image):
        pipeline = SyntheticAugmentationPipeline(aug_prob=1.0)
        assert pipeline(wide_image).dtype == np.uint8

    def test_output_values_in_valid_range(self, wide_image):
        pipeline = SyntheticAugmentationPipeline(aug_prob=1.0)
        result = pipeline(wide_image)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_aug_prob_getter_reflects_initial_value(self):
        pipeline = SyntheticAugmentationPipeline(aug_prob=0.7)
        assert pipeline.aug_prob == pytest.approx(0.7)

    def test_aug_prob_setter_updates_getter(self):
        pipeline = SyntheticAugmentationPipeline(aug_prob=0.5)
        pipeline.aug_prob = 0.0
        assert pipeline.aug_prob == pytest.approx(0.0)

    def test_aug_prob_setter_rebuilds_compose(self):
        pipeline = SyntheticAugmentationPipeline(aug_prob=0.5)
        original_compose = pipeline._compose
        pipeline.aug_prob = 0.0
        assert pipeline._compose is not original_compose

    def test_aug_prob_zero_disables_gradient(self, wide_image):
        """With aug_prob=0.0, add_grayscale_gradient must never be called."""
        pipeline = SyntheticAugmentationPipeline(aug_prob=0.0)
        with patch(
            "deepfont.data.augmentations.add_grayscale_gradient"
        ) as mock_gradient:
            for _ in range(20):
                pipeline(wide_image)
        mock_gradient.assert_not_called()

    def test_deepcopy_produces_independent_pipeline(self, wide_image):
        original = SyntheticAugmentationPipeline(aug_prob=0.5)
        cloned = copy.deepcopy(original)
        cloned.aug_prob = 0.0
        assert original.aug_prob == pytest.approx(0.5)
        assert cloned.aug_prob == pytest.approx(0.0)

    @pytest.mark.parametrize("aug_prob", [0.0, 0.5, 1.0])
    def test_various_aug_probs_produce_correct_shape(self, wide_image, aug_prob):
        pipeline = SyntheticAugmentationPipeline(aug_prob=aug_prob)
        assert pipeline(wide_image).shape == (IMAGE_SIZE, IMAGE_SIZE)


# ---------------------------------------------------------------------------
# RealAugmentationPipeline
# ---------------------------------------------------------------------------


class TestRealAugmentationPipeline:
    def test_output_shape(self, wide_image):
        pipeline = RealAugmentationPipeline(aug_prob=0.5)
        assert pipeline(wide_image).shape == (IMAGE_SIZE, IMAGE_SIZE)

    def test_output_dtype_is_uint8(self, wide_image):
        assert RealAugmentationPipeline(aug_prob=1.0)(wide_image).dtype == np.uint8

    def test_output_values_in_valid_range(self, wide_image):
        result = RealAugmentationPipeline(aug_prob=1.0)(wide_image)
        assert int(result.min()) >= 0
        assert int(result.max()) <= 255

    def test_aug_prob_getter_reflects_initial_value(self):
        assert RealAugmentationPipeline(aug_prob=0.3).aug_prob == pytest.approx(0.3)

    def test_aug_prob_setter_updates_getter(self):
        pipeline = RealAugmentationPipeline(aug_prob=0.5)
        pipeline.aug_prob = 0.0
        assert pipeline.aug_prob == pytest.approx(0.0)

    def test_aug_prob_setter_rebuilds_compose(self):
        pipeline = RealAugmentationPipeline(aug_prob=0.5)
        original_compose = pipeline._compose
        pipeline.aug_prob = 0.0
        assert pipeline._compose is not original_compose

    def test_deepcopy_produces_independent_pipeline(self):
        original = RealAugmentationPipeline(aug_prob=0.5)
        cloned = copy.deepcopy(original)
        cloned.aug_prob = 0.0
        assert original.aug_prob == pytest.approx(0.5)

    @pytest.mark.parametrize("aug_prob", [0.0, 0.5, 1.0])
    def test_various_aug_probs_produce_correct_shape(self, wide_image, aug_prob):
        assert RealAugmentationPipeline(aug_prob=aug_prob)(wide_image).shape == (
            IMAGE_SIZE,
            IMAGE_SIZE,
        )


# ---------------------------------------------------------------------------
# EvalAugmentationPipeline
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Dataset aug_prob property
# ---------------------------------------------------------------------------


class TestDatasetAugProbProperty:
    def test_pretrain_aug_prob_getter(self, pretrain_dataset):
        assert pretrain_dataset.aug_prob == pytest.approx(0.5)

    def test_pretrain_aug_prob_setter_updates_pipeline(self, pretrain_dataset):
        pretrain_dataset.aug_prob = 0.0
        assert pretrain_dataset._aug_prob == pytest.approx(0.0)
        assert pretrain_dataset._synthetic_pipeline.aug_prob == pytest.approx(0.0)
        assert pretrain_dataset._real_pipeline.aug_prob == pytest.approx(0.0)

    def test_finetune_aug_prob_getter(self, finetune_dataset):
        assert finetune_dataset.aug_prob == pytest.approx(0.5)

    def test_finetune_aug_prob_setter_updates_pipeline(self, finetune_dataset):
        finetune_dataset.aug_prob = 0.0
        assert finetune_dataset._aug_prob == pytest.approx(0.0)
        assert finetune_dataset._synthetic_pipeline.aug_prob == pytest.approx(0.0)

    def test_val_set_aug_prob_mutation_does_not_affect_train_set(self, finetune_dataset):
        """Mirrors the finetune.py pattern: val_set.aug_prob = 0.0."""
        train_set, val_set = finetune_dataset.split_data_random(train_ratio=0.8)
        original_train_aug_prob = train_set.aug_prob

        val_set.aug_prob = 0.0

        # val_set is updated
        assert val_set.aug_prob == pytest.approx(0.0)
        assert val_set._synthetic_pipeline.aug_prob == pytest.approx(0.0)
        # train_set is unaffected
        assert train_set.aug_prob == pytest.approx(original_train_aug_prob)
        assert train_set._synthetic_pipeline.aug_prob == pytest.approx(original_train_aug_prob)

    def test_pretrain_val_set_aug_prob_mutation_independent(self, pretrain_dataset):
        train_set, val_set = pretrain_dataset.split_data_random(train_ratio=0.8)
        val_set.aug_prob = 0.0

        assert val_set._synthetic_pipeline.aug_prob == pytest.approx(0.0)
        assert val_set._real_pipeline.aug_prob == pytest.approx(0.0)
        assert train_set.aug_prob == pytest.approx(0.5)

    def test_deepcopy_of_dataset_produces_independent_pipelines(self, finetune_dataset):
        cloned = copy.deepcopy(finetune_dataset)
        cloned.aug_prob = 0.0
        assert finetune_dataset.aug_prob == pytest.approx(0.5)
        assert finetune_dataset._synthetic_pipeline.aug_prob == pytest.approx(0.5)
