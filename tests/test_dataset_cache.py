"""Tests for the image caching round-trip in PretrainData and FinetuneData.

Verifies that cached images match their uncached counterparts, ensuring
the flat shared-memory packing/unpacking preserves image data exactly.
"""

import struct

import numpy as np
import pytest
import torch
from PIL import Image
from io import BytesIO

from deepfont.data.config import PretrainDataConfig, FinetuneDataConfig
from deepfont.data.datasets import PretrainData, FinetuneData


def _make_png_bytes(width: int, height: int, seed: int) -> bytes:
    """Create a deterministic grayscale PNG as bytes."""
    rng = np.random.RandomState(seed)
    pixels = rng.randint(0, 256, (height, width), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _write_bcf(path: str, png_list: list[bytes]) -> None:
    """Write a list of PNG byte strings into BCF format."""
    num_files = len(png_list)
    with open(path, "wb") as f:
        # Header: number of files (uint64)
        f.write(struct.pack("<Q", num_files))
        # File sizes (uint64 each)
        for png in png_list:
            f.write(struct.pack("<Q", len(png)))
        # Concatenated file contents
        for png in png_list:
            f.write(png)


def _write_labels(path: str, labels: list[int]) -> None:
    """Write a binary label file (uint32 per label)."""
    with open(path, "wb") as f:
        for label in labels:
            f.write(struct.pack("<I", label))


@pytest.fixture
def pretrain_dataset(tmp_path):
    """Create a small PretrainData dataset with synthetic images only."""
    # Create 20 synthetic images with varying dimensions
    pngs = [_make_png_bytes(width=50 + i * 10, height=110, seed=i) for i in range(20)]
    bcf_path = str(tmp_path / "train.bcf")
    _write_bcf(bcf_path, pngs)

    config = PretrainDataConfig(
        synthetic_bcf_file=bcf_path,
        real_image_dir=None,
        aug_prob=0.0,
        image_normalization="0to1",
    )
    return PretrainData(config)


@pytest.fixture
def finetune_dataset(tmp_path):
    """Create a small FinetuneData dataset with synthetic images and labels."""
    pngs = [_make_png_bytes(width=50 + i * 10, height=110, seed=i + 100) for i in range(20)]
    bcf_path = str(tmp_path / "train.bcf")
    label_path = str(tmp_path / "train.label")
    _write_bcf(bcf_path, pngs)
    _write_labels(label_path, [i % 5 for i in range(20)])

    config = FinetuneDataConfig(
        synthetic_bcf_file=bcf_path,
        label_file=label_path,
        aug_prob=0.0,
        image_normalization="0to1",
    )
    return FinetuneData(config)


@pytest.fixture
def pretrain_dataset_with_aug(tmp_path):
    """PretrainData dataset with augmentations enabled."""
    pngs = [_make_png_bytes(width=50 + i * 10, height=110, seed=i) for i in range(20)]
    bcf_path = str(tmp_path / "train.bcf")
    _write_bcf(bcf_path, pngs)

    config = PretrainDataConfig(
        synthetic_bcf_file=bcf_path,
        real_image_dir=None,
        aug_prob=1.0,
        image_normalization="0to1",
    )
    return PretrainData(config)


@pytest.fixture
def finetune_dataset_with_aug(tmp_path):
    """FinetuneData dataset with augmentations enabled."""
    pngs = [_make_png_bytes(width=50 + i * 10, height=110, seed=i + 100) for i in range(20)]
    bcf_path = str(tmp_path / "train.bcf")
    label_path = str(tmp_path / "train.label")
    _write_bcf(bcf_path, pngs)
    _write_labels(label_path, [i % 5 for i in range(20)])

    config = FinetuneDataConfig(
        synthetic_bcf_file=bcf_path,
        label_file=label_path,
        aug_prob=1.0,
        image_normalization="0to1",
    )
    return FinetuneData(config)


class TestPretrainCacheRoundTrip:
    """Verify PretrainData cache preserves raw image data exactly."""

    def test_cached_raw_images_match_uncached(self, pretrain_dataset):
        """Every cached image's raw pixels must equal the uncached load."""
        # Load uncached references
        uncached = [pretrain_dataset._load_image(i) for i in range(len(pretrain_dataset))]

        pretrain_dataset.cache_images(len(pretrain_dataset))
        assert pretrain_dataset.num_cached_images == len(pretrain_dataset)

        for i in range(pretrain_dataset.num_cached_images):
            offset = pretrain_dataset._cache_offsets[i].item()
            h, w = pretrain_dataset._cache_shapes[i]
            cached = pretrain_dataset._cache_data[offset : offset + h * w].reshape(1, h, w)
            torch.testing.assert_close(cached, uncached[i])

    def test_partial_cache_matches(self, pretrain_dataset):
        """Caching fewer images than the dataset still works correctly."""
        n_cache = 5
        uncached = [pretrain_dataset._load_image(i) for i in range(n_cache)]

        pretrain_dataset.cache_images(n_cache)
        assert pretrain_dataset.num_cached_images == n_cache

        for i in range(n_cache):
            offset = pretrain_dataset._cache_offsets[i].item()
            h, w = pretrain_dataset._cache_shapes[i]
            cached = pretrain_dataset._cache_data[offset : offset + h * w].reshape(1, h, w)
            torch.testing.assert_close(cached, uncached[i])

    def test_getitem_cached_vs_uncached_shapes_match(self, pretrain_dataset):
        """__getitem__ output shape must be the same whether cached or not."""
        # Get uncached output — __getitem__ returns (image, is_real)
        uncached_image, _ = pretrain_dataset[0]

        pretrain_dataset.cache_images(len(pretrain_dataset))
        cached_image, _ = pretrain_dataset[0]

        assert uncached_image.shape == cached_image.shape
        assert cached_image.shape == (1, 105, 105)

    def test_getitem_returns_is_real_flag(self, pretrain_dataset):
        """__getitem__ returns a boolean is_real scalar alongside the image."""
        for i in range(len(pretrain_dataset)):
            _, is_real = pretrain_dataset[i]
            assert is_real.dtype == torch.bool
            assert is_real.ndim == 0
            expected = i >= pretrain_dataset.num_syn_images
            assert is_real.item() == expected

    def test_variable_size_images_preserved(self, pretrain_dataset):
        """Images with different dimensions are all preserved correctly."""
        pretrain_dataset.cache_images(len(pretrain_dataset))

        # Verify shapes vary (our fixture creates images with different widths)
        widths = [pretrain_dataset._cache_shapes[i][1].item()
                  for i in range(pretrain_dataset.num_cached_images)]
        assert len(set(widths)) > 1, "Test fixture should have varying widths"

    def test_cache_after_split(self, pretrain_dataset):
        """Caching after split_data_random works correctly."""
        train_set, val_set = pretrain_dataset.split_data_random(0.8)

        # Load uncached from train split
        uncached = [train_set._load_image(i) for i in range(len(train_set))]

        train_set.cache_images(len(train_set))

        for i in range(train_set.num_cached_images):
            offset = train_set._cache_offsets[i].item()
            h, w = train_set._cache_shapes[i]
            cached = train_set._cache_data[offset : offset + h * w].reshape(1, h, w)
            torch.testing.assert_close(cached, uncached[i])


class TestFinetuneCacheRoundTrip:
    """Verify FinetuneData cache preserves raw image data exactly."""

    def test_cached_raw_images_match_uncached(self, finetune_dataset):
        """Every cached image's raw pixels must equal the uncached load."""
        uncached = [finetune_dataset._load_image(i) for i in range(len(finetune_dataset))]

        finetune_dataset.cache_images(len(finetune_dataset))
        assert finetune_dataset.num_cached_images == len(finetune_dataset)

        for i in range(finetune_dataset.num_cached_images):
            offset = finetune_dataset._cache_offsets[i].item()
            h, w = finetune_dataset._cache_shapes[i]
            cached = finetune_dataset._cache_data[offset : offset + h * w].reshape(1, h, w)
            torch.testing.assert_close(cached, uncached[i])

    def test_getitem_returns_correct_label_when_cached(self, finetune_dataset):
        """Labels must be unaffected by caching."""
        _, label_before = finetune_dataset[0]
        finetune_dataset.cache_images(len(finetune_dataset))
        _, label_after = finetune_dataset[0]
        assert label_before == label_after

    def test_cache_after_split(self, finetune_dataset):
        """Caching after stratified split works correctly."""
        train_set, val_set = finetune_dataset.split_data_random(0.8)

        uncached = [train_set._load_image(i) for i in range(len(train_set))]

        train_set.cache_images(len(train_set))

        for i in range(train_set.num_cached_images):
            offset = train_set._cache_offsets[i].item()
            h, w = train_set._cache_shapes[i]
            cached = train_set._cache_data[offset : offset + h * w].reshape(1, h, w)
            torch.testing.assert_close(cached, uncached[i])


class TestPretrainCacheImmutability:
    """Verify the cache slice is cloned before augmentation can touch it.

    The augmentation pipeline currently happens to allocate new arrays
    internally, so checking cache contents after __getitem__ would pass
    even without the clone().  Instead, we replace the dataset's persistent
    pipeline instance with a callable that writes into its input and verify
    the cache is still intact.
    """

    def test_inplace_augmentation_does_not_corrupt_cache(self, pretrain_dataset):
        """Cache must survive an augmentation that mutates its input array."""
        ds = pretrain_dataset
        ds.cache_images(len(ds))
        snapshot = ds._cache_data.clone()

        # Replace the persistent pipeline instances with callables that zero
        # the input array in-place before delegating to the real pipelines.
        original_syn = ds._synthetic_pipeline
        original_real = ds._real_pipeline

        def destructive_syn(image: np.ndarray) -> np.ndarray:
            image[:] = 0
            return original_syn(image)

        def destructive_real(image: np.ndarray) -> np.ndarray:
            image[:] = 0
            return original_real(image)

        ds._synthetic_pipeline = destructive_syn
        ds._real_pipeline = destructive_real
        try:
            for i in range(len(ds)):
                _ = ds[i]
        finally:
            ds._synthetic_pipeline = original_syn
            ds._real_pipeline = original_real

        torch.testing.assert_close(ds._cache_data, snapshot)

    def test_cache_unchanged_after_getitem(self, pretrain_dataset_with_aug):
        """Accessing items must not modify the underlying cache buffer."""
        ds = pretrain_dataset_with_aug
        ds.cache_images(len(ds))
        snapshot = ds._cache_data.clone()

        for _ in range(3):
            for i in range(len(ds)):
                _ = ds[i]

        torch.testing.assert_close(ds._cache_data, snapshot)


class TestFinetuneCacheImmutability:
    """Verify the cache slice is cloned before augmentation can touch it."""

    def test_inplace_augmentation_does_not_corrupt_cache(self, finetune_dataset):
        """Cache must survive an augmentation that mutates its input array."""
        ds = finetune_dataset
        ds.cache_images(len(ds))
        snapshot = ds._cache_data.clone()

        # Replace the persistent pipeline instance with a callable that zeros
        # the input array in-place before delegating to the real pipeline.
        original_syn = ds._synthetic_pipeline

        def destructive_syn(image: np.ndarray) -> np.ndarray:
            image[:] = 0
            return original_syn(image)

        ds._synthetic_pipeline = destructive_syn
        try:
            for i in range(len(ds)):
                _ = ds[i]
        finally:
            ds._synthetic_pipeline = original_syn

        torch.testing.assert_close(ds._cache_data, snapshot)

    def test_cache_unchanged_after_getitem(self, finetune_dataset_with_aug):
        """Accessing items must not modify the underlying cache buffer."""
        ds = finetune_dataset_with_aug
        ds.cache_images(len(ds))
        snapshot = ds._cache_data.clone()

        for _ in range(3):
            for i in range(len(ds)):
                _ = ds[i]

        torch.testing.assert_close(ds._cache_data, snapshot)
