import os
import copy
from io import BytesIO
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# Increase the maximum text chunk size for PNG images
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset, get_worker_info

from .bcf import BCFStoreFile, read_label
from .config import EvalDataConfig, FinetuneDataConfig, PretrainDataConfig
from .augmentations import (
    EvalAugmentationPipeline,
    RealAugmentationPipeline,
    SyntheticAugmentationPipeline,
)

PngImagePlugin.MAX_TEXT_CHUNK = 1048576 * 10  # ty: ignore[invalid-assignment]

# Load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True  # ty: ignore[invalid-assignment]


def bcf_worker_init_fn(worker_id: int) -> None:
    """Reopen BCF file handles so each DataLoader worker has its own.

    When PyTorch forks worker processes the parent's file descriptors are
    shared, making the seek/read pair in BCFStoreFile.get() racy. Calling
    reset_file_pointer() gives every worker an independent descriptor.

    Args:
        worker_id: Worker index supplied by the DataLoader (unused but
            required by the worker_init_fn signature).
    """
    info = get_worker_info()
    if info is None:
        return
    dataset = info.dataset
    if hasattr(dataset, "bcf_store"):
        dataset.bcf_store.reset_file_pointer()  # ty: ignore[unresolved-attribute]


class BaseDataset(Dataset):
    """Base class providing shared caching, normalization, and BCF-store helpers.

    Subclasses must set ``self.image_normalization`` before calling any helper
    that depends on it, and must implement ``_load_image``.
    """

    image_normalization: str

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def _load_image(self, index: int) -> torch.Tensor: ...

    @staticmethod
    def _build_bcf_store(bcf_path: str, df: pd.DataFrame) -> BCFStoreFile:
        """Build a BCFStoreFile, using per-entry offsets when available.

        Args:
            bcf_path: Absolute path to the BCF file.
            df: Manifest DataFrame slice for this BCF file. Must contain
                ``bcf_offset`` and ``bcf_size`` columns when offset-indexed.

        Returns:
            A BCFStoreFile opened either with or without an offset table.
        """
        if "bcf_offset" in df.columns and not df["bcf_offset"].isna().any():
            return BCFStoreFile.from_manifest(
                bcf_path,
                df["bcf_offset"].to_numpy(np.uint64),
                df["bcf_size"].to_numpy(np.uint64),
            )
        return BCFStoreFile(bcf_path)

    def _init_cache(self) -> None:
        """Initialize the flat image cache to an empty state."""
        self._cache_data: torch.Tensor | None = None
        self._cache_offsets: torch.Tensor | None = None
        self._cache_shapes: torch.Tensor | None = None
        self.num_cached_images = 0

    def _normalize(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize an image tensor according to ``self.image_normalization``.

        Args:
            image: Raw image tensor with pixel values in [0, 255].

        Returns:
            Normalized image tensor (float32).
        """
        if self.image_normalization == "0to1":
            return image / 255.0
        elif self.image_normalization == "-1to1":
            return (image / 127.5) - 1.0
        return image

    def _get_cached_or_loaded(self, index: int) -> torch.Tensor:
        """Return the raw image at *index* from the cache or from disk.

        Args:
            index: Dataset index of the image to retrieve.

        Returns:
            Raw image tensor of shape (1, H, W) with dtype uint8.
        """
        if self.num_cached_images > 0 and index < self.num_cached_images:
            assert self._cache_offsets is not None
            assert self._cache_shapes is not None
            assert self._cache_data is not None
            offset = self._cache_offsets[index].item()
            h, w = self._cache_shapes[index]
            # clone() so the augmentation pipeline doesn't write back
            # into the cache buffer (which would also trigger COW copies
            # in forked DataLoader workers).
            return self._cache_data[offset : offset + h * w].reshape(1, h, w).clone()
        return self._load_image(index)

    def cache_images(self, num_images_to_cache: int) -> None:
        """Preloads images into memory for faster iteration during training.

        Loads the first N images (in their raw, unaugmented form) into a single
        flat shared-memory tensor. An offsets tensor and shapes tensor allow
        reconstructing individual images on access.

        Using a single shared-memory tensor (instead of one per image) avoids
        exhausting file descriptor limits and prevents copy-on-write duplication
        when DataLoader workers are forked.

        Augmentations are still applied on-the-fly even to cached images, so each
        access returns a different augmented version.

        Args:
            num_images_to_cache: The maximum number of images to cache. If this
                exceeds the dataset size, all images will be cached. Memory usage
                scales linearly with this value.

        Note:
            Caching loads RAW images before augmentation. The same cached image
            produces different outputs due to random augmentations. Images are
            cached starting from index 0, so synthetic images are cached first.

        Warning:
            This method is incompatible with split_data_random(). The cache must
            be empty before splitting, and should be repopulated after splitting.
        """
        self.num_cached_images = min(num_images_to_cache, len(self))
        # First pass: record shapes to compute total size without
        # keeping all image tensors in memory simultaneously.
        shapes = []
        total_pixels = 0
        for index in range(self.num_cached_images):
            image = self._load_image(index)
            h, w = image.shape[1], image.shape[2]
            shapes.append((h, w))
            total_pixels += h * w
        # Allocate the flat buffer and metadata tensors, then fill in
        # a second pass so only one image is in memory at a time.
        offsets = []
        offset = 0
        for h, w in shapes:
            offsets.append(offset)
            offset += h * w
        self._cache_offsets = torch.tensor(offsets, dtype=torch.int64)
        self._cache_shapes = torch.tensor(shapes, dtype=torch.int64)
        self._cache_data = torch.empty(total_pixels, dtype=torch.uint8)
        # Second pass: load images directly into the flat buffer.
        for index in range(self.num_cached_images):
            image = self._load_image(index)
            o = self._cache_offsets[index].item()
            h, w = shapes[index]
            self._cache_data[o : o + h * w] = image.reshape(-1)


class PretrainData(BaseDataset):
    """PyTorch Dataset for autoencoder pretraining with mixed synthetic and real images.

    This dataset combines synthetic images from a BCF (Binary Concatenated File) store
    with real images from a directory to create a diverse training set for autoencoder
    pretraining. The mixed data helps the model learn robust representations that
    generalize across both synthetic and real-world image domains.

    The dataset applies different augmentation pipelines depending on image type:
    synthetic images undergo more aggressive augmentation (blur, noise, gradients)
    while real images use gentler transformations. This is essential for bridging
    the domain gap between rendered and natural images.

    Features:
        - Automatic handling of synthetic (BCF) and real (directory) images
        - Configurable augmentation probability
        - Image caching for faster iteration
        - Random train/validation splitting with stratification
        - Real image upsampling to match synthetic data size
        - Multiple normalization schemes (0-1 or -1 to 1)

    Attributes:
        config: PretrainDataConfig instance controlling data sources and augmentation.
        synthetic_bcf_file: Path to the BCF store file containing synthetic images.
        real_image_dir: Path to the directory containing real images.
        aug_prob: Probability of applying each augmentation (0.0 to 1.0).
        image_normalization: Normalization scheme ("0to1" or "-1to1").
        bcf_store: BCFStoreFile instance for reading synthetic images.
        num_syn_images: Total count of synthetic images.
        num_real_images: Total count of real images.
        image_cache: List of cached image tensors.
        num_cached_images: Number of images currently cached.
    """

    def __init__(self, config: PretrainDataConfig):
        """Initializes the PretrainData dataset with synthetic and real images.

        Loads synthetic images from a BCF store file and discovers real images in
        the specified directory. Sets up the augmentation pipeline parameters and
        validates the normalization scheme.

        Args:
            config: A PretrainDataConfig instance specifying the data sources
                and augmentation settings. The config is validated at construction
                time via Pydantic.

        Raises:
            FileNotFoundError: If synthetic_bcf_file doesn't exist.
            IOError: If the BCF file cannot be read.
        """
        # Store the config and extract mutable runtime parameters
        self.config = config
        self.synthetic_bcf_file = config.synthetic_bcf_file
        self.real_image_dir = config.real_image_dir
        self._aug_prob = config.aug_prob
        self._synthetic_pipeline = SyntheticAugmentationPipeline(self._aug_prob)
        self._real_pipeline = RealAugmentationPipeline(self._aug_prob)
        self.image_normalization = config.image_normalization

        if config.manifest_file is not None:
            manifest_dir = Path(config.manifest_file).resolve().parent
            df = pd.read_parquet(config.manifest_file)
            syn_df = df[df["image_type"] == "synthetic"]
            real_df = df[df["image_type"] == "real"]
            # Build BCF store if there are synthetic images
            if len(syn_df) > 0:
                bcf_path = str(manifest_dir / syn_df["bcf_file"].iloc[0])
                self.bcf_store = self._build_bcf_store(bcf_path, syn_df)
                self.syn_image_index_list = syn_df["bcf_index"].to_numpy(np.int64)
            else:
                self.bcf_store = BCFStoreFile(self.synthetic_bcf_file)
                self.syn_image_index_list = np.empty(0, dtype=np.int64)
            self.num_syn_images = len(self.syn_image_index_list)
            # Resolve real image paths relative to manifest directory
            self.real_image_path_list = [
                str(manifest_dir / p) for p in real_df["filepath"].dropna().tolist()
            ]
            self.num_real_images = len(self.real_image_path_list)
        else:
            # Load the BCF store file
            self.bcf_store = BCFStoreFile(self.synthetic_bcf_file)
            # Find the number of synthetic images
            self.num_syn_images = self.bcf_store.size()
            # Create the synthetic image index list
            self.syn_image_index_list = np.arange(self.num_syn_images)
            # Find the number of images and their full paths
            self.num_real_images, self.real_image_path_list = self._find_images()

        self._init_cache()

    @property
    def aug_prob(self) -> float:
        return self._aug_prob

    @aug_prob.setter
    def aug_prob(self, value: float) -> None:
        self._aug_prob = value
        self._synthetic_pipeline.aug_prob = value
        self._real_pipeline.aug_prob = value

    def __len__(self) -> int:
        """Returns the total number of images in the dataset.

        Calculates the dataset size as the sum of synthetic images from the BCF
        store and real images from the directory.

        Returns:
            The total number of images (synthetic + real) available in the dataset.
        """
        return self.num_syn_images + self.num_real_images

    def __getitem__(self, index) -> torch.Tensor:
        """Retrieves and augments an image at the specified index.

        Loads the image (from cache if available, otherwise from disk), applies
        the appropriate augmentation pipeline (synthetic or real), and normalizes
        the pixel values according to the configured normalization scheme.

        Args:
            index: The index of the image to retrieve. Indices [0, num_syn_images)
                correspond to synthetic images, while indices [num_syn_images, total)
                correspond to real images.

        Returns:
            A normalized image tensor of shape (1, 105, 105) and dtype float32,
            ready for model input.
        """
        return self._normalize(self._get_image(index))

    def _load_image(self, index: int) -> torch.Tensor:
        """Loads a raw image from disk without augmentation.

        Retrieves an image from either the BCF store (for synthetic images) or
        the file system (for real images) based on the index. For real images,
        validates that the image has non-zero dimensions and resamples if necessary
        to avoid corrupted or invalid images.

        Args:
            index: The index of the image to load. Indices < num_syn_images load
                from the BCF store, while higher indices load from the real image
                directory.

        Returns:
            A raw image tensor of shape (1, H, W) with dtype uint8, where H and W
            are the original image dimensions before augmentation.

        Note:
            Real images with dimensions of 0 or 1 are considered invalid and are
            replaced by randomly selected valid images from the dataset.
        """
        # Check if the image is synthetic or real
        if index < self.num_syn_images:
            # Get the synthetic image
            image = Image.open(
                BytesIO(self.bcf_store.get(int(self.syn_image_index_list[index])))
            ).convert("L")
        else:
            # Get the real image
            image = Image.open(self.real_image_path_list[index - self.num_syn_images]).convert("L")
            # Check if any of the image dimensions are zero and resample if needed
            while 0 in image.size or 1 in image.size:
                image = Image.open(
                    self.real_image_path_list[np.random.randint(0, self.num_real_images)]
                ).convert("L")
        # Convert the image to a numpy array
        image = np.array(image, dtype=np.uint8)
        # Convert the image to a torch tensor
        image = torch.from_numpy(image).unsqueeze(0)

        return image

    def _get_image(self, index: int) -> torch.Tensor:
        """Retrieves an image and applies appropriate augmentations.

        Either loads from cache (if available) or from disk, then applies the
        augmentation pipeline corresponding to the image type (synthetic or real).
        The augmented image is returned as a float tensor ready for normalization.

        Args:
            index: The index of the image to retrieve and augment.

        Returns:
            An augmented image tensor of shape (1, 105, 105) with dtype float32
            containing unnormalized pixel values in [0, 255].

        Note:
            Images with index < num_syn_images use the synthetic pipeline, while
            others use the real image pipeline.
        """
        image = self._get_cached_or_loaded(index).numpy()
        if index < self.num_syn_images:
            image = self._synthetic_pipeline(image[0])
        else:
            image = self._real_pipeline(image[0])
        return torch.from_numpy(image).float().unsqueeze(0)

    def split_data_random(self, train_ratio: float = 0.8):
        """Randomly splits the dataset into training and validation subsets.

        Creates two independent dataset instances by randomly partitioning both
        synthetic and real images according to the specified ratio. The split is
        stratified across both data types to maintain similar distributions in
        train and validation sets.

        This method creates deep copies of the dataset with separate image lists,
        ensuring that train and validation sets don't share indices.

        Args:
            train_ratio: The fraction of data to allocate to training (0.0 to 1.0).
                For example, 0.8 means 80% training and 20% validation. Default is 0.8.

        Returns:
            A tuple of (train_data, val_data) where both are PretrainData instances
            with independently shuffled subsets of the original dataset.

        Raises:
            ValueError: If the image cache is not empty. Images must not be cached
                before splitting to avoid memory issues with dataset copies.

        Note:
            The split is truly random and not reproducible unless a random seed is
            set before calling this method. Both synthetic and real images are
            shuffled and split independently.
        """
        if self.num_cached_images > 0:
            raise ValueError("The image cache must be empty before splitting the data.")
        # Shuffle the image paths and synthetic indices
        real_image_path_list = copy.deepcopy(self.real_image_path_list)
        np.random.shuffle(real_image_path_list)
        syn_image_index_list = copy.deepcopy(self.syn_image_index_list)
        np.random.shuffle(syn_image_index_list)
        # Split
        train_real_image_paths = real_image_path_list[: int(self.num_real_images * train_ratio)]
        val_real_image_paths = real_image_path_list[int(self.num_real_images * train_ratio) :]
        train_syn_image_indices = syn_image_index_list[: int(self.num_syn_images * train_ratio)]
        val_syn_image_indices = syn_image_index_list[int(self.num_syn_images * train_ratio) :]
        # Create the training and validation datasets
        train_data = copy.deepcopy(self)
        val_data = copy.deepcopy(self)
        train_data.real_image_path_list = train_real_image_paths
        val_data.real_image_path_list = val_real_image_paths
        train_data.syn_image_index_list = train_syn_image_indices
        val_data.syn_image_index_list = val_syn_image_indices
        train_data.num_real_images = len(train_real_image_paths)
        val_data.num_real_images = len(val_real_image_paths)
        train_data.num_syn_images = len(train_syn_image_indices)
        val_data.num_syn_images = len(val_syn_image_indices)

        return train_data, val_data

    def _find_images(self) -> tuple[int, list[str]]:
        """Discovers all valid image files in the real images directory.

        Scans the data folder for files with common image extensions (.png, .jpg,
        .jpeg, .gif) and returns their count and full absolute paths. If no data
        folder is specified (None), returns empty results.

        Returns:
            A tuple of (num_images, image_paths) where num_images is the count of
            found images and image_paths is a list of absolute file paths.
            Returns (0, []) if real_image_dir is None.
        """
        if self.real_image_dir is None:
            return 0, []
        # Get the full image paths
        image_path_list = [
            os.path.join(self.real_image_dir, x)
            for x in os.listdir(self.real_image_dir)
            if x.endswith((".png", ".jpg", ".jpeg", ".gif"))
        ]
        return len(image_path_list), image_path_list

    def upsample_real_images(self):
        """Upsamples real images to match the number of synthetic images.

        Randomly resamples the real image list (with replacement) to equal the
        synthetic image count. This is useful for balancing the dataset when there
        are significantly fewer real images than synthetic ones. Some real images
        will appear multiple times in the dataset after upsampling.

        This operation modifies the dataset in-place, updating both the image name
        list and the count.

        Note:
            This uses random sampling with replacement, so the same real image may
            appear multiple times in a single epoch. The sampling is not stratified
            and has no guarantees about image diversity.
        """
        # Upsample the real images
        self.real_image_path_list = np.random.choice(
            self.real_image_path_list, self.num_syn_images, replace=True
        ).tolist()
        self.num_real_images = len(self.real_image_path_list)


class FinetuneData(BaseDataset):
    """PyTorch Dataset for font classification fine-tuning with labeled data.

    This dataset is designed for supervised fine-tuning of font classification models.
    It loads synthetic images and their corresponding class labels from BCF and label
    files, applying augmentations during training to improve model robustness.

    Unlike PretrainData, this dataset:
        - Uses only synthetic images (no real images)
        - Returns (image, label) pairs for supervised learning
        - Supports stratified train/validation splitting by class
        - Always uses synthetic augmentation pipeline

    Features:
        - Stratified random splitting that maintains class distribution
        - Image caching for faster training iteration
        - Multiple normalization schemes
        - Automatic validation of image-label correspondence

    Attributes:
        config: FinetuneDataConfig instance controlling data source and augmentation.
        synthetic_bcf_file: Path to the BCF store file containing labeled synthetic images.
        label_file: Path to the binary label file (uint32 format).
        aug_prob: Probability of applying each augmentation (0.0 to 1.0).
        image_normalization: Normalization scheme ("0to1" or "-1to1").
        bcf_store: BCFStoreFile instance for reading images.
        labels: NumPy array of integer class labels.
        num_images: Total count of labeled images.
        image_cache: List of cached image tensors.
        num_cached_images: Number of images currently cached.
    """

    def __init__(self, config: FinetuneDataConfig):
        """Initializes the FinetuneData dataset with labeled synthetic images.

        Loads images from a BCF store and their corresponding labels from a binary
        label file. Validates that the number of images and labels match to ensure
        data consistency.

        Args:
            config: A FinetuneDataConfig instance specifying the data source
                and augmentation settings. The config is validated at construction
                time via Pydantic.

        Raises:
            ValueError: If the number of images and labels don't match.
            FileNotFoundError: If synthetic_bcf_file or label_file doesn't exist.
            IOError: If the files cannot be read.
        """
        # Store the config and extract mutable runtime parameters
        self.config = config
        self.synthetic_bcf_file = config.synthetic_bcf_file
        self.label_file = config.label_file
        self._aug_prob = config.aug_prob
        self._synthetic_pipeline = SyntheticAugmentationPipeline(self._aug_prob)
        self.image_normalization = config.image_normalization

        if config.manifest_file is not None:
            manifest_dir = Path(config.manifest_file).resolve().parent
            df = pd.read_parquet(config.manifest_file)
            bcf_path = str(manifest_dir / df["bcf_file"].iloc[0])
            self.bcf_store = self._build_bcf_store(bcf_path, df)
            self.image_index_list = df["bcf_index"].to_numpy(np.int64)
            self.labels = df["label"].to_numpy(np.uint32)
            self.num_images = len(self.image_index_list)
        else:
            # Load the BCF store file
            self.bcf_store = BCFStoreFile(self.synthetic_bcf_file)
            # Load the labels
            self.labels = read_label(self.label_file)
            # Find the number of images
            self.num_images = self.bcf_store.size()
            # Create the image index list
            self.image_index_list = np.arange(self.num_images)
            # Check that the images and labels are the same size
            if self.num_images != len(self.labels):
                raise ValueError("The number of images and labels must be the same.")

        self._init_cache()

    @property
    def aug_prob(self) -> float:
        return self._aug_prob

    @aug_prob.setter
    def aug_prob(self, value: float) -> None:
        self._aug_prob = value
        self._synthetic_pipeline.aug_prob = value

    def __len__(self) -> int:
        """Returns the total number of labeled images in the dataset.

        Returns:
            The number of images available for fine-tuning.
        """
        return self.num_images

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieves an augmented image and its corresponding label.

        Loads the image, applies synthetic augmentation pipeline, normalizes pixel
        values, and returns both the processed image and its class label.

        Args:
            index: The index of the image-label pair to retrieve.

        Returns:
            A tuple of (image, label) where:
                - image: Normalized tensor of shape (1, 105, 105) with dtype float32
                - label: Integer class label tensor with dtype int64 (long)
        """
        image = self._normalize(self._get_image(index))
        label = torch.tensor(int(self.labels[index])).long()
        return image, label

    def _load_image(self, index: int) -> torch.Tensor:
        """Loads a raw synthetic image from the BCF store.

        Retrieves an image at the specified index, converts it to grayscale,
        and returns it as an unaugmented tensor.

        Args:
            index: The index of the image to load from the BCF store.

        Returns:
            A raw image tensor of shape (1, H, W) with dtype uint8, where H and W
            are the original image dimensions before augmentation.
        """
        # Get the synthetic image
        image = Image.open(BytesIO(self.bcf_store.get(int(self.image_index_list[index])))).convert(
            "L"
        )
        # Convert the image to a numpy array
        image = np.array(image, dtype=np.uint8)
        # Convert the image to a torch tensor
        image = torch.from_numpy(image).unsqueeze(0)

        return image

    def _get_image(self, index: int) -> torch.Tensor:
        """Retrieves an image and applies synthetic augmentations.

        Either loads from cache (if available) or from disk, then applies the
        synthetic augmentation pipeline. Returns the augmented image as a float
        tensor ready for normalization.

        Args:
            index: The index of the image to retrieve and augment.

        Returns:
            An augmented image tensor of shape (1, 105, 105) with dtype float32
            containing unnormalized pixel values in [0, 255].

        Note:
            Always uses the synthetic augmentation pipeline since FinetuneData
            only contains synthetic images.
        """
        image = self._get_cached_or_loaded(index).numpy()
        image = self._synthetic_pipeline(image[0])
        return torch.from_numpy(image).float().unsqueeze(0)

    def split_data_random(self, train_ratio: float = 0.8):
        """Performs stratified random split into training and validation sets.

        Creates two independent dataset instances by randomly partitioning the data
        while maintaining class distribution. Each class is split according to the
        specified ratio, ensuring that rare classes are represented in both sets.

        This stratified approach is crucial for classification tasks with imbalanced
        classes, as it prevents validation sets from missing entire classes.

        Args:
            train_ratio: The fraction of data per class to allocate to training
                (0.0 to 1.0). For example, 0.8 means 80% of each class goes to
                training and 20% to validation. Default is 0.8.

        Returns:
            A tuple of (train_data, val_data) where both are FinetuneData instances
            with stratified subsets maintaining the original class distribution.

        Raises:
            ValueError: If the image cache is not empty. Images must not be cached
                before splitting to avoid memory issues.

        Note:
            Unlike PretrainData's random split, this uses stratified sampling to
            ensure all classes appear in both train and validation sets. The split
            is random within each class but not globally reproducible unless a seed
            is set.
        """
        if self.num_cached_images > 0:
            raise ValueError("The image cache must be empty before splitting the data.")
        # Randomly sample uniformly from all the classes
        train_index_list = []
        val_index_list = []
        for label in np.unique(self.labels):
            label_indices = np.where(self.labels == label)[0]
            np.random.shuffle(label_indices)
            train_index_list.extend(label_indices[: int(len(label_indices) * train_ratio)])
            val_index_list.extend(label_indices[int(len(label_indices) * train_ratio) :])
        # Split the image names
        train_index_list = [self.image_index_list[i] for i in train_index_list]
        val_index_list = [self.image_index_list[i] for i in val_index_list]
        # Create the training and validation datasets
        train_data = copy.deepcopy(self)
        val_data = copy.deepcopy(self)
        train_data.image_index_list = train_index_list
        val_data.image_index_list = val_index_list
        train_data.labels = np.array([self.labels[i] for i in train_index_list])
        val_data.labels = np.array([self.labels[i] for i in val_index_list])
        train_data.num_images = len(train_index_list)
        val_data.num_images = len(val_index_list)

        return train_data, val_data


class EvalData(BaseDataset):
    """PyTorch Dataset for font classification evaluation with test-time augmentation.

    This dataset is optimized for model evaluation and inference, generating multiple
    augmented crops of each image to enable test-time augmentation (TTA). By creating
    multiple views of the same image and averaging predictions, the model can achieve
    more robust and accurate classifications.

    Unlike training datasets, this:
        - Generates multiple crops per image (configurable)
        - Uses only geometric augmentations (no blur, noise, or color changes)
        - Returns batched crops for ensemble prediction
        - Has no augmentation probability (all augmentations always applied)

    The evaluation pipeline uses more aggressive scaling (±40%) compared to training
    (±15%) to ensure comprehensive coverage of possible image variations.

    Attributes:
        config: EvalDataConfig instance controlling data source and TTA settings.
        synthetic_bcf_file: Path to the BCF store file containing test images.
        label_file: Path to the binary label file (uint32 format).
        image_normalization: Normalization scheme ("0to1" or "-1to1").
        num_image_crops: Number of augmented crops to generate per image.
        bcf_store: BCFStoreFile instance for reading images.
        labels: NumPy array of integer class labels.
        num_images: Total count of test images.
    """

    def __init__(self, config: EvalDataConfig):
        """Initializes the EvalData dataset for test-time augmentation.

        Loads images and labels for evaluation, configuring the test-time augmentation
        parameters. Validates that images and labels are properly paired.

        Args:
            config: An EvalDataConfig instance specifying the data source and
                TTA settings. The config is validated at construction time via
                Pydantic.

        Raises:
            ValueError: If the number of images and labels don't match.
            FileNotFoundError: If synthetic_bcf_file or label_file doesn't exist.
            IOError: If the files cannot be read.
        """
        # Store the config and extract parameters
        self.config = config
        self.synthetic_bcf_file = config.synthetic_bcf_file
        self.label_file = config.label_file
        self.image_normalization = config.image_normalization
        self.num_image_crops = config.num_image_crops
        self._eval_pipeline = EvalAugmentationPipeline()

        if config.manifest_file is not None:
            manifest_dir = Path(config.manifest_file).resolve().parent
            df = pd.read_parquet(config.manifest_file)
            bcf_path = str(manifest_dir / df["bcf_file"].iloc[0])
            self.bcf_store = self._build_bcf_store(bcf_path, df)
            self.labels = df["label"].to_numpy(np.uint32)
            self.num_images = len(self.labels)
        else:
            # Load the BCF store file
            self.bcf_store = BCFStoreFile(self.synthetic_bcf_file)
            # Load the labels
            self.labels = read_label(self.label_file)
            # Find the number of images
            self.num_images = self.bcf_store.size()
            # Check that the images and labels are the same size
            if self.num_images != len(self.labels):
                raise ValueError("The number of images and labels must be the same.")

    def __len__(self) -> int:
        """Returns the total number of test images in the dataset.

        Returns:
            The number of images available for evaluation.
        """
        return self.num_images

    def _load_image(self, index: int) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        """Generates multiple augmented crops of an image with its label.

        Loads an image, applies the evaluation augmentation pipeline to create
        multiple random crops, normalizes all crops, and returns them as a batch
        along with the single ground-truth label.

        During inference, predictions on all crops can be averaged or ensembled
        for more robust classification.

        Args:
            index: The index of the image to retrieve and augment.

        Returns:
            A tuple of (image_crops, label) where:
                - image_crops: Normalized tensor of shape (num_image_crops, 1, 105, 105)
                  with dtype float32, containing multiple augmented views of the image
                - label: Integer class label tensor with dtype int64 (long)

        Example:
            >>> dataset = EvalData(EvalDataConfig(
            ...     synthetic_bcf_file='test.bcf',
            ...     label_file='test.labels',
            ...     num_image_crops=10,
            ... ))
            >>> images, label = dataset[0]
            >>> images.shape
            torch.Size([10, 1, 105, 105])
            >>> # Feed all 10 crops to model and average predictions
        """
        # Get the image
        image = Image.open(BytesIO(self.bcf_store.get(index))).convert("L")
        # Convert the image to a numpy array
        image = np.array(image, dtype=np.uint8)
        # Apply the augmentations
        image_crops = self._eval_pipeline(image, self.num_image_crops)
        # Convert the image to a tensor and normalize
        image = self._normalize(torch.tensor(image_crops).float().unsqueeze(1))
        # Get the label
        label = torch.tensor(int(self.labels[index])).long()

        return image, label
