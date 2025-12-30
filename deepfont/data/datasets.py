import os
import copy
from io import BytesIO

import numpy as np
import torch

# Increase the maximum text chunk size for PNG images
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import Dataset

from .bcf import BCFStoreFile, read_label
from .augmentations import eval_pipeline, augmentation_pipeline

PngImagePlugin.MAX_TEXT_CHUNK = 1048576 * 10

# Load truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class PretrainData(Dataset):
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
        bcf_store_file: Path to the BCF store file containing synthetic images.
        data_folder_name: Path to the directory containing real images.
        aug_prob: Probability of applying each augmentation (0.0 to 1.0).
        image_normalization: Normalization scheme ("0to1" or "-1to1").
        bcf_store: BCFStoreFile instance for reading synthetic images.
        num_syn_images: Total count of synthetic images.
        num_real_images: Total count of real images.
        image_cache: List of cached image tensors.
        num_cached_images: Number of images currently cached.
    """

    def __init__(
        self,
        bcf_store_file: str,
        data_folder_name: str,
        aug_prob: float,
        image_normalization: str = "0to1",
    ):
        """Initializes the PretrainData dataset with synthetic and real images.

        Loads synthetic images from a BCF store file and discovers real images in
        the specified directory. Sets up the augmentation pipeline parameters and
        validates the normalization scheme.

        Args:
            bcf_store_file: The path to the BCF store file containing concatenated
                synthetic images. The file must be in valid BCF format.
            data_folder_name: The path to the directory containing real images.
                Supports .png, .jpg, .jpeg, and .gif files. Can be None if using
                only synthetic images.
            aug_prob: The probability (0.0 to 1.0) of applying each augmentation
                in the pipeline. Higher values result in more aggressive augmentation.
                Typical values range from 0.3 to 0.8.
            image_normalization: The normalization scheme to apply to images.
                Either "0to1" (scales to [0, 1]) or "-1to1" (scales to [-1, 1]).
                Default is "0to1".

        Raises:
            ValueError: If image_normalization is not "0to1" or "-1to1".
            FileNotFoundError: If bcf_store_file doesn't exist.
            IOError: If the BCF file cannot be read.
        """
        # Store the parameters
        self.bcf_store_file = bcf_store_file
        self.data_folder_name = data_folder_name
        self.aug_prob = aug_prob
        self.image_normalization = image_normalization
        # Check the feature normalization type
        if self.image_normalization not in ["0to1", "-1to1"]:
            raise ValueError("The image normalization type must be either '0to1' or '-1to1'.")
        # Load the BCF store file
        self.bcf_store = BCFStoreFile(bcf_store_file)
        # Find the number of synthetic images
        self.num_syn_images = self.bcf_store.size()
        # Create the sythetic image index list
        self.syn_image_index_list = np.arange(self.num_syn_images)
        # Find the number of images and their names
        self.num_real_images, self.real_image_name_list = self._find_images()
        # Create the image cache
        self.image_cache = []
        self.num_cached_images = 0

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
        # Get the image
        image = self._get_image(index)
        # Normalize the image
        if self.image_normalization == "0to1":
            image = image / 255.0
        elif self.image_normalization == "-1to1":
            image = (image / 127.5) - 1.0

        return image

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
            image = Image.open(
                os.path.join(
                    self.data_folder_name,
                    self.real_image_name_list[index - self.num_syn_images],
                )
            ).convert("L")
            # Check if any of the image dimensions are zero and respample if needed
            while 0 in image.size or 1 in image.size:
                image = Image.open(
                    os.path.join(
                        self.data_folder_name,
                        self.real_image_name_list[np.random.randint(0, self.num_real_images)],
                    )
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
        # Check if the image is cached
        if self.num_cached_images > 0 and index < self.num_cached_images:
            image = self.image_cache[index]
        else:
            image = self._load_image(index)
        # Apply the augmentation pipeline
        image = image.numpy()
        image = augmentation_pipeline(
            image[0],
            "synthetic" if index < self.num_syn_images else "real",
            self.aug_prob,
        )
        image = torch.from_numpy(image).float().unsqueeze(0)

        return image

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
        # Shuffle the image names
        real_image_name_list = copy.deepcopy(self.real_image_name_list)
        np.random.shuffle(real_image_name_list)
        syn_image_index_list = copy.deepcopy(self.syn_image_index_list)
        np.random.shuffle(syn_image_index_list)
        # Split the image names
        train_real_image_names = real_image_name_list[: int(self.num_real_images * train_ratio)]
        val_real_image_names = real_image_name_list[int(self.num_real_images * train_ratio) :]
        train_syn_image_indices = syn_image_index_list[: int(self.num_syn_images * train_ratio)]
        val_syn_image_indices = syn_image_index_list[int(self.num_syn_images * train_ratio) :]
        # Create the training and validation datasets
        train_data = copy.deepcopy(self)
        val_data = copy.deepcopy(self)
        train_data.real_image_name_list = train_real_image_names
        val_data.real_image_name_list = val_real_image_names
        train_data.syn_image_index_list = train_syn_image_indices
        val_data.syn_image_index_list = val_syn_image_indices
        train_data.num_real_images = len(train_real_image_names)
        val_data.num_real_images = len(val_real_image_names)
        train_data.num_syn_images = len(train_syn_image_indices)
        val_data.num_syn_images = len(val_syn_image_indices)

        return train_data, val_data

    def _find_images(self) -> tuple[int, list[str]]:
        """Discovers all valid image files in the real images directory.

        Scans the data folder for files with common image extensions (.png, .jpg,
        .jpeg, .gif) and returns their count and filenames. If no data folder is
        specified (None), returns empty results.

        Returns:
            A tuple of (num_images, image_names) where num_images is the count of
            found images and image_names is a list of filenames (not full paths).
            Returns (0, []) if data_folder_name is None.

        Note:
            Only the filename is stored, not the full path. The full path is
            constructed at load time by joining with data_folder_name.
        """
        if self.data_folder_name is None:
            return 0, []
        # Get the image names
        image_name_list = [
            x
            for x in os.listdir(self.data_folder_name)
            if x.endswith((".png", ".jpg", ".jpeg", ".gif"))
        ]
        # Return the number of images and their names
        return len(image_name_list), image_name_list

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
        self.real_image_name_list = np.random.choice(
            self.real_image_name_list, self.num_syn_images, replace=True
        )
        self.num_real_images = len(self.real_image_name_list)

    def cache_images(self, num_images_to_cache: int):
        """Preloads images into memory for faster iteration during training.

        Loads the first N images (in their raw, unaugmented form) into a cache
        stored as a nested tensor. Cached images are loaded once and reused across
        epochs, which significantly reduces I/O overhead at the cost of memory usage.

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
        self.image_cache = []
        self.num_cached_images = min(num_images_to_cache, len(self))
        # Cache the images
        for index in range(self.num_cached_images):
            image = self._load_image(index)
            # Store the image
            self.image_cache.append(image)
        # Convert the image cache to a tensor
        self.image_cache = torch.nested.nested_tensor(self.image_cache)


class FinetuneData(Dataset):
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
        bcf_store_file: Path to the BCF store file containing labeled synthetic images.
        label_file: Path to the binary label file (uint32 format).
        aug_prob: Probability of applying each augmentation (0.0 to 1.0).
        image_normalization: Normalization scheme ("0to1" or "-1to1").
        bcf_store: BCFStoreFile instance for reading images.
        labels: NumPy array of integer class labels.
        num_images: Total count of labeled images.
        image_cache: List of cached image tensors.
        num_cached_images: Number of images currently cached.
    """

    def __init__(
        self,
        bcf_store_file: str,
        label_file: str,
        aug_prob: float,
        image_normalization: str = "0to1",
    ):
        """Initializes the FinetuneData dataset with labeled synthetic images.

        Loads images from a BCF store and their corresponding labels from a binary
        label file. Validates that the number of images and labels match to ensure
        data consistency.

        Args:
            bcf_store_file: The path to the BCF store file containing synthetic
                images. Must be in valid BCF format.
            label_file: The path to the binary label file containing uint32 class
                labels. Must have exactly one label per image in the BCF store.
            aug_prob: The probability (0.0 to 1.0) of applying each augmentation
                in the synthetic pipeline. Higher values increase augmentation
                strength. Typical values: 0.3-0.8 for training, 0.0 for validation.
            image_normalization: The normalization scheme for pixel values.
                Either "0to1" (scales to [0, 1]) or "-1to1" (scales to [-1, 1]).
                Default is "0to1".

        Raises:
            ValueError: If image_normalization is not "0to1" or "-1to1", or if
                the number of images and labels don't match.
            FileNotFoundError: If bcf_store_file or label_file doesn't exist.
            IOError: If the files cannot be read.
        """
        # Store the parameters
        self.bcf_store_file = bcf_store_file
        self.label_file = label_file
        self.aug_prob = aug_prob
        self.image_normalization = image_normalization
        # Check the feature normalization type
        if self.image_normalization not in ["0to1", "-1to1"]:
            raise ValueError("The image normalization type must be either '0to1' or '-1to1'.")
        # Load the BCF store file
        self.bcf_store = BCFStoreFile(bcf_store_file)
        # Load the labels
        self.labels = read_label(label_file)
        # Find the number of images
        self.num_images = self.bcf_store.size()
        # Create the image index list
        self.image_index_list = np.arange(self.num_images)
        # Check that the images and labels are the same size
        if self.num_images != len(self.labels):
            raise ValueError("The number of images and labels must be the same.")
        # Create the image cache
        self.image_cache = []
        self.num_cached_images = 0

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
        # Get the image
        image = self._get_image(index)
        # Normalize the image
        if self.image_normalization == "0to1":
            image = image / 255.0
        elif self.image_normalization == "-1to1":
            image = (image / 127.5) - 1.0
        # Get the label
        label = int(self.labels[index])
        label = torch.tensor(label).long()

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
        # Check if the image is cached
        if self.num_cached_images > 0 and index < self.num_cached_images:
            image = self.image_cache[index]
        else:
            image = self._load_image(index)
        # Apply the augmentation pipeline
        image = image.numpy()
        image = augmentation_pipeline(image[0], "synthetic", self.aug_prob)
        image = torch.from_numpy(image).float().unsqueeze(0)

        return image

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

    def cache_images(self, num_images_to_cache: int):
        """Preloads images into memory for faster training iteration.

        Loads the first N images (in their raw, unaugmented form) into a cache
        stored as a nested tensor. This significantly reduces I/O overhead during
        training at the cost of memory usage. Augmentations are still applied
        on-the-fly to cached images.

        Args:
            num_images_to_cache: The maximum number of images to cache. If this
                exceeds the dataset size, all images will be cached. Memory usage
                scales linearly with this value.

        Note:
            Cached images are stored in raw form before augmentation, so each
            access still produces a different augmented version. Images are cached
            starting from index 0.

        Warning:
            Must be called after split_data_random(), not before. The cache must
            be empty before splitting the dataset.
        """
        self.image_cache = []
        self.num_cached_images = min(num_images_to_cache, len(self))
        # Cache the images
        for index in range(self.num_cached_images):
            image = self._load_image(index)
            # Store the image
            self.image_cache.append(image)
        # Convert the image cache to a tensor
        self.image_cache = torch.nested.nested_tensor(self.image_cache)


class EvalData(Dataset):
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
        bcf_store_file: Path to the BCF store file containing test images.
        label_file: Path to the binary label file (uint32 format).
        image_normalization: Normalization scheme ("0to1" or "-1to1").
        num_image_crops: Number of augmented crops to generate per image.
        bcf_store: BCFStoreFile instance for reading images.
        labels: NumPy array of integer class labels.
        num_images: Total count of test images.
    """

    def __init__(
        self,
        bcf_store_file: str,
        label_file: str,
        image_normalization: str = "0to1",
        num_image_crops: int = 15,
    ):
        """Initializes the EvalData dataset for test-time augmentation.

        Loads images and labels for evaluation, configuring the test-time augmentation
        parameters. Validates that images and labels are properly paired.

        Args:
            bcf_store_file: The path to the BCF store file containing test images.
                Must be in valid BCF format.
            label_file: The path to the binary label file containing uint32 class
                labels. Must have exactly one label per image.
            image_normalization: The normalization scheme for pixel values.
                Either "0to1" (scales to [0, 1]) or "-1to1" (scales to [-1, 1]).
                Default is "0to1".
            num_image_crops: The number of augmented crops to generate per image
                for test-time augmentation. More crops improve accuracy but increase
                computation time. Typical values: 10-20. Default is 15.

        Raises:
            ValueError: If image_normalization is not "0to1" or "-1to1", or if
                the number of images and labels don't match.
            FileNotFoundError: If bcf_store_file or label_file doesn't exist.
            IOError: If the files cannot be read.
        """
        # Store the parameters
        self.bcf_store_file = bcf_store_file
        self.label_file = label_file
        self.image_normalization = image_normalization
        self.num_image_crops = num_image_crops
        # Check the feature normalization type
        if self.image_normalization not in ["0to1", "-1to1"]:
            raise ValueError("The image normalization type must be either '0to1' or '-1to1'.")
        # Load the BCF store file
        self.bcf_store = BCFStoreFile(bcf_store_file)
        # Load the labels
        self.labels = read_label(label_file)
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
            >>> dataset = EvalData('test.bcf', 'test.labels', num_image_crops=10)
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
        image_crops = eval_pipeline(image, self.num_image_crops)
        # Convert the image to a tensor
        image = torch.tensor(image_crops).float().unsqueeze(1)
        # Normalize the image
        if self.image_normalization == "0to1":
            image = image / 255.0
        elif self.image_normalization == "-1to1":
            image = (image / 127.5) - 1.0
        # Get the label
        label = int(self.labels[index])
        label = torch.tensor(label).long()

        return image, label
