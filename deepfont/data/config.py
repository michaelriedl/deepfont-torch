"""Pydantic configuration classes for DeepFont dataset classes.

These configuration classes parameterize the PretrainData, FinetuneData,
and EvalData dataset classes, making the distinction between synthetic
(rendered) and real (photographed/scanned) data sources explicit in the
field names.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, BaseModel, ConfigDict


class PretrainDataConfig(BaseModel):
    """Configuration for the PretrainData dataset.

    Controls the data sources and augmentation settings for autoencoder
    pretraining, which mixes synthetic images from a BCF store with
    optional real images from a directory.

    Example:
        >>> from deepfont.data.config import PretrainDataConfig
        >>> config = PretrainDataConfig(
        ...     synthetic_bcf_file="data/train.bcf",
        ...     real_image_dir="data/real_images",
        ...     aug_prob=0.5,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    synthetic_bcf_file: str = Field(
        default="",
        description="Path to BCF store file containing synthetic font images.",
    )
    real_image_dir: str | None = Field(
        default=None,
        description=(
            "Directory containing real/scanned font images. Supports .png, "
            ".jpg, .jpeg, and .gif files. None means synthetic data only."
        ),
    )
    aug_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of applying each augmentation in the pipeline. "
            "Higher values result in more aggressive augmentation."
        ),
    )
    image_normalization: Literal["0to1", "-1to1"] = Field(
        default="0to1",
        description=(
            'Normalization scheme for pixel values. "0to1" scales to '
            '[0, 1], "-1to1" scales to [-1, 1].'
        ),
    )
    manifest_file: str | None = Field(
        default=None,
        description=(
            "Path to a parquet manifest file. When set, image enumeration and label "
            "loading are read from the manifest instead of scanning the filesystem. "
            "All paths in the manifest are relative to the manifest file's directory."
        ),
    )


class FinetuneDataConfig(BaseModel):
    """Configuration for the FinetuneData dataset.

    Controls the data source and augmentation settings for supervised
    font classification fine-tuning using labeled synthetic images.

    Example:
        >>> from deepfont.data.config import FinetuneDataConfig
        >>> config = FinetuneDataConfig(
        ...     synthetic_bcf_file="data/finetune.bcf",
        ...     label_file="data/finetune.labels",
        ... )
    """

    model_config = ConfigDict(frozen=True)

    synthetic_bcf_file: str = Field(
        default="",
        description="Path to BCF store file containing labeled synthetic images.",
    )
    label_file: str = Field(
        default="",
        description="Path to binary label file (uint32 format, one label per image).",
    )
    aug_prob: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Probability of applying each augmentation in the pipeline. "
            "Higher values result in more aggressive augmentation."
        ),
    )
    image_normalization: Literal["0to1", "-1to1"] = Field(
        default="0to1",
        description=(
            'Normalization scheme for pixel values. "0to1" scales to '
            '[0, 1], "-1to1" scales to [-1, 1].'
        ),
    )
    manifest_file: str | None = Field(
        default=None,
        description=(
            "Path to a parquet manifest file. When set, image enumeration and label "
            "loading are read from the manifest instead of scanning the filesystem. "
            "All paths in the manifest are relative to the manifest file's directory."
        ),
    )


class EvalDataConfig(BaseModel):
    """Configuration for the EvalData dataset.

    Controls the data source and test-time augmentation settings for
    model evaluation with multiple augmented crops per image.

    Example:
        >>> from deepfont.data.config import EvalDataConfig
        >>> config = EvalDataConfig(
        ...     synthetic_bcf_file="data/test.bcf",
        ...     label_file="data/test.labels",
        ...     num_image_crops=15,
        ... )
    """

    model_config = ConfigDict(frozen=True)

    synthetic_bcf_file: str = Field(
        default="",
        description="Path to BCF store file containing test images.",
    )
    label_file: str = Field(
        default="",
        description="Path to binary label file (uint32 format, one label per image).",
    )
    image_normalization: Literal["0to1", "-1to1"] = Field(
        default="0to1",
        description=(
            'Normalization scheme for pixel values. "0to1" scales to '
            '[0, 1], "-1to1" scales to [-1, 1].'
        ),
    )
    num_image_crops: int = Field(
        default=15,
        ge=1,
        description=(
            "Number of augmented crops to generate per image for test-time "
            "augmentation. More crops improve accuracy but increase "
            "computation time."
        ),
    )
    manifest_file: str | None = Field(
        default=None,
        description=(
            "Path to a parquet manifest file. When set, label loading is read from "
            "the manifest instead of scanning the filesystem. All paths in the manifest "
            "are relative to the manifest file's directory."
        ),
    )
