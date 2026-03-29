"""Tests for deepfont.data.config Pydantic configuration classes.

These tests lock the default values and validate the constraints of all
three dataset configuration classes so that accidental changes produce
immediate, descriptive failures.

Test classes:
    TestPretrainDataConfigDefaults     -- pins every field default
    TestPretrainDataConfigValidation   -- field constraints and immutability
    TestFinetuneDataConfigDefaults     -- pins every field default
    TestFinetuneDataConfigValidation   -- field constraints and immutability
    TestEvalDataConfigDefaults         -- pins every field default
    TestEvalDataConfigValidation       -- field constraints and immutability
"""

import pytest
from pydantic import ValidationError

from deepfont.data.config import EvalDataConfig, FinetuneDataConfig, PretrainDataConfig


class TestPretrainDataConfigDefaults:
    """Pin default values for all PretrainDataConfig fields."""

    def test_synthetic_bcf_file_default(self):
        assert PretrainDataConfig().synthetic_bcf_file == ""

    def test_real_image_dir_default(self):
        assert PretrainDataConfig().real_image_dir is None

    def test_aug_prob_default(self):
        assert PretrainDataConfig().aug_prob == 0.5

    def test_image_normalization_default(self):
        assert PretrainDataConfig().image_normalization == "0to1"


class TestPretrainDataConfigValidation:
    """Field constraints and immutability for PretrainDataConfig."""

    def test_config_is_frozen(self):
        config = PretrainDataConfig()
        with pytest.raises(ValidationError):
            config.aug_prob = 0.9

    def test_custom_synthetic_bcf_file(self):
        assert PretrainDataConfig(synthetic_bcf_file="train.bcf").synthetic_bcf_file == "train.bcf"

    def test_custom_real_image_dir(self):
        assert PretrainDataConfig(real_image_dir="data/real").real_image_dir == "data/real"

    def test_custom_aug_prob(self):
        assert PretrainDataConfig(aug_prob=0.8).aug_prob == pytest.approx(0.8)

    def test_custom_image_normalization(self):
        assert PretrainDataConfig(image_normalization="-1to1").image_normalization == "-1to1"

    def test_aug_prob_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="aug_prob"):
            PretrainDataConfig(aug_prob=-0.1)

    def test_aug_prob_above_one_rejected(self):
        with pytest.raises(ValidationError, match="aug_prob"):
            PretrainDataConfig(aug_prob=1.1)

    def test_aug_prob_zero_accepted(self):
        assert PretrainDataConfig(aug_prob=0.0).aug_prob == pytest.approx(0.0)

    def test_aug_prob_one_accepted(self):
        assert PretrainDataConfig(aug_prob=1.0).aug_prob == pytest.approx(1.0)

    def test_invalid_image_normalization_rejected(self):
        with pytest.raises(ValidationError):
            PretrainDataConfig(image_normalization="minmax")


class TestFinetuneDataConfigDefaults:
    """Pin default values for all FinetuneDataConfig fields."""

    def test_synthetic_bcf_file_default(self):
        assert FinetuneDataConfig().synthetic_bcf_file == ""

    def test_label_file_default(self):
        assert FinetuneDataConfig().label_file == ""

    def test_aug_prob_default(self):
        assert FinetuneDataConfig().aug_prob == 0.5

    def test_image_normalization_default(self):
        assert FinetuneDataConfig().image_normalization == "0to1"


class TestFinetuneDataConfigValidation:
    """Field constraints and immutability for FinetuneDataConfig."""

    def test_config_is_frozen(self):
        config = FinetuneDataConfig()
        with pytest.raises(ValidationError):
            config.aug_prob = 0.9

    def test_custom_synthetic_bcf_file(self):
        assert FinetuneDataConfig(synthetic_bcf_file="ft.bcf").synthetic_bcf_file == "ft.bcf"

    def test_custom_label_file(self):
        assert FinetuneDataConfig(label_file="ft.labels").label_file == "ft.labels"

    def test_custom_aug_prob(self):
        assert FinetuneDataConfig(aug_prob=0.3).aug_prob == pytest.approx(0.3)

    def test_custom_image_normalization(self):
        assert FinetuneDataConfig(image_normalization="-1to1").image_normalization == "-1to1"

    def test_aug_prob_below_zero_rejected(self):
        with pytest.raises(ValidationError, match="aug_prob"):
            FinetuneDataConfig(aug_prob=-0.1)

    def test_aug_prob_above_one_rejected(self):
        with pytest.raises(ValidationError, match="aug_prob"):
            FinetuneDataConfig(aug_prob=1.1)

    def test_invalid_image_normalization_rejected(self):
        with pytest.raises(ValidationError):
            FinetuneDataConfig(image_normalization="standardize")


class TestEvalDataConfigDefaults:
    """Pin default values for all EvalDataConfig fields."""

    def test_synthetic_bcf_file_default(self):
        assert EvalDataConfig().synthetic_bcf_file == ""

    def test_label_file_default(self):
        assert EvalDataConfig().label_file == ""

    def test_image_normalization_default(self):
        assert EvalDataConfig().image_normalization == "0to1"

    def test_num_image_crops_default(self):
        assert EvalDataConfig().num_image_crops == 15


class TestEvalDataConfigValidation:
    """Field constraints and immutability for EvalDataConfig."""

    def test_config_is_frozen(self):
        config = EvalDataConfig()
        with pytest.raises(ValidationError):
            config.num_image_crops = 10

    def test_custom_synthetic_bcf_file(self):
        assert EvalDataConfig(synthetic_bcf_file="test.bcf").synthetic_bcf_file == "test.bcf"

    def test_custom_label_file(self):
        assert EvalDataConfig(label_file="test.labels").label_file == "test.labels"

    def test_custom_image_normalization(self):
        assert EvalDataConfig(image_normalization="-1to1").image_normalization == "-1to1"

    def test_custom_num_image_crops(self):
        assert EvalDataConfig(num_image_crops=20).num_image_crops == 20

    def test_num_image_crops_zero_rejected(self):
        with pytest.raises(ValidationError, match="num_image_crops"):
            EvalDataConfig(num_image_crops=0)

    def test_num_image_crops_negative_rejected(self):
        with pytest.raises(ValidationError, match="num_image_crops"):
            EvalDataConfig(num_image_crops=-1)

    def test_num_image_crops_one_accepted(self):
        assert EvalDataConfig(num_image_crops=1).num_image_crops == 1

    def test_invalid_image_normalization_rejected(self):
        with pytest.raises(ValidationError):
            EvalDataConfig(image_normalization="zscore")
