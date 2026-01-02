"""Smoke tests for the deepfont package.

These tests verify that the package and its core modules can be imported
and that basic functionality is available without runtime errors.
"""

import torch


class TestImports:
    """Test that all core modules can be imported."""

    def test_import_base_package(self):
        """Test that the base deepfont package can be imported."""
        import deepfont

        assert deepfont is not None

    def test_import_models_module(self):
        """Test that the models module can be imported."""
        from deepfont import models

        assert models is not None

    def test_import_data_module(self):
        """Test that the data module can be imported."""
        from deepfont import data

        assert data is not None

    def test_import_deepfont_model(self):
        """Test that the DeepFont model can be imported."""
        from deepfont.models.deepfont import DeepFontAE

        assert DeepFontAE is not None

    def test_import_datasets(self):
        """Test that the datasets module can be imported."""
        from deepfont.data import datasets

        assert datasets is not None

    def test_import_augmentations(self):
        """Test that the augmentations module can be imported."""
        from deepfont.data import augmentations

        assert augmentations is not None

    def test_import_bcf_subpackage(self):
        """Test that the BCF subpackage can be imported."""
        from deepfont.data import bcf

        assert bcf is not None


class TestModelInstantiation:
    """Test that core models can be instantiated."""

    def test_deepfont_ae_instantiation(self):
        """Test that DeepFontAE can be instantiated with default parameters."""
        from deepfont.models.deepfont import DeepFontAE

        model = DeepFontAE()
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_deepfont_ae_with_sigmoid(self):
        """Test that DeepFontAE can be instantiated with sigmoid activation."""
        from deepfont.models.deepfont import DeepFontAE

        model = DeepFontAE(output_activation="sigmoid")
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_deepfont_ae_with_relu(self):
        """Test that DeepFontAE can be instantiated with ReLU activation."""
        from deepfont.models.deepfont import DeepFontAE

        model = DeepFontAE(output_activation="relu")
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_deepfont_ae_has_encoder(self):
        """Test that DeepFontAE has an encoder attribute."""
        from deepfont.models.deepfont import DeepFontAE

        model = DeepFontAE()
        assert hasattr(model, "encoder")
        assert model.encoder is not None

    def test_deepfont_ae_has_decoder(self):
        """Test that DeepFontAE has a decoder attribute."""
        from deepfont.models.deepfont import DeepFontAE

        model = DeepFontAE()
        assert hasattr(model, "decoder")
        assert model.decoder is not None


class TestModelForward:
    """Test that models can perform forward passes."""

    def test_deepfont_ae_forward_pass(self):
        """Test that DeepFontAE can perform a forward pass with dummy input."""
        from deepfont.models.deepfont import DeepFontAE

        model = DeepFontAE()
        model.eval()

        # Create a dummy input (batch_size=2, channels=1, height=105, width=105)
        dummy_input = torch.randn(2, 1, 105, 105)

        with torch.no_grad():
            output = model(dummy_input)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape == dummy_input.shape  # Autoencoder should preserve shape

    def test_deepfont_ae_output_types(self):
        """Test that DeepFontAE output is a valid tensor."""
        from deepfont.models.deepfont import DeepFontAE

        model = DeepFontAE()
        model.eval()

        dummy_input = torch.randn(1, 1, 105, 105)

        with torch.no_grad():
            output = model(dummy_input)

        assert torch.is_tensor(output)
        assert output.dtype == torch.float32
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestAugmentations:
    """Test that augmentation functions are available."""

    def test_augmentation_pipeline_exists(self):
        """Test that augmentation_pipeline function exists."""
        from deepfont.data.augmentations import augmentation_pipeline

        assert augmentation_pipeline is not None
        assert callable(augmentation_pipeline)

    def test_eval_pipeline_exists(self):
        """Test that eval_pipeline function exists."""
        from deepfont.data.augmentations import eval_pipeline

        assert eval_pipeline is not None
        assert callable(eval_pipeline)

    def test_image_size_constant(self):
        """Test that IMAGE_SIZE constant is defined."""
        from deepfont.data.augmentations import IMAGE_SIZE

        assert IMAGE_SIZE is not None
        assert isinstance(IMAGE_SIZE, int)
        assert IMAGE_SIZE > 0


class TestBCFModules:
    """Test that BCF-related modules are available."""

    def test_bcf_store_file_import(self):
        """Test that BCFStoreFile class can be imported."""
        from deepfont.data.bcf import BCFStoreFile

        assert BCFStoreFile is not None

    def test_read_label_import(self):
        """Test that read_label function can be imported."""
        from deepfont.data.bcf import read_label

        assert read_label is not None
        assert callable(read_label)
