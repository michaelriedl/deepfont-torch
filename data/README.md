# data/

This directory contains example datasets for local development and testing. It is not intended to be used in production.

## Structure

The data directory is organized into three subdirectories representing different stages of the training pipeline:

### pretraining/

Contains data for autoencoder pretraining:
- **VFR_syn_train/**: Synthetic training images
  - `train.bcf`: Binary Concatenated File containing synthetic font images
  - `train.label`: Binary label file with font class indices (uint32)
- **VFR_real_u/**: Real-world unlabeled images scraped from the web

### finetuning/

Contains data for supervised fine-tuning:
- **VFR_syn_val/**: Synthetic validation images
  - `val.bcf`: Binary Concatenated File containing synthetic font images
  - `val.label`: Binary label file with font class indices (uint32)

### testing/

Contains data for model evaluation:
- **VFR_real_test/**: Real-world test images with labels
  - `vfr_large.bcf`: Binary Concatenated File containing real font images
  - `vfr_large.label`: Binary label file with font class indices (uint32)

## Font List

The `fontlist.txt` file contains all 2,383 font names used in this dataset, with each font name on a separate line. The label indices in the `.label` files correspond to line numbers in this file (0-indexed).

## BCF Format

BCF (Binary Concatenated File) is a custom binary format for efficiently storing multiple images:
- Header: 8 bytes (uint64) - number of files
- Size array: 8 bytes per file (uint64) - individual file sizes
- Data: Concatenated file contents (typically PNG images)

This format allows for efficient random access to individual images without decompressing the entire dataset.