# DeepFont-Torch

A PyTorch implementation of DeepFont for font recognition.

## Overview

DeepFont is a deep learning approach for recognizing fonts from images. This project reimplements the DeepFont architecture using PyTorch.

## Features

- PyTorch implementation of the DeepFont architecture
- Font recognition from text images
- Pre-trained model support
- Training and inference pipelines

## Installation

```bash
# Clone the repository
git clone https://github.com/michaelriedl/deepfont-torch.git
cd deepfont-torch

# Install dependencies using uv
uv sync
```

## Requirements

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer
- PyTorch
- Additional dependencies listed in `pyproject.toml`

## Usage

### Training

```python
# Example training code
```

### Inference

```python
# Example inference code
```

## Model Architecture

Description of the DeepFont architecture and implementation details.

## Dataset

Information about the dataset used for training and evaluation.

## Results

Performance metrics and comparison with the original paper.

## Project Structure

```
deepfont-torch/
├── pyproject.toml       # Project dependencies
├── README.md            # This file
└── LICENSE              # License file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms specified in the LICENSE file.

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{wang2015deepfont,
  title={Deepfont: Identify your font from an image},
  author={Wang, Zhangyang and Yang, Jianchao and Jin, Hailin and Shechtman, Eli and Agarwala, Aseem and Brandt, Jonathan and Huang, Thomas S},
  booktitle={Proceedings of the 23rd ACM international conference on Multimedia},
  pages={451--459},
  year={2015}
}

@software{riedl2025deepfont,
  title={DeepFont-Torch: A PyTorch Implementation of DeepFont},
  author={Riedl, Michael},
  year={2025},
  url={https://github.com/michaelriedl/deepfont-torch}
}
```

## References

- [Original DeepFont Paper](https://arxiv.org/abs/1507.03196)

## Acknowledgments

Based on the original DeepFont work by Wang et al.
