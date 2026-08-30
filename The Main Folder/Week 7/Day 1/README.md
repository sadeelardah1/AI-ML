# Week 7 — Day 1: Convolutional Neural Networks

## Overview

This repository contains the Day 1 work for Week 7 of the BinX Tech AI & Machine Learning Internship.

The notebook focuses on the foundations of Convolutional Neural Networks (CNNs) and introduces convolution through a simple edge-detection example applied to a real skin-lesion image.

The project uses a binary image-classification dataset with two classes:

- Benign
- Malignant

The main purpose of this notebook is to understand why CNNs are more suitable than fully connected networks for image data before building a complete CNN model on Day 2.

## Learning Objectives

By the end of the notebook, the following topics are covered:

- Sprint 2 planning and core-model direction
- Why dense networks become inefficient for image data
- Convolution and local feature extraction
- Filters / kernels
- Feature maps
- Stride and padding
- Parameter sharing
- Feature hierarchy in CNNs
- Applying a hand-defined vertical-edge filter to a real image
- Selecting CNN as the core architecture for this image-based project

## Dataset

The dataset contains skin-lesion images organized into training and testing folders.

```text
dataset/
├── train/
│   ├── Benign/
│   └── Malignant/
└── test/
    ├── Benign/
    └── Malignant/
```

Dataset summary used in the notebook:

| Split | Class | Images |
|---|---|---:|
| Train | Benign | 6,289 |
| Train | Malignant | 5,590 |
| Test | Benign | 1,000 |
| Test | Malignant | 1,000 |

Each image is:

```text
224 × 224 pixels
RGB — 3 channels
```

This means one image contains:

```text
224 × 224 × 3 = 150,528 input values
```

## Why CNNs for Images?

A dense layer connects every input value to every neuron in the next layer.

For an image with 150,528 input values and a dense layer containing 128 neurons:

```text
150,528 × 128 = 19,267,584 weights
```

This is a large number of parameters for only one layer.

CNNs solve this problem by applying small filters locally across the image instead of learning separate weights for every pixel position.

## Convolution Demo

The notebook implements a simple 2D convolution manually using NumPy.

A 3 × 3 vertical-edge filter is used:

```text
-1   0   1
-1   0   1
-1   0   1
```

The filter is applied to a real image from the training dataset.

The original image is converted to grayscale and passed through the filter to produce a feature map.

Input shape:

```text
(224, 224)
```

Feature-map shape:

```text
(222, 222)
```

The output is smaller because the demonstration uses valid convolution, where the filter is not allowed to move outside the image boundary.

## Parameter Sharing

The edge-detection filter contains only:

```text
3 × 3 = 9 weights
```

The same 9 values are reused across different spatial locations in the image.

For the simplified comparison used in the notebook:

```text
Dense-layer weights: 19,267,584
Filter weights: 9
Ratio: approximately 2,140,843 : 1
```

This comparison is used only to demonstrate the idea of parameter sharing. A complete CNN contains many filters and therefore more than 9 trainable parameters.

## Feature Hierarchy

CNNs learn features in stages.

| Layer Depth | Typical Features |
|---|---|
| Early layers | Edges and simple patterns |
| Middle layers | Textures and shapes |
| Deeper layers | More complex visual structures |

The Day 1 notebook uses a manually defined filter so the convolution process can be understood clearly before moving to a trainable CNN.

## Architecture Decision

The project data type is:

```text
Images
```

The selected core architecture is:

```text
Convolutional Neural Network (CNN)
```

CNNs are appropriate for this project because they preserve spatial relationships and learn local visual patterns efficiently.

## Project Files

A recommended repository structure is:

```text
Week-7-Day-1/
├── Day1.ipynb
├── README.md
└── dataset/
    ├── train/
    │   ├── Benign/
    │   └── Malignant/
    └── test/
        ├── Benign/
        └── Malignant/
```

If the dataset is too large for GitHub, keep it locally and exclude it using `.gitignore`.

Example:

```gitignore
dataset/
.ipynb_checkpoints/
__pycache__/
```

## Requirements

The notebook uses:

- Python 3
- NumPy
- Pandas
- Matplotlib
- Pillow
- Jupyter Notebook

Install the required packages with:

```bash
pip install numpy pandas matplotlib pillow jupyter
```

## Running the Notebook

1. Clone or download the repository.
2. Place the extracted dataset inside a folder named `dataset`.
3. Make sure `train` and `test` are directly inside the `dataset` folder.
4. Open `Day1.ipynb`.
5. Restart the kernel.
6. Run all cells from top to bottom.

The notebook uses a relative path:

```python
DATASET_ROOT = Path("dataset")
```

so it can run without a machine-specific absolute path when the repository structure is kept the same.

## Day 1 Summary

The notebook demonstrates that:

- Images contain a large number of spatially related values.
- Dense networks become inefficient when applied directly to image pixels.
- Convolution applies small filters to local regions of an image.
- A feature map shows where a filter responds strongly.
- Parameter sharing allows the same filter to be reused across the image.
- CNN is the selected architecture for the skin-lesion image-classification project.

## Next Step

Day 2 will move from manual convolution to a complete CNN built with Keras, including convolution layers, pooling, data augmentation, and transfer learning.
