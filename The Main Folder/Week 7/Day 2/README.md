# Week 7 — Day 2: Building CNNs and Transfer Learning

This repository/notebook covers Day 2 of Week 7 of the BinXTech AI & Machine Learning Internship: building a complete convolutional neural network, reducing overfitting with data augmentation, and applying transfer learning with a pre-trained model.

## Day 2 Learning Objectives

By the end of the notebook, you should be able to:

- Build a full CNN using convolution, pooling, and dense/classification layers.
- Explain why max pooling is used to shrink feature maps.
- Apply image data augmentation to improve generalization.
- Use MobileNetV2 as a pre-trained feature extractor.
- Freeze a pre-trained backbone and train a new classifier head.
- Fine-tune part of a pre-trained model using a small learning rate.
- Compare scratch training, augmentation, and transfer learning using validation/test metrics and training time.

## Dataset

The supplied dataset is a binary image-classification dataset with two classes:

| Split | Benign | Malignant | Total |
| --- | ---: | ---: | ---: |
| Train | 6,289 | 5,590 | 11,879 |
| Test | 1,000 | 1,000 | 2,000 |
| Total | 7,289 | 6,590 | 13,879 |

The images are RGB and 224 × 224 pixels. The notebook uses 160 × 160 by default for faster training; this can be changed through `IMG_SIZE`.

Expected extracted structure:

```text
dataset/
├── train/
│   ├── Benign/
│   └── Malignant/
└── test/
    ├── Benign/
    └── Malignant/
```

The notebook uses the project folder directly:

```text
Day 2/
├── dataset/
│   ├── train/
│   │   ├── Benign/
│   │   └── Malignant/
│   └── test/
│       ├── Benign/
│       └── Malignant/
├── Day2.ipynb
└── README.md
```

`Day2.ipynb` uses `Path("dataset")`, so no ZIP extraction or manual path configuration is needed when the project keeps this structure.

## Notebook Structure

1. Setup and GPU check
2. Day 2 roadmap and dataset structure
3. Pooling
4. Full CNN architecture
5. Data augmentation
6. Transfer learning
7. Dataset loading and visualization
8. CNN from scratch
9. CNN with data augmentation
10. Frozen MobileNetV2 transfer learning
11. Fine-tuning
12. Test evaluation and model comparison
13. Common mistakes and best practices
14. Final reflection and summary

## Main Models

### Model A — CNN From Scratch

A compact convolutional model trained from random initialization. This establishes the baseline for Day 2.

### Model B — CNN + Data Augmentation

The same general CNN workflow with random flip, rotation, zoom, and contrast augmentation applied to training images.

### Model C — MobileNetV2 Transfer Learning

A MobileNetV2 backbone pre-trained on ImageNet is used as a frozen feature extractor. A new binary classification head is trained on the supplied dataset, followed by an optional fine-tuning stage for the upper backbone layers.

## Recommended Environment

Google Colab with a GPU is recommended.

Core libraries:

```text
Python 3
TensorFlow / Keras
NumPy
Pandas
Matplotlib
scikit-learn
```

In Google Colab, most of these packages are already available. If TensorFlow is missing in another environment, install the dependencies before running the notebook.

## Running in Google Colab

1. Open the notebook in Google Colab.
2. Select `Runtime -> Change runtime type -> GPU` if a GPU is available.
3. Make sure the folder `dataset/` is beside `Day2.ipynb` and contains `train/` and `test/`.
4. Run the notebook from top to bottom.
5. Check the printed `Dataset root` and `Class names` before training.
6. Review the validation curves and final comparison table.
7. Complete the Final Reflection section using your actual results.

## Default Training Configuration

```python
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
EPOCHS = 5
SEED = 42
```

Increase the number of epochs only if the validation curves show that the models are still improving. Early stopping is included to restore the best validation-loss weights.

## Evaluation

The notebook reports:

- Test loss
- Test accuracy
- AUC
- Precision
- Recall
- Training time
- Classification report
- Confusion matrix

Because the supplied test split contains 1,000 images from each class, accuracy is easy to interpret, but precision, recall, AUC, and the confusion matrix should still be reviewed to understand the error pattern.

## Reproducibility Notes

- A fixed random seed (`42`) is used.
- The training folder is split into training and validation subsets using the same seed.
- The test folder is kept separate from training and validation.
- Data augmentation is applied only to training images.
- MobileNetV2 uses its required `preprocess_input` preprocessing.
- Fine-tuning uses a much smaller learning rate than the frozen transfer-learning stage.

## Important Notes

- The first use of `MobileNetV2(weights="imagenet")` may download pre-trained ImageNet weights.
- Do not use the test set to decide architecture or hyperparameters; use validation results for model-development decisions.
- If fine-tuning worsens validation performance, keep the frozen transfer-learning result instead.
- For a repository, avoid committing generated checkpoints and temporary extracted folders unless they are intentionally part of the project.

## Reference

Additional deep-learning reading listed in the Week 7 resources:
