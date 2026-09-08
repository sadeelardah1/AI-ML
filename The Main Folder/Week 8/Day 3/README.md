# Day 3 — Computer Vision Preprocessing with OpenCV

## Overview

Day 3 focuses on preparing raw images before they are used by a computer vision model.

The main idea is simple:

```text
Raw Image
   ↓
Read with OpenCV
   ↓
Resize
   ↓
BGR → RGB
   ↓
Normalize
   ↓
Augment Training Images
   ↓
Match the Pre-trained Model
   ↓
Model-Ready Input
```

For the practical work, a **garbage classification image dataset** is used so that each preprocessing step can be applied to a realistic computer vision problem.

---

## Main Concepts

### 1. Why Image Preprocessing Is Needed

Real-world images may have different:

- Widths and heights
- Color formats
- Pixel ranges
- Lighting conditions
- Orientations and scales

Preprocessing makes the input more consistent before it reaches the model.

---

### 2. OpenCV Fundamentals

OpenCV is used to perform the main image-processing operations in the notebook:

```python
import cv2
```

Important operations include:

```python
cv2.imread()
cv2.resize()
cv2.cvtColor()
cv2.Canny()
```

---

### 3. Reading and Resizing Images

Images are loaded with OpenCV:

```python
image = cv2.imread(path)
```

They are then resized to a fixed input size:

```python
image = cv2.resize(image, (224, 224))
```

This ensures that every image has the same spatial dimensions.

---

### 4. BGR vs. RGB

OpenCV reads color images using **BGR** channel order.

Most visualization and deep-learning workflows use **RGB**.

```python
image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)
```

Without this conversion, image colors can appear incorrect.

---

### 5. Normalization

Standard 8-bit image pixels usually range from:

```text
0 → 255
```

A common generic normalization converts them to:

```text
0 → 1
```

```python
image = image.astype("float32") / 255.0
```

---

### 6. Edge Detection

The notebook also demonstrates classical edge detection with OpenCV:

```python
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
```

Canny edge detection highlights strong visual boundaries in an image.

---

## Reusable Preprocessing Pipeline

The main preprocessing function combines the core Day 3 steps:

```python
def preprocess_image(path, target_size=(224, 224), normalize=True):
    image_bgr = cv2.imread(str(path))

    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")

    image_bgr = cv2.resize(image_bgr, target_size)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype("float32")

    if normalize:
        image_rgb = image_rgb / 255.0

    return image_rgb
```

Pipeline:

```text
Image Path
   ↓
Read
   ↓
Resize to 224 × 224
   ↓
BGR → RGB
   ↓
Convert to float32
   ↓
Normalize
   ↓
Processed Image
```

---

## Data Augmentation

Data augmentation creates realistic variations of training images.

Examples used in Day 3 include:

- Rotation
- Zoom
- Horizontal flip
- Brightness changes

Example configuration:

```python
ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    rescale=1.0 / 255.0,
)
```

Augmentation helps reduce memorization and can improve model generalization.

> Augmentation should normally be applied to the **training set only**, not validation or test data.

---

## Transfer Learning Preprocessing

A pre-trained model expects images to be processed in the same way used during its original training.

For example, **MobileNetV2** uses its own preprocessing function:

```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

image = preprocess_input(image)
```

Its expected pixel range is approximately:

```text
-1 → 1
```

This is different from generic `0 → 1` normalization.

### Important Rule

Do not apply generic normalization and then apply model-specific preprocessing unless the model documentation explicitly requires both.

---

## Garbage Classification Application

The notebook applies the Day 3 pipeline to a garbage image dataset.

The workflow is:

```text
Garbage Dataset
      ↓
Explore Classes
      ↓
Read Images
      ↓
Resize
      ↓
Correct Color Format
      ↓
Normalize
      ↓
Visualize Augmentation
      ↓
Prepare for Transfer Learning
```

This makes the notebook a practical example of preparing real-world image data for a computer vision classification system.

---


## Key Takeaway

> **Good computer vision preprocessing makes images consistent, preserves useful visual information, and matches the input expectations of the model.**

---

## Files

```text
Day3.ipynb
garbage_classification/
README.md
```