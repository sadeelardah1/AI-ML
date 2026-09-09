# Day 4 — Model Integration & Error Analysis

## Overview

Day 4 focuses on connecting preprocessing and the trained model into one consistent prediction pipeline, then analyzing where the model fails.

The main workflow is:

```text
Raw Garbage Image
        ↓
Shared Preprocessing
        ↓
Feature Preparation
        ↓
Trained Model
        ↓
Prediction
        ↓
Error Analysis
```

The project continues using the **Garbage Classification** dataset from Day 3.

---

## Main Concepts

### 1. Model Integration

Model integration means combining preprocessing and prediction into one complete workflow.

Instead of running separate notebook cells manually:

```text
Read Image
→ Resize
→ Convert Color
→ Normalize
→ Prepare Features
→ Predict
```

Day 4 builds one end-to-end pipeline that handles these steps consistently.

---

### 2. Reusing Day 3 Preprocessing

Day 3 and Day 4 use the same shared preprocessing logic.

The shared file:

```text
preprocessing.py
```

contains:

```python
def preprocess_image(path, target_size=(224, 224)):
    image_bgr = cv2.imread(str(path))

    if image_bgr is None:
        raise ValueError(f"Could not read image: {path}")

    image_bgr = cv2.resize(image_bgr, target_size)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype("float32") / 255.0

    return image_rgb
```

This gives one source of truth for preprocessing.

```text
Day 3
   ↓
preprocessing.py
   ↓
Day 4
```

---

## 3. Training / Serving Consistency

A model should receive the same preprocessing during prediction that it received during training.

Correct:

```text
Training:
Resize → RGB → Normalize → Model

Prediction:
Resize → RGB → Normalize → Model
```

Incorrect preprocessing can create **training / serving skew**.

### Golden Rule

> Prediction-time preprocessing must match training-time preprocessing.

---

## 4. End-to-End `predict()` Function

The integrated prediction function accepts a raw image and returns a model prediction.

Conceptually:

```python
def predict(raw_input):
    processed = preprocess_image(raw_input)
    prepared = prepare_features(processed)
    prediction = model.predict(prepared)

    return prediction
```

Pipeline:

```text
Raw Image
   ↓
preprocess_image()
   ↓
Feature Preparation
   ↓
model.predict()
   ↓
Predicted Garbage Class
```

---

## 5. Baseline Model

A lightweight baseline model is used so that the Day 4 integration and error-analysis workflow can run completely.

The notebook uses:

- Image color information
- Basic channel statistics
- Edge information
- Logistic Regression

The main goal is not to introduce a new model architecture.

The goal is to practice:

```text
Integration
+
Prediction
+
Error Analysis
```

---

## 6. Error Analysis

Accuracy alone does not explain where the model is failing.

Day 4 analyzes the model by asking:

- Which classes are confused most often?
- Which class pairs produce the most errors?
- Are some errors caused by the data?
- Are some errors caused by model weakness?

---

## 7. Confusion Matrix

The confusion matrix compares:

```text
Actual Class
vs.
Predicted Class
```

Example:

```text
Actual: Cardboard
Predicted: Paper
```

This helps identify repeated confusion between similar classes.

---

## 8. Most Common Error Type

Incorrect predictions are grouped by:

```text
Actual Class → Predicted Class
```

The notebook then identifies the most frequent error pair.

Example:

```text
cardboard → paper
```

This gives a clearer understanding of the model's main weakness.

---

## 9. Inspecting Misclassified Examples

The notebook displays actual images that the model classified incorrectly.

Each example is reviewed using:

```text
Actual Label
Predicted Label
Prediction Confidence
```

This step helps explain why the mistake happened.

---

## 10. Data Issue vs. Model Weakness

At least three misclassified examples are reviewed manually.

### Data Issue

Use this category when the image is:

- Ambiguous
- Poorly cropped
- Too dark
- Unusual
- Possibly mislabeled

### Model Weakness

Use this category when:

- The image is reasonably clear
- The label is correct
- The model still predicts the wrong class

Example:

```text
Actual: Cardboard
Predicted: Paper

Category: Model Weakness
Reason: The image is clear, but the two classes have similar visual texture.
```

---

## What I Learned Today

By the end of Day 4, I learned how to:

- Integrate preprocessing and prediction into one pipeline.
- Reuse Day 3 preprocessing through a shared module.
- Build an end-to-end `predict()` function.
- Maintain training / serving consistency.
- Explain training / serving skew.
- Evaluate a model beyond accuracy.
- Build and interpret a confusion matrix.
- Find the most common prediction error.
- Inspect misclassified images.
- Distinguish between data issues and model weaknesses.

---

## Key Takeaway

> **A professional machine-learning workflow does not stop at prediction. It keeps preprocessing consistent and studies the model's mistakes to understand what should be improved.**

---

## Project Structure

```text
Week 8/
│
├── preprocessing.py
│
├── Day 3/
│   ├── garbage_classification/
│   └── Day3.ipynb
│
└── Day 4/
    ├── Day4.ipynb
    └── README.md
```

The garbage dataset remains in **Day 3** and is reused by Day 4 instead of being duplicated.