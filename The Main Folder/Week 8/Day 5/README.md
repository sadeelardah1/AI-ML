# Day 5 — Full Evaluation & Explainability

## Overview

Day 5 focuses on evaluating an imbalanced classification model correctly, improving minority-class learning, tuning the decision threshold, and explaining model predictions.

The practical project uses a **Credit Card Fraud Detection** dataset.

The main workflow is:

```text
Credit Card Transactions
        ↓
Train / Validation / Test Split
        ↓
Baseline Model
        ↓
Full Evaluation
        ↓
Handle Class Imbalance with SMOTE
        ↓
Compare Results
        ↓
Precision–Recall Trade-off
        ↓
Threshold Tuning
        ↓
Final Test Evaluation
        ↓
SHAP Explainability
```

---

## Dataset

The dataset file is:

```text
creditcard.csv
```

The target column is:

```text
Class
```

where:

```text
Class = 0 → Normal Transaction
Class = 1 → Fraud Transaction
```

The dataset is highly imbalanced because fraud transactions are much rarer than normal transactions.

---

## 1. Why Accuracy Alone Is Not Enough

In an imbalanced dataset, a model can achieve very high accuracy by predicting the majority class most of the time.

For fraud detection, this can be misleading.

Example:

```text
Most transactions are Normal

Model predicts:
Normal
Normal
Normal
Normal
...

Accuracy → High
Fraud Recall → Very Low
```

Because of this, Day 5 uses several classification metrics instead of relying only on accuracy.

---

## 2. Evaluation Metrics

### Precision

Precision answers:

> When the model predicts Fraud, how often is that prediction correct?

```text
Precision =
Correct Fraud Predictions
-------------------------
All Predicted Fraud Cases
```

High precision means fewer false fraud alarms.

---

### Recall

Recall answers:

> Of all real fraud transactions, how many did the model detect?

```text
Recall =
Detected Fraud Cases
--------------------
All Real Fraud Cases
```

High recall means fewer missed fraud transactions.

---

### F1-Score

F1-score combines precision and recall into one metric.

It is useful when both false positives and false negatives matter.

```text
Precision + Recall
        ↓
     F1-Score
```

---

### ROC-AUC

ROC-AUC measures how well the model separates positive and negative classes across many decision thresholds.

A higher ROC-AUC generally means better ranking ability.

---

### PR-AUC

PR-AUC summarizes the precision-recall behavior of the model.

It is especially useful when the positive class is rare, such as fraud detection.

---

## 3. Train / Validation / Test Split

The data is divided into three parts:

```text
Training Set
    ↓
Train the model
Apply SMOTE

Validation Set
    ↓
Compare models
Tune the threshold

Test Set
    ↓
Final evaluation
```

This is important because threshold tuning should not be performed directly on the final test set.

---

## 4. Baseline Model

A simple **Logistic Regression** model is used as the baseline.

The purpose of the baseline is to create a reference point before handling class imbalance.

```text
Training Data
     ↓
StandardScaler
     ↓
Logistic Regression
     ↓
Fraud Probability
```

The model is then evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

---

## 5. Class Imbalance

Fraud detection contains two classes with very different sizes:

```text
Normal Transactions → Majority Class
Fraud Transactions  → Minority Class
```

This imbalance can make the model biased toward the majority class.

---

## 6. SMOTE

**SMOTE — Synthetic Minority Over-sampling Technique** creates synthetic examples of the minority class.

The workflow is:

```text
Imbalanced Training Data
        ↓
SMOTE
        ↓
More Balanced Training Data
        ↓
Model Training
```

### Important Rule

SMOTE must be applied to the **training data only**.

```text
Training Data   → SMOTE ✅
Validation Data → No SMOTE
Test Data       → No SMOTE
```

Applying SMOTE before splitting the data can create **data leakage** and make the final evaluation unreliable.

---

## 7. Baseline vs. SMOTE

After training the SMOTE model, its performance is compared with the baseline model.

The comparison focuses on:

```text
Precision
Recall
F1
ROC-AUC
PR-AUC
```

SMOTE may increase recall because the model sees more minority-class examples during training.

However, higher recall can sometimes reduce precision.

This leads to an important Day 5 concept:

## Precision–Recall Trade-off

```text
Higher Recall
     ↕
Possible Lower Precision
```

The best balance depends on the application.

---

## 8. Decision Threshold

A binary classifier usually produces a probability.

Example:

```text
Fraud Probability = 0.82
```

A threshold converts that probability into a class.

```text
Threshold = 0.50

0.82 ≥ 0.50
→ Fraud
```

Another example:

```text
Fraud Probability = 0.42

0.42 < 0.50
→ Normal
```

---

## 9. Threshold Tuning

The default threshold is usually:

```text
0.50
```

But this is not always the best threshold for the task.

Lowering the threshold usually:

```text
↑ Recall
↑ Fraud detections
↑ Possible false alarms
```

Increasing the threshold usually:

```text
↑ Precision
↓ Number of fraud predictions
↑ Possible missed fraud cases
```

In this project, the threshold is selected using the **validation set**.

The final chosen threshold is then evaluated on the untouched test set.

---

## 10. Confusion Matrix

The confusion matrix helps us understand the actual prediction errors.

For binary fraud detection:

```text
True Negative  → Normal correctly predicted as Normal
False Positive → Normal incorrectly predicted as Fraud
False Negative → Fraud incorrectly predicted as Normal
True Positive  → Fraud correctly predicted as Fraud
```

For fraud detection, **False Negatives** are especially important because they represent fraud transactions that the model failed to detect.

---

## 11. SHAP Explainability

**SHAP — SHapley Additive exPlanations** is used to explain how features influence model predictions.

The main idea is:

```text
Model Prediction
      ↓
SHAP
      ↓
Which features pushed the prediction higher?
Which features pushed the prediction lower?
```

SHAP is used in two ways:

```text
Global Explanation
        +
Local Explanation
```

---

## 12. Global Explanation

Global explainability answers:

> Which features influence the model most across many transactions?

The importance can be summarized using the average absolute SHAP value.

```text
Many Predictions
      ↓
SHAP Values
      ↓
Feature Importance
```

This helps us understand the overall behavior of the model.

---

## 13. Local Explanation

Local explainability answers:

> Why did the model classify this specific transaction this way?

A SHAP waterfall plot can show which features pushed the prediction:

```text
Toward Fraud
or
Toward Normal
```

This gives a detailed explanation for one individual transaction.

### Important Interpretation Rule

SHAP explains the **model's behavior**.

It does not prove that a feature causes fraud in the real world.

---

## What I Learned Today

By the end of Day 5, I learned how to:

- Understand class imbalance in fraud detection.
- Explain why accuracy can be misleading.
- Evaluate a classification model using precision, recall, F1-score, ROC-AUC, and PR-AUC.
- Use a confusion matrix to understand prediction errors.
- Apply SMOTE to the training data only.
- Avoid data leakage when handling imbalance.
- Compare a baseline model with a SMOTE-based model.
- Understand the precision–recall trade-off.
- Explain how a classification threshold converts probability into a class.
- Tune the decision threshold using validation data.
- Perform final evaluation on an untouched test set.
- Use SHAP to explain model predictions.
- Distinguish between global and local model explanations.

---

## Key Takeaway

> **A strong classification model is not defined by accuracy alone. It should be evaluated with the right metrics, handle class imbalance carefully, use an appropriate decision threshold, and provide explainable predictions.**

---

## Project Structure

```text
Week 8/
└── Day 5/
    ├── creditcard.csv
    ├── Day5.ipynb
    └── README.md
```