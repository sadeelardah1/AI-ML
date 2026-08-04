# Week 3 — Day 3: Logistic Regression & Classification Metrics

**BinX Tech · AI & Machine Learning Internship Program · Phase 2: Classical Machine Learning**
**Week 3 of 10 · Day 3 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day3.ipynb`](#how-to-use-day3ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. From Regression to Classification](#1-from-regression-to-classification)
   - [2. Why Accuracy Alone Is Misleading](#2-why-accuracy-alone-is-misleading)
   - [3. The Confusion Matrix](#3-the-confusion-matrix)
   - [4. Precision, Recall & F1](#4-precision-recall--f1)
   - [5. AUC-ROC](#5-auc-roc)
7. [Hands-On Lab — Building a Classifier](#hands-on-lab--building-a-classifier)
8. [Tools Used](#tools-used)
9. [Best Practices & Reproducibility](#best-practices--reproducibility)
10. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 3 of Week 3 moves from predicting a number to predicting a **category**. Yesterday's Linear
Regression model output a house price; today's **Logistic Regression** model outputs a
probability — did a passenger survive, or not? Same Scikit-learn workflow as Day 2
(`instantiate → fit → predict`), applied to a new kind of target, plus a full toolkit for judging
whether a classifier is actually any good.

## Learning Objectives

By the end of Day 3, you should be able to:

- **Train** a logistic regression classifier and obtain class probabilities.
- **Explain** why accuracy alone is misleading on imbalanced data.
- **Read** a confusion matrix and compute precision, recall, F1, and AUC-ROC.

## Key Topics

- Logistic regression: weighted sum + sigmoid → probability
- Why accuracy is misleading on imbalanced data
- The confusion matrix: TP, FP, FN, TN
- Precision, recall, F1, and their trade-off
- AUC-ROC

## Files in This Folder

| File | Description |
|---|---|
| `Day3.ipynb` | The full, detailed Day 3 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs (sigmoid curve, confusion matrix heatmap, and ROC curve). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |
| `train_and_test2.csv` | The Titanic-style dataset used in the Hands-On Lab to predict passenger survival. |

## How to Use `Day3.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to implement it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** (Section 7) is a fast reference you can come back to for the rest of the program.
6. Make sure `train_and_test2.csv` is in the same folder as the notebook before running the
   Hands-On Lab cells.

## Lesson Summary

### 1. From Regression to Classification
Logistic regression computes the same weighted sum as linear regression, then passes it through
the **sigmoid function**, which squashes any number into a probability between 0 and 1:
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)   # class probabilities
```
If the probability exceeds a threshold (usually 0.5), the model predicts the positive class.

### 2. Why Accuracy Alone Is Misleading
Accuracy is the fraction of correct predictions — but on imbalanced data it can be dangerously
misleading. If 95% of patients don't have a disease, a model that always predicts "no disease"
scores 95% accuracy while catching zero real cases. This is why classification needs richer metrics.

### 3. The Confusion Matrix
The foundation of every classification metric — it breaks predictions into four categories:

| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) — missed |
| **Actual Negative** | False Positive (FP) — false alarm | True Negative (TN) |

```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predictions))
```

### 4. Precision, Recall & F1

| Metric | Formula | Answers the Question |
|---|---|---|
| Precision | TP / (TP + FP) | Of everything I predicted positive, how much was right? |
| Recall | TP / (TP + FN) | Of all actual positives, how many did I catch? |
| F1-score | harmonic mean of the two | A single balanced score when you need both |

The trade-off depends on the problem: for disease screening, recall matters most (never miss a
real case); for a spam filter, precision matters most (never block a real email).
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
```

### 5. AUC-ROC
The ROC curve plots the trade-off between catching positives and raising false alarms across
every threshold; the AUC summarizes it as a single number from 0.5 (random guessing) to 1.0
(perfect). It is the standard metric for comparing classifiers independently of any single
threshold choice.
```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, probabilities[:, 1])
```

## Hands-On Lab — Building a Classifier

Using a real **Titanic passenger dataset** (`train_and_test2.csv`), the lab runs a complete
classification workflow:

1. **Train** a `LogisticRegression` model to predict passenger survival.
2. **Generate predictions** and produce the confusion matrix (visualized as a heatmap).
3. **Compute** precision, recall, and F1 with `classification_report`, and interpret each.
4. **Decide** whether precision or recall matters more for this specific problem, and justify it.
5. **Compute the AUC-ROC**, plot the ROC curve, and document what it says about the model.

## Tools Used

Scikit-learn (LogisticRegression) • Pandas • Matplotlib • Seaborn • Jupyter Notebook

## Best Practices & Reproducibility

- Always fix `random_state=42` for the train/test split, for reproducible results.
- Report more than one metric — never accuracy alone on a real classification problem.
- Always inspect the confusion matrix before trusting a summary metric.
- Choose which metric to prioritize (precision vs. recall) based on the real-world cost of each
  type of mistake, before looking at the results.
- Evaluate exclusively on the test set, never the training set.

## Where This Leads Next

Day 3's classification toolkit — the confusion matrix, precision/recall/F1, and AUC-ROC — is the
exact evaluation framework reused for every classifier for the rest of the program. **Day 4**
introduces new *kinds* of classifiers (decision trees, random forests, SVMs, k-NN) and compares
them fairly using these same metrics.

