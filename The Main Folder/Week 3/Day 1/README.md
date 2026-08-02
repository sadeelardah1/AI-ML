# Week 3 — Day 1: Supervised Learning Concepts & the Scikit-learn API

**BinX Tech · AI & Machine Learning Internship Program · Phase 2: Classical Machine Learning**
**Week 3 of 10 · Day 1 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day1.ipynb`](#how-to-use-day1ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. What Supervised Learning Is](#1-what-supervised-learning-is)
   - [2. Regression vs. Classification](#2-regression-vs-classification)
   - [3. Features (X) and Target (y)](#3-features-x-and-target-y)
   - [4. The Scikit-learn API](#4-the-scikit-learn-api)
   - [5. The Train/Test Split](#5-the-traintest-split)
7. [Hands-On Lab — Setting Up the ML Workflow](#hands-on-lab--setting-up-the-ml-workflow)
8. [Tools Used](#tools-used)
9. [Best Practices & Reproducibility](#best-practices--reproducibility)
10. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 1 of Week 3 opens **Phase 2: Classical Machine Learning**, moving from the math foundations
of Week 2 into building actual predictive models. Today establishes the conceptual and
practical vocabulary used for the rest of the program: what supervised learning is, how
regression differs from classification, how a dataset splits into features and a target, and
the consistent four-step Scikit-learn workflow every model in the library follows.

## Learning Objectives

By the end of Day 1, you should be able to:

- Explain **supervised learning** and distinguish **regression** from **classification**.
- Separate a dataset into **features (X)** and **target (y)**.
- Perform a **train/test split** and explain why evaluating on unseen data is essential.

## Key Topics

- What supervised learning is: learning from labeled examples
- Regression vs. classification
- Features (X) and target (y)
- The consistent Scikit-learn API: instantiate, fit, predict, score
- The train/test split and why it matters

## Files in This Folder

| File | Description |
|---|---|
| `Day1.ipynb` | The full, detailed Day 1 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs. |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |

## How to Use `Day1.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to implement it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** (Section 7) is a fast reference you can come back to for the rest of the program.

## Lesson Summary

### 1. What Supervised Learning Is
Trains a model on labeled examples — data where the correct answer (the target) is already
known — so it can predict that answer for new, unseen data.

### 2. Regression vs. Classification

| | Regression | Classification |
|---|---|---|
| Predicts | A continuous number | A category / class |
| Example | House price, temperature, income | Spam / not spam, churn / stay |
| Example metric | RMSE, MAE, R² | Accuracy, precision, recall, F1 |

### 3. Features (X) and Target (y)
```python
X = df.drop("target", axis=1)   # features: everything except the target
y = df["target"]                # target: the column to predict
```

### 4. The Scikit-learn API

| Step | Method | What It Does |
|---|---|---|
| 1. Instantiate | `model = Model()` | Create the model, set its options |
| 2. Fit | `model.fit(X_train, y_train)` | Learn patterns from training data |
| 3. Predict | `model.predict(X_test)` | Predict on new data |
| 4. Score | `model.score(X_test, y_test)` | Evaluate performance |

### 5. The Train/Test Split
Never evaluate a model on the same data it was trained on — a model can memorize training data
and look perfect, yet fail completely on new data:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Hands-On Lab — Setting Up the ML Workflow

Using a sample "interns" dataset (predicting whether an intern completes the program), the lab
sets up the complete data-preparation workflow every supervised learning project in this program
will start with:

1. **Load a dataset** and separate it into features `X` and target `y`.
2. **Perform an 80/20 train/test split** with a fixed `random_state`.
3. **Confirm the shapes** of `X_train`, `X_test`, `y_train`, `y_test` are consistent, with
   explicit assertion checks.
4. **Explain in writing** why the model must never see the test set during training.

## Tools Used

Scikit-learn • Pandas • Jupyter Notebook

## Best Practices & Reproducibility

- Always fix a random seed and `random_state` wherever randomness is used: `np.random.seed(42)`
  for sample data, `random_state=42` for the train/test split.
- Always separate `X` and `y` clearly before doing anything else.
- Always perform the train/test split **before** calling `.fit()`.
- Always score a model on the **test** set, never the training set.

## Where This Leads Next

Day 1's workflow — separate X/y, split, instantiate, fit, predict, score — is the exact template
used for every model built for the rest of the program. **Day 2** applies it to a real
**linear regression** model; **Days 3–4** apply the same pattern to **classification** models.

---

*Prepared by BinX Tech · Palestine | Nablus*
