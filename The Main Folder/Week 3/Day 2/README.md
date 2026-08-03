# Week 3 — Day 2: Linear Regression

**BinX Tech · AI & Machine Learning Internship Program · Phase 2: Classical Machine Learning**
**Week 3 of 10 · Day 2 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day2.ipynb`](#how-to-use-day2ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. What Linear Regression Does](#1-what-linear-regression-does)
   - [2. Training and Predicting](#2-training-and-predicting)
   - [3. Interpreting Coefficients](#3-interpreting-coefficients)
   - [4. Regression Metrics](#4-regression-metrics)
   - [5. Always Compare Against a Baseline](#5-always-compare-against-a-baseline)
7. [Hands-On Lab — Predicting a Continuous Value](#hands-on-lab--predicting-a-continuous-value)
8. [Tools Used](#tools-used)
9. [Best Practices & Reproducibility](#best-practices--reproducibility)
10. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 2 of Week 3 puts Day 1's Scikit-learn workflow to work on the first real predictive model of
the program: **Linear Regression**. Today connects directly back to Week 2's linear algebra —
a linear regression prediction is literally the dot product (`X @ weights + bias`) built from
scratch in Week 2, Day 3, now trained automatically by Scikit-learn instead of by hand.

## Learning Objectives

By the end of Day 2, you should be able to:

- **Train** a linear regression model and generate predictions.
- **Interpret** the model's coefficients and intercept.
- **Evaluate** a regression model with MAE, RMSE, and R² against a baseline.

## Key Topics

- What linear regression does: fitting the best line
- Training and predicting with Scikit-learn
- Interpreting coefficients and the intercept
- Regression metrics: MAE, RMSE, R²
- Comparing against a baseline

## Files in This Folder

| File | Description |
|---|---|
| `Day2.ipynb` | The full, detailed Day 2 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs (including a scatter plot with the fitted regression line). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |

## How to Use `Day2.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to implement it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** (Section 7) is a fast reference you can come back to for the rest of the program.

## Lesson Summary

### 1. What Linear Regression Does
Predicts a continuous number by fitting the best straight line (or hyperplane) — exactly the
Week 2 dot product:
```
prediction = (feature₁ × weight₁) + (feature₂ × weight₂) + ... + bias
```

### 2. Training and Predicting
Same Scikit-learn pattern as Day 1:
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### 3. Interpreting Coefficients
```python
print(model.coef_)        # one weight per feature
print(model.intercept_)   # the bias term
```
A coefficient tells you how much the prediction changes when that feature increases by one unit,
holding others constant — a major reason linear regression stays interpretable and widely used.

### 4. Regression Metrics

| Metric | Meaning | Interpretation |
|---|---|---|
| MAE | Mean Absolute Error | Average size of the error, in the target's units — easy to explain |
| RMSE | Root Mean Squared Error | Like MAE but penalizes large errors more heavily |
| R² | Coefficient of determination | Fraction of variance explained; 1.0 is perfect, 0 is no better than the mean |

### 5. Always Compare Against a Baseline
A model that cannot beat a simple baseline (predicting the mean for every row) has learned
nothing useful. A raw MAE or RMSE number means little without this comparison.

## Hands-On Lab — Predicting a Continuous Value

Using a realistic simulated house-price dataset (size, bedrooms, age, distance to center), the
lab runs a complete regression workflow:

1. **Train** a `LinearRegression` model on the dataset.
2. **Report the coefficients** and identify the feature with the strongest effect on price.
3. **Evaluate** the model with MAE, RMSE, and R² on the test set.
4. **Compare RMSE against a baseline** that always predicts the mean price, and explicitly state
   whether the model adds real value.
5. **Document the interpretation** of all results in plain language.

## Tools Used

Scikit-learn (LinearRegression) • Pandas • Jupyter Notebook

## Best Practices & Reproducibility

- Always fix a random seed and `random_state` wherever randomness is used: `np.random.seed(42)`
  for sample data, `random_state=42` for the train/test split.
- Report at least one error metric (MAE/RMSE) **and** R² — not just one alone.
- Always compare against a baseline before claiming a model is useful.
- Check that coefficients make intuitive sense (e.g. a negative coefficient for something that
  should reduce the target).
- Evaluate exclusively on the test set, never the training set.

## Where This Leads Next

Day 2's regression foundation — training, interpreting, and honestly evaluating a model against
a baseline — is the exact workflow reused for every model in the rest of the program. **Days 3–4**
apply the same Scikit-learn pattern to **classification** models, where the target is a category
instead of a number.


