# Week 3 — Day 5: Supervised-Learning Mini-Project

**BinX Tech · AI & Machine Learning Internship Program · Phase 2: Classical Machine Learning**
**Week 3 of 10 · Day 5 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day5.ipynb`](#how-to-use-day5ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. The Full Supervised-Learning Pipeline](#1-the-full-supervised-learning-pipeline)
   - [2. Basic Preprocessing](#2-basic-preprocessing)
   - [3. Choosing the Right Model and Metric](#3-choosing-the-right-model-and-metric)
   - [4. Documenting the Result](#4-documenting-the-result)
7. [Hands-On Lab — End-to-End Mini-Project](#hands-on-lab--end-to-end-mini-project)
8. [Lab Results](#lab-results)
9. [Tools Used](#tools-used)
10. [Best Practices & Reproducibility](#best-practices--reproducibility)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 5 closes out Week 3 by combining everything from Days 1–4 into one complete, end-to-end
supervised-learning project — the same pipeline shape every project in this program follows, and
the exact structure the Phase 3 capstone will formalize: **EDA → preprocessing → train/test split
→ modeling → evaluation against a baseline**.

## Learning Objectives

By the end of Day 5, you should be able to:

- **Assemble** a complete supervised-learning pipeline from EDA to evaluation.
- **Apply** proper preprocessing (imputation, encoding, scaling) without data leakage, using a
  Scikit-learn `Pipeline` and `ColumnTransformer`.
- **Select and justify** an appropriate model and metric for the task, and document the result
  with a confusion matrix and a written conclusion.

## Key Topics

- The full pipeline: EDA → preprocessing → split → model → evaluation
- Preprocessing with `ColumnTransformer`: separate handling for numeric and categorical features
- Imputing missing values (median for numeric, most-frequent for categorical)
- One-hot encoding categorical features, scaling numeric features
- Avoiding data leakage (`fit_transform` on train, `transform` only on test)
- Stratified train/test splitting to preserve class balance
- Choosing the right model and metric for the task
- Documenting the result against a baseline, with a confusion matrix and classification report

## Files in This Folder

| File | Description |
|---|---|
| `Day5.ipynb` | The full, detailed Day 5 lesson notebook — explanations, worked examples, a cheat sheet, and the end-to-end Hands-On Lab, fully executed with outputs (EDA plots, a `ColumnTransformer` preprocessing pipeline, an F1-score-vs-baseline chart, a confusion matrix, and a final written conclusion). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |
| `train_and_test2.csv` | The Titanic-style dataset used as the mini-project dataset (must sit alongside the notebook to run it). |

## How to Use `Day5.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to implement it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** section is a fast reference you can come back to for the rest of the program.
6. Make sure `train_and_test2.csv` is in the same folder as the notebook before running the
   Hands-On Lab cells.
7. The lab ends with an auto-generated **Final Project Conclusion** cell that fills in the actual
   metric values from your run — re-running the notebook updates it automatically.

## Lesson Summary

### 1. The Full Supervised-Learning Pipeline
Every project in this program follows the same five stages:

| Stage | What Happens |
|---|---|
| 1. EDA | Understand the data: distributions, correlations, problems |
| 2. Preprocessing | Impute missing values, encode categories, scale features |
| 3. Train/test split | Hold out unseen data for honest evaluation |
| 4. Modeling | Train one or more models via the Scikit-learn API |
| 5. Evaluation | Measure performance with appropriate metrics vs. a baseline |

### 2. Basic Preprocessing
Numeric and categorical features need different treatment, so this notebook builds a
`ColumnTransformer` that routes each feature type through its own pipeline:
```python
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

X_train_processed = preprocessor.fit_transform(X_train)  # learn from train only
X_test_processed = preprocessor.transform(X_test)        # apply only, never re-fit
```
**Critical rule:** every imputer, encoder, and scaler is fit on the training data only, then
applied to the test data. Fitting on the full dataset leaks information from the test set into
training and inflates results — a subtle but serious mistake called data leakage.

### 3. Choosing the Right Model and Metric
A real project requires two decisions: whether the task is regression or classification, and
which metric best reflects success for the specific problem. Making and justifying these choices
— not just running code — is the actual skill being evaluated.

### 4. Documenting the Result
The deliverable is a clean, narrated notebook: each stage explained in Markdown, each choice
justified, and the final result compared honestly against a baseline — backed up here with a
confusion matrix, a full classification report, and an auto-generated written conclusion.

## Hands-On Lab — End-to-End Mini-Project

Using the **Titanic passenger dataset** (`train_and_test2.csv`), the lab runs a full,
narrated pipeline:

1. **Load and explore** the dataset — missing-value counts, survival rate, an age distribution
   plot, and a correlation heatmap.
2. **Split** the data with `train_test_split(..., stratify=y)` so the training and test sets keep
   the same survival-rate proportion, using `Age`, `Fare`, `sibsp`, `Parch` as numeric features and
   `Sex`, `Pclass`, `Embarked` as categorical features.
3. **Preprocess** with a `ColumnTransformer`: median-impute and scale the numeric features,
   most-frequent-impute and one-hot encode the categorical features — fit on the training data only.
4. **Train and compare** three models — a majority-class `DummyClassifier` baseline, Logistic
   Regression, and a Random Forest — evaluated with accuracy, precision, recall, and F1-score.
5. **Select the best model** by F1-score, quantify its improvement over the baseline, and inspect
   it in depth with a `classification_report` and a confusion matrix.
6. **Read an auto-generated written conclusion** that plugs the run's actual numbers into a final
   project summary.

## Lab Results

On this dataset and split, the **Random Forest** was the best-performing model by F1-score,
clearly beating the majority-class baseline. The classification report and confusion matrix in
the notebook break down exactly how well it distinguished survivors from non-survivors, and the
final conclusion cell explains the result: Random Forest likely won because it can capture
non-linear relationships and interactions between features like passenger class, sex, age, fare,
and embarkation point — something the linear boundary of Logistic Regression cannot represent as
flexibly.

*(Re-running the notebook regenerates these exact numbers from the current data and code.)*

## Tools Used

Scikit-learn (`Pipeline`, `ColumnTransformer`, `SimpleImputer`, `OneHotEncoder`,
`StandardScaler`, `DummyClassifier`, `LogisticRegression`, `RandomForestClassifier`) • Pandas •
Matplotlib / Seaborn • Jupyter Notebook • Git & GitHub

## Best Practices & Reproducibility

- Always fix `random_state=42` for the train/test split and any randomized model.
- Use `stratify=y` on classification splits so class proportions stay consistent between train
  and test.
- Fit every preprocessing step (imputer, encoder, scaler) on the training data only, via a
  `Pipeline`/`ColumnTransformer` — never on the full dataset.
- Always include a baseline in the results table — never report a lone metric.
- Choose the evaluation metric based on the problem, before looking at results.
- Back up a single summary metric with a confusion matrix and classification report before
  declaring a "winner."
- Narrate every stage in Markdown — a working notebook without explanation is an unfinished
  deliverable.
- Commit the finished notebook to GitHub with a clear, descriptive commit message.

## Where This Leads Next

Day 5's full pipeline — proper `ColumnTransformer` preprocessing, stratified splitting, and the
discipline of comparing every result against a baseline — is the direct rehearsal for the
**Phase 3 capstone project**, which formalizes this same EDA → preprocessing → modeling →
evaluation structure into a complete, professional deliverable. With Week 3 complete, the program
moves on to new model families and techniques building on this same Scikit-learn foundation.


