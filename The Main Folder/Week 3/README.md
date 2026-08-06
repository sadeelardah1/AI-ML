# Week 3 — Supervised Learning: Regression & Classification with Scikit-learn

**BinX Tech · AI & Machine Learning Internship Program**
**Phase 2: Classical Machine Learning · Week 3 of 10 · 40 hours · 5 Training Days**

---

## Table of Contents

1. [Overview](#overview)
2. [Week 3 Learning Objectives](#week-3-learning-objectives)
3. [Daily Schedule](#daily-schedule)
4. [Folder Structure](#folder-structure)
5. [Day-by-Day Summary](#day-by-day-summary)
   - [Day 1 — Supervised Learning Concepts & the Scikit-learn API](#day-1--supervised-learning-concepts--the-scikit-learn-api)
   - [Day 2 — Linear Regression](#day-2--linear-regression)
   - [Day 3 — Logistic Regression & Classification Metrics](#day-3--logistic-regression--classification-metrics)
   - [Day 4 — Trees, Forests, SVMs & k-NN](#day-4--trees-forests-svms--k-nn)
   - [Day 5 — Supervised-Learning Mini-Project](#day-5--supervised-learning-mini-project)
6. [Datasets Used](#datasets-used)
7. [The Thread That Connects the Week](#the-thread-that-connects-the-week)
8. [Technical Stack](#technical-stack)
9. [Best Practices Applied All Week](#best-practices-applied-all-week)
10. [Week 3 Deliverables](#week-3-deliverables)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Week 3 is the first week of **Phase 2: Classical Machine Learning** — the point where the Python,
Pandas, and math foundations from Weeks 1–2 turn into actual predictive models. Over five days,
the week moves from core Scikit-learn concepts, to a regression model, to a classification model,
to comparing several classifier families, and finally to one complete, professional-grade
end-to-end project.

Every notebook in this folder follows the same visual and structural style: colour-coded callout
boxes (🔵 Note, 🟡 Goal, 🍷 Important, 🟢 Tip), a linked table of contents, worked code examples,
a quick-reference cheat sheet, and a fully executed Hands-On Lab at the end.

## Week 3 Learning Objectives

By the end of Week 3, an intern should be able to:

- Explain what supervised learning is and distinguish regression from classification.
- Split data correctly into training and test sets and explain why this prevents misleading results.
- Train, predict with, and evaluate a linear regression model and interpret its coefficients and error.
- Train and evaluate a logistic regression classifier and read a confusion matrix.
- Train and compare tree-based models (decision trees, random forests) plus SVM and k-NN.
- Assemble a complete supervised-learning mini-project: EDA → preprocessing → model → evaluation.

## Daily Schedule

| Day | Hours | Topic Focus |
|---|---|---|
| Day 1 | 8 hrs | Supervised learning concepts; the Scikit-learn API; train/test split |
| Day 2 | 8 hrs | Linear regression: fitting, predicting, coefficients, regression metrics |
| Day 3 | 8 hrs | Logistic regression & classification; confusion matrix and classification metrics |
| Day 4 | 8 hrs | Decision trees, random forests, SVMs, k-NN; comparing classifiers |
| Day 5 | 8 hrs | Supervised-learning mini-project: full pipeline on a real dataset |

## Folder Structure

```
Week 3/
├── README.md                          ← this file
├── Day 1/
│   ├── Day1.ipynb
│   ├── Day1.py
│   └── README.md
├── Day 2/
│   ├── Day2.ipynb
│   ├── Day2.py
│   └── README.md
├── Day 3/
│   ├── Day3.ipynb
│   ├── Day3.py
│   ├── README.md
│   ├── train_and_test2.csv            ← Titanic-style dataset (classification lab)
│   └── customer_churn_day3.csv        ← extra churn dataset for practice
├── Day 4/
│   ├── Day4.ipynb
│   ├── Day4_Practice.ipynb            ← extra self-practice notebook
│   ├── README.md
│   ├── train_and_test2.csv
│   └── Screenshot *.png               ← reference screenshots
└── Day 5/
    ├── Day5.ipynb
    ├── README.md
    └── train_and_test2.csv
```

## Day-by-Day Summary

### Day 1 — Supervised Learning Concepts & the Scikit-learn API
Opens Phase 2 by establishing the vocabulary used for the rest of the program: what supervised
learning is, regression vs. classification, splitting a dataset into features (`X`) and target
(`y`), the four-step Scikit-learn pattern (`instantiate → fit → predict → score`), and the
train/test split. The Hands-On Lab sets up the complete data-preparation workflow on a sample
"interns" dataset. → See `Day 1/README.md`.

### Day 2 — Linear Regression
Trains the first real predictive model: **Linear Regression**, connecting directly back to the
dot-product math from Week 2. Covers training and predicting, interpreting coefficients and the
intercept, the three core regression metrics (MAE, RMSE, R²), and the rule of always comparing
against a baseline. The Hands-On Lab predicts house prices on a simulated dataset. →
See `Day 2/README.md`.

### Day 3 — Logistic Regression & Classification Metrics
Moves from predicting a number to predicting a category with **Logistic Regression** (weighted
sum → sigmoid → probability). Covers why accuracy alone is misleading on imbalanced data, the
confusion matrix (TP/FP/FN/TN), precision/recall/F1 and their trade-off, and AUC-ROC. The
Hands-On Lab trains a classifier on the Titanic dataset (`train_and_test2.csv`) to predict
passenger survival. → See `Day 3/README.md`.

### Day 4 — Trees, Forests, SVMs & k-NN
Introduces four more classifier families — **decision trees**, **random forests**, **SVMs**, and
**k-NN** — and teaches the professional habit of comparing multiple models fairly on the same
train/test split and metric (the "no free lunch" principle) instead of assuming one algorithm is
best. The Hands-On Lab reuses the Day 3 Titanic split to train and compare all four models,
inspecting the random forest's feature importances. → See `Day 4/README.md`.

### Day 5 — Supervised-Learning Mini-Project
Closes the week by combining everything into one complete, end-to-end pipeline: EDA →
preprocessing (imputation, encoding, scaling via a `ColumnTransformer`, fit on train only) →
stratified train/test split → training a baseline plus multiple models → evaluation with a
confusion matrix, classification report, and F1-score comparison against the baseline → a written
project conclusion. This is a direct rehearsal for the Phase 3 capstone pipeline. →
See `Day 5/README.md`.

## Datasets Used

| Dataset | Used In | Purpose |
|---|---|---|
| `train_and_test2.csv` | Days 3, 4, 5 | Titanic-style passenger data (`Age`, `Fare`, `Sex`, `sibsp`, `Parch`, `Pclass`, `Embarked` → `Survived`); the running classification example that ties Days 3–5 together on one consistent train/test split. |
| `customer_churn_day3.csv` | Day 3 (extra practice) | A simulated telecom churn dataset (tenure, monthly charge, support calls, satisfaction score, etc. → `churn`) for additional classification practice. |

## The Thread That Connects the Week

A key design of this week is that the **same Titanic dataset and train/test split** carries
through Days 3, 4, and 5:

- **Day 3** trains a single logistic regression classifier on it and learns how to judge a
  classifier's quality.
- **Day 4** reuses the exact same split to compare logistic regression against decision trees,
  random forests, SVM, and k-NN — a fair, apples-to-apples comparison.
- **Day 5** rebuilds the pipeline properly from scratch with leak-free preprocessing
  (`ColumnTransformer`, imputers, stratified split) and a written evaluation against a baseline,
  formalizing everything practiced in Days 1–4 into one professional deliverable.

## Technical Stack

| Category | Tools |
|---|---|
| Core ML Library | Scikit-learn (`LinearRegression`, `LogisticRegression`, `tree`, `ensemble`, `svm`, `neighbors`, `dummy`) |
| Preprocessing | `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`, `train_test_split` |
| Metrics | MAE, RMSE, R², `confusion_matrix`, `classification_report`, precision/recall/F1, AUC-ROC |
| Data & Visualization | Pandas, Matplotlib, Seaborn |
| Environment | Python 3.10+, Jupyter Notebook |

## Best Practices Applied All Week

- Fix `random_state=42` everywhere randomness is used, for reproducible results.
- Always evaluate on a held-out **test set**, never on training data.
- Always compare a model's result against a **baseline** — a raw metric means nothing on its own.
- Report more than one metric on classification tasks; accuracy alone is misleading on imbalanced data.
- Fit every preprocessing step (scaler, encoder, imputer) on the **training data only** to avoid
  data leakage.
- Narrate every notebook in Markdown — a working notebook without explanation is an unfinished deliverable.

## Week 3 Deliverables

- A workflow notebook demonstrating the features/target split and a reproducible train/test split (Day 1).
- A linear regression notebook with interpreted coefficients and MAE/RMSE/R² evaluated against a baseline (Day 2).
- A logistic regression notebook with a confusion matrix, precision/recall/F1, and AUC-ROC (Day 3).
- A model-comparison notebook evaluating decision tree, random forest, SVM, and k-NN on one split (Day 4).
- The Week 3 end-to-end mini-project: a trained, evaluated model with full pipeline documentation (Day 5).

## Where This Leads Next

Week 3's supervised-learning foundation — the Scikit-learn API, regression, classification,
model comparison, and a full leak-free pipeline — is reused for every remaining model in the
program. **Week 4** builds on this foundation with new techniques, and the pipeline discipline
established here (EDA → preprocessing → split → model → evaluation) is the exact structure the
**Phase 3 capstone project** will formalize.


