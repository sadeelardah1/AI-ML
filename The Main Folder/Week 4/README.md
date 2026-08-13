# Week 4 — Evaluation, Tuning & Pipelines

**BinX Tech - AI & Machine Learning Internship Program**
**Phase 2: Evaluation, Tuning & Pipelines | Week 4 of 10 | 5 Days | 40 Hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Weekly Learning Objectives](#weekly-learning-objectives)
3. [Dataset](#dataset)
4. [Repository Structure](#repository-structure)
5. [Day 1 — Train / Validation / Test Splits](#day-1--train--validation--test-splits)
6. [Day 2 — Cross-Validation](#day-2--cross-validation)
7. [Day 3 — Bias-Variance & Diagnosing Model Fit](#day-3--bias-variance--diagnosing-model-fit)
8. [Day 4 — Feature Engineering & Hyperparameter Tuning](#day-4--feature-engineering--hyperparameter-tuning)
9. [Day 5 — Scikit-learn Pipelines & Tuned Mini-Project](#day-5--scikit-learn-pipelines--tuned-mini-project)
10. [How to Run](#how-to-run)
11. [Tools & Libraries Used](#tools--libraries-used)
12. [Best Practices Followed Across the Week](#best-practices-followed-across-the-week)
13. [Week 4 Summary](#week-4-summary)

---

## Overview

Week 4 is the second week of Phase 2. It moves beyond simply building a model that runs, toward building a model that can be trusted and evaluated correctly. The week starts with the discipline of splitting data honestly, moves through cross-validation and diagnosing overfitting/underfitting, adds feature engineering and systematic hyperparameter tuning, and closes by chaining the entire workflow into a single, leak-free Scikit-learn Pipeline.

Every day builds directly on the one before it, using the same Titanic dataset and the same `random_state=42` discipline throughout, so results stay comparable across the whole week.

## Weekly Learning Objectives

By the end of Week 4, the learner should be able to:

- Split data correctly into training, validation, and test sets, and explain why each set has exactly one job.
- Use k-fold and stratified k-fold cross-validation to obtain a reliable performance estimate.
- Diagnose whether a model is underfitting or overfitting from the train-vs-validation score gap.
- Apply regularization (Ridge / Lasso) and explain why it only fixes overfitting, not underfitting.
- Engineer new features and justify each transformation.
- Distinguish hyperparameters from learned parameters and tune them systematically with GridSearchCV.
- Build a Scikit-learn Pipeline with a ColumnTransformer, and tune the entire pipeline without leaking data.
- Evaluate a final tuned model on a held-out test set exactly once, after every decision is final.

## Dataset

All five days use the same dataset:

| File | Description |
|---|---|
| `train_and_test2.csv` | Titanic passenger dataset, reused across every notebook. Target column `2urvived` is renamed to `Survived` in each notebook. |

## Repository Structure

```
Week 4/
├── Day 1/
│   ├── Day1.ipynb
│   ├── Day1Practice.py
│   ├── README.md
│   └── train_and_test2.csv
├── Day 2/
│   ├── Day2.ipynb
│   ├── Day2.py
│   ├── Day 2.pdf
│   ├── README.md
│   └── train_and_test2.csv
├── Day 3/
│   ├── Day3.ipynb
│   ├── Day3.py
│   ├── Day3.pdf
│   ├── README.md
│   └── train_and_test2.csv
├── Day 4/
│   ├── Day4.ipynb
│   ├── Day4.py
│   ├── README.md
│   └── train_and_test2.csv
└── Day 5/
    ├── Day5.ipynb
    ├── README.md
    └── train_and_test2.csv
```

## Day 1 — Train / Validation / Test Splits

Introduces the three-way split (train, validation, test) and explains why repeatedly checking a single test set while tuning makes that test score dishonest.

| Item | Detail |
|---|---|
| Key topics | Problem with a single test set, the three-way split, creating the split in code, limits of one validation set |
| Core tools | `train_test_split`, `RandomForestClassifier` |
| Hands-On Lab | Build a 60/20/20 split, tune `max_depth` on the validation set, evaluate on the test set once, report F1-score |
| Result | Foundation split reused in every later day |

## Day 2 — Cross-Validation

Replaces the single validation split with k-fold cross-validation for a more stable performance estimate, and introduces stratified k-fold for classification.

| Item | Detail |
|---|---|
| Key topics | What cross-validation does, how k-fold works, `cross_val_score`, stratified k-fold, common mistakes |
| Core tools | `cross_val_score`, `StratifiedKFold`, `train_test_split` |
| Hands-On Lab | Run 5-fold cross-validation on the training set, compare mean/std against the Day 1 single-split score |
| Note | The Day 1 test set is never touched; cross-validation runs on the training set only |

## Day 3 — Bias-Variance & Diagnosing Model Fit

Teaches how to diagnose a disappointing score as underfitting or overfitting, and how to fix it with regularization.

| Item | Detail |
|---|---|
| Key topics | Underfitting vs. overfitting, bias-variance trade-off, train-vs-validation gap diagnosis, Ridge/Lasso regularization |
| Models used | `DecisionTreeClassifier` (fit demonstration), `RidgeClassifier` (regularization demonstration) |
| Hands-On Lab | Deliberately overfit and underfit a decision tree, then regularize; document each diagnosis with score evidence |
| Result | Overfit model: Train F1 ≈ 0.97, Val F1 ≈ 0.50. Underfit model: Train F1 ≈ 0.58, Val F1 ≈ 0.56. Gap closes after regularization |

## Day 4 — Feature Engineering & Hyperparameter Tuning

Covers the two levers that most improve a model in practice: better features and systematic hyperparameter search.

| Item | Detail |
|---|---|
| Key topics | Why feature engineering often beats model choice, common engineering techniques, hyperparameters vs. parameters, GridSearchCV |
| Core tools | `GridSearchCV`, `cross_val_score`, `RandomForestClassifier` |
| Hands-On Lab | Engineer `family_size`, `fare_per_person`, `is_alone`; grid-search `n_estimators` and `max_depth`; compare tuned vs. untuned baseline |
| Result | Best params: `max_depth=10`, `n_estimators=200`. Baseline F1 ≈ 0.515, tuned F1 ≈ 0.517 |

## Day 5 — Scikit-learn Pipelines & Tuned Mini-Project

Closes the week by chaining splitting, cross-validation, diagnosis, feature engineering, and tuning into a single leak-free Pipeline object.

| Item | Detail |
|---|---|
| Key topics | Why Pipelines prevent data leakage, building a Pipeline, ColumnTransformer for mixed data, tuning a whole pipeline with the double-underscore convention |
| Core tools | `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `GridSearchCV` |
| Hands-On Lab | Build and tune a full pipeline including Day 4 engineered features; evaluate once on the held-out test set |
| Result | Best params: `max_depth=10`, `n_estimators=100`, cross-validated F1 ≈ 0.547. Baseline test F1 ≈ 0.420, tuned test F1 ≈ 0.435 |

## How to Run

1. Open the relevant day's notebook (`DayN.ipynb`) in Jupyter Notebook, VS Code, or Google Colab.
2. Make sure `train_and_test2.csv` is in the same folder as the notebook.
3. Run all cells top to bottom (`Kernel → Restart & Run All`). All notebooks are already executed once, so saved outputs can also be read without re-running.
4. Follow the notebooks in order (Day 1 through Day 5), since each day depends on the split and decisions made in the previous day.

## Tools & Libraries Used

Pandas • NumPy • Matplotlib • Scikit-learn (`train_test_split`, `cross_val_score`, `StratifiedKFold`, `RandomForestClassifier`, `DecisionTreeClassifier`, `Ridge`, `Lasso`, `GridSearchCV`, `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) • Jupyter Notebook

## Best Practices Followed Across the Week

- The test set is carved off first, before any tuning, and opened exactly once at the very end.
- Every split uses `random_state=42` for full reproducibility.
- All tuning decisions are made using the validation set or cross-validation, never the test set.
- Cross-validation and GridSearchCV always run on the training set only.
- Preprocessing is placed inside the Pipeline, not applied separately, to avoid data leakage.

## Week 4 Summary

| Day | Focus | Key Deliverable |
|---|---|---|
| 1 | Train/Validation/Test Splits | Correct three-way split methodology |
| 2 | Cross-Validation | Stable performance estimate via k-fold |
| 3 | Bias-Variance & Diagnosis | Underfitting/overfitting diagnosis and regularization |
| 4 | Feature Engineering & Tuning | Engineered features and GridSearchCV tuning |
| 5 | Pipelines & Mini-Project | Full leak-free pipeline, tuned and evaluated once on test data |

By the end of Week 4, the full evaluation-and-tuning workflow — split, validate, diagnose, engineer, tune, and pipeline — is complete and reusable for any future model in the program.
