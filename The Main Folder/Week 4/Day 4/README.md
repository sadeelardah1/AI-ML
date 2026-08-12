# Week 4 — Day 4: Feature Engineering & Hyperparameter Tuning

**BinX Tech — AI & Machine Learning Internship Program**

## Overview

This notebook is the Day 4 deliverable of Week 4. It builds on Day 1–3 (splits, cross-validation, bias-variance) and covers the two levers that most improve a model in practice: better features, and systematic hyperparameter search instead of hand-tuning.

Each concept in this notebook is explained in three layers: what it is, why it exists, and why it matters in AI/ML practice specifically — not just as a Scikit-learn function to memorize.

By the end of this notebook, the reader will be able to:

- Explain what a feature is and why a model can only ever learn from the columns it is given
- Engineer new features and apply appropriate transformations, understanding the reasoning behind each technique
- Distinguish hyperparameters from learned parameters, and explain why that distinction matters
- Tune a model systematically with GridSearchCV and cross-validation, and explain why cross-validation is built into the search
- Apply all of the above hands-on, on the Titanic dataset (`train_and_test2.csv`), with a documented comparison against the untuned baseline

| | |
|---|---|
| Dataset | `train_and_test2.csv` (Titanic passengers — same file as Days 1–3) |
| Target | `Survived` (renamed from `2urvived`) |
| Model | `RandomForestClassifier`, tuned with `GridSearchCV` |
| Core tools | `GridSearchCV`, `cross_val_score`, `RandomForestClassifier`, feature engineering with `pandas` |
| Cells | 46 total — 10 code cells, 36 markdown cells |
| Status | Executed end-to-end with no errors |

## Table of Contents

| # | Section | What it covers |
|---|---|---|
| 0 | Setup | Importing Pandas, NumPy & Scikit-learn; library versions |
| 1 | Why Feature Engineering Often Beats Model Choice | What a feature is, what feature engineering is, and why it usually beats a fancier model |
| 2 | Common Feature Engineering Techniques | Creating features, binning, one-hot encoding, datetime extraction, scaling — each explained with its own "why" |
| 3 | Hyperparameters vs. Parameters | What each one is, and why the distinction is what makes model behaviour explainable |
| 4 | GridSearchCV | Defining a grid, why cross-validation is built into the search, reading `best_params_` / `best_score_` |
| 5 | Common Mistakes to Avoid | Five pitfalls: leakage while engineering, oversized grids, skipping the baseline, etc. |
| 6 | Quick Reference | One table with every key line of code from the lesson |
| 7 | Hands-On Lab — Engineering Features & Tuning | The graded lab: 5 steps + 1 reflection question, run on real data |
| 8 | Best Practices & Reproducibility | Checklist for engineering and tuning correctly every time |
| 9 | Summary — What I Learned Today | Five-bullet recap of the day |

## Section 7 in Detail — Hands-On Lab

The lab reuses the Day 1–3 Titanic split (`random_state=42`) and follows the five steps from the curriculum exactly:

| Step | Task | Result on this dataset |
|---|---|---|
| 7.0 | Load `train_and_test2.csv`, rename target to `Survived` | 1,309 rows |
| 1 | Engineer at least two new features, justified in Markdown | `family_size`, `fare_per_person`, `is_alone` |
| 2 | Define a hyperparameter grid for a Week 3 model (random forest) | `n_estimators: [100, 200]`, `max_depth: [5, 10, None]` |
| 3 | Run GridSearchCV with 5-fold cross-validation, report best params/score | best params `{max_depth: 10, n_estimators: 200}`, F1 ≈ 0.517 |
| 4 | Compare tuned score against the untuned Week 3 baseline | baseline F1 ≈ 0.515, tuned F1 ≈ 0.517 — a small, honest improvement |
| 5 | Document which engineered feature and hyperparameter mattered most | feature-importance bar chart plus a written reflection |
| 6 | Final reflection | `fare_per_person` was the strongest engineered feature; GridSearchCV selected `max_depth=10` and `n_estimators=200` |

All splits (`X_train`/`X_val`/`X_test`) reuse the exact Day 1–3 discipline — the held-out test set is never touched in this notebook. GridSearchCV's cross-validation runs on `X_train`/`y_train` only.

## How to Run

1. Place `train_and_test2.csv` in the same folder as this notebook.
2. Open `Day4.ipynb` in Jupyter and run all cells top to bottom (Kernel → Restart & Run All).
3. Requires: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.

## Related Files

- `Day1.ipynb` — Train/Validation/Test Splits (prerequisite notebook)
- `Day2.ipynb` — Cross-Validation (prerequisite notebook)
- `Day3.ipynb` — Bias-Variance & Diagnosing Model Fit (prerequisite notebook)
- `train_and_test2.csv` — shared Titanic dataset used across Week 4
- `week_4__Resources.pdf` / `BinX_Tech_AI___ML_Internship_Week4.pdf` — source curriculum this notebook is built from
