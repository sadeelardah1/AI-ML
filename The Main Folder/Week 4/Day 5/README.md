# Week 4 — Day 5: Scikit-learn Pipelines & Tuned Mini-Project

**BinX Tech — AI & Machine Learning Internship Program — Phase 2**
Part of the *Evaluation, Tuning & Pipelines* week — the closing notebook of Week 4.

## Overview

This notebook is the Day 5 deliverable of Week 4. It closes the week by chaining everything from Days 1–4 — splitting, cross-validation, diagnosing fit, feature engineering, and hyperparameter tuning — into a single, leak-free, reproducible object: a Scikit-learn Pipeline.

Each concept is explained in three layers: what it is, why it exists, and why it matters in AI/ML practice specifically — not just as a Scikit-learn function to memorize.

By the end of this notebook, the reader will be able to:

- Explain what data leakage actually is and why a Pipeline prevents it structurally, not just by convention
- Build a Pipeline that chains preprocessing and modeling into one object
- Use a ColumnTransformer to preprocess numeric and categorical columns differently
- Tune an entire pipeline with GridSearchCV using the double-underscore parameter convention
- Apply all of the above hands-on, on the Titanic dataset (`train_and_test2.csv`), evaluating a final tuned pipeline exactly once on the held-out test set

| | |
|---|---|
| Dataset | `train_and_test2.csv` (Titanic passengers — same file as Days 1–4) |
| Target | `Survived` (renamed from `2urvived`) |
| Model | `RandomForestClassifier` inside a `Pipeline` with a `ColumnTransformer`, tuned with `GridSearchCV` |
| Core tools | `Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `GridSearchCV` |
| Status | Executed end-to-end with no errors |

## Table of Contents

| # | Section | What it covers |
|---|---|---|
| 0 | Setup | Importing Pandas, NumPy & Scikit-learn; library versions |
| 1 | Why Pipelines Exist | What data leakage is, why it is dangerous, and what a Pipeline actually is |
| 2 | Building a Pipeline | What goes inside a Pipeline, and what happens when `.fit()`/`.predict()` are called |
| 3 | ColumnTransformer for Mixed Data | Why one preprocessor is not enough, and how ColumnTransformer solves it |
| 4 | Tuning a Whole Pipeline | The double-underscore naming convention, and why the whole object is tuned, not just the model |
| 5 | The Week 4 Mini-Project | How the whole week's work converges into one tuned pipeline |
| 6 | Common Mistakes to Avoid | Five pitfalls: preprocessing outside the Pipeline, missing step prefixes, re-touching the test set, etc. |
| 7 | Quick Reference | One table with every key line of code from the lesson |
| 8 | Hands-On Lab — Tuned End-to-End Pipeline | The graded lab: 5 steps + 1 reflection question, run on real data |
| 9 | Best Practices & Reproducibility | Checklist for building and tuning pipelines correctly every time |
| 10 | Summary — What I Learned This Week | A day-by-day recap tying all of Week 4 together |

## Section 8 in Detail — Hands-On Lab

The lab reuses the Day 1–4 Titanic split (`random_state=42`) and follows the five steps from the curriculum exactly:

| Step | Task | Result on this dataset |
|---|---|---|
| 8.0 | Load `train_and_test2.csv`, rename target to `Survived` | 1,309 rows |
| 1 | Build a Pipeline with a ColumnTransformer (scaling + encoding) | numeric columns scaled, `Sex`/`Pclass` one-hot encoded |
| 2 | Add the Day 4 engineered features (`family_size`, `fare_per_person`, `is_alone`) into the workflow | Train 785 / Val 262 / Test 262 rows |
| 3 | Tune the full pipeline with GridSearchCV and 5-fold cross-validation | best params `{max_depth: 10, n_estimators: 100}`, cross-validated F1 ≈ 0.547 |
| 4 | Evaluate the final tuned pipeline once on the held-out test set, against a baseline | baseline test F1 ≈ 0.420, tuned test F1 ≈ 0.435 — a small, honest improvement |
| 5 | Note the finished workflow's structure (pipeline diagram) | visual confirmation of the leak-free, single-object structure |

The held-out test set is opened exactly once in Step 4 — after every tuning decision was already finalized using cross-validation on the training set only, closing the loop that started with the Day 1 three-way split.

## How to Run

1. Place `train_and_test2.csv` in the same folder as this notebook.
2. Open `Day5.ipynb` in Jupyter and run all cells top to bottom (Kernel → Restart & Run All).
3. Requires: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.

## Related Files

- `Day1.ipynb` — Train/Validation/Test Splits (prerequisite notebook)
- `Day2.ipynb` — Cross-Validation (prerequisite notebook)
- `Day3.ipynb` — Bias-Variance & Diagnosing Model Fit (prerequisite notebook)
- `Day4.ipynb` — Feature Engineering & Hyperparameter Tuning (prerequisite notebook)
- `train_and_test2.csv` — shared Titanic dataset used across Week 4
- `week_4__Resources.pdf` / `BinX_Tech_AI___ML_Internship_Week4.pdf` — source curriculum this notebook is built from
