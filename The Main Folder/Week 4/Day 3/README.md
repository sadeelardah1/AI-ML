#  Week 4 — Day 3: Bias-Variance &amp; Diagnosing Model Fit

**BinX Tech · AI & Machine Learning Internship Program**


---

## Overview

This notebook is the Day 3 deliverable of Week 4. It builds on Day 1 (train/validation/test discipline) and Day 2 (cross-validation), and teaches how to **diagnose** a disappointing score: is the model underfitting or overfitting, and which fix actually applies.

By the end of this notebook, the reader will be able to:

- Distinguish underfitting from overfitting by their symptoms
- Explain the bias-variance trade-off and its role in tuning
- Diagnose model fit from the train-vs-validation score gap
- Apply regularization (Ridge / Lasso) — and explain why it only cures overfitting, not underfitting
- Do all of the above hands-on, on the Titanic dataset (`train_and_test2.csv`), with real score evidence

| | |
|---|---|
| **Dataset** | `train_and_test2.csv` (Titanic passengers — same file as Day 1 & Day 2) |
| **Target** | `Survived` (renamed from `2urvived`) |
| **Models used** | `DecisionTreeClassifier` (over/under-fit demo), `RidgeClassifier` (regularization demo) |
| **Core tools** | `Ridge`, `Lasso`, `DecisionTreeClassifier`, `StandardScaler`, `f1_score` |
| **Cells** | 41 total — 9 code cells, 32 markdown cells |
| **Status** |  Executed end-to-end with no errors |

---

##  Table of Contents

| # | Section | What it covers |
|---|---|---|
| 0 | [Setup](#0-setup) | Importing Pandas, NumPy & Scikit-learn; library versions |
| 1 | [The Two Ways a Model Fails](#1-the-two-ways-a-model-fails) | Underfitting vs. overfitting — symptom / cause / fix table |
| 2 | [The Bias-Variance Trade-off](#2-the-bias-variance-trade-off) | Bias vs. variance; the complexity "sweet spot"; a conceptual chart |
| 3 | [Diagnosing With the Train-vs-Validation Gap](#3-diagnosing-with-the-train-vs-validation-gap) | The three-row diagnostic table: low/low, high/much-lower, high/high |
| 4 | [Regularization — Ridge (L2) &amp; Lasso (L1)](#4-regularization) | What regularization does; the `alpha` parameter; why it only fixes overfitting |
| 5 | [Common Mistakes to Avoid](#5-common-mistakes) | Five pitfalls: wrong-direction fixes, mismatched metrics, guessing alpha, etc. |
| 6 | [Quick Reference — Cheat Sheet](#6-quick-reference) | One table with every key line of code and every diagnostic signature |
| 7 | [Hands-On Lab — Diagnosing and Fixing Model Fit](#7-hands-on-lab) | The graded lab: 4 steps + 1 reflection question, run on real data |
| 8 | [Best Practices & Reproducibility](#8-best-practices) | Checklist for diagnosing and fixing model fit correctly every time |
| 9 | [Summary — What I Learned Today](#9-summary) | Five-bullet recap of the day |

---

##  Section 7 in Detail — Hands-On Lab

The lab reuses the Day 1/Day 2 Titanic split (`random_state=42`) and deliberately pushes a model to each failure mode, then fixes it:

| Step | Task | Result on this dataset |
|---|---|---|
| 7.0 | Load `train_and_test2.csv`, rename target to `Survived`, select features | 1,309 rows × 6 features |
| 1 | Overfit deliberately — unlimited-depth decision tree | Train F1 ≈ 0.97, Val F1 ≈ 0.50 → **large gap** (overfitting) |
| 2 | Underfit deliberately — depth-1 decision stump | Train F1 ≈ 0.58, Val F1 ≈ 0.56 → **both low, small gap** (underfitting) |
| 3 | Regularize / simplify the overfit model — `max_depth=4`, plus a `RidgeClassifier` alpha sweep | Gap shrinks to ≈ 0 as complexity is constrained |
| 4 | Document each diagnosis with the score evidence | Fill-in table + written reflection |

> All three splits (`X_train`/`X_val`/`X_test`) are the exact same split used in Day 1 & Day 2 — the held-out `X_test`/`y_test` is **never touched** in this notebook.

---

##  How to Run

1. Place `train_and_test2.csv` in the same folder as this notebook.
2. Open `Day3.ipynb` in Jupyter and run all cells top to bottom (`Kernel → Restart & Run All`).
3. Requires: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.

---

##  Related Files

- `Day1.ipynb` — Train/Validation/Test Splits (prerequisite notebook)
- `Day2.ipynb` — Cross-Validation (prerequisite notebook)
- `train_and_test2.csv` — shared Titanic dataset used across Week 4
- `week_4__Resources.pdf` / `BinX_Tech_AI___ML_Internship_Week4.pdf` — source curriculum this notebook is built from
