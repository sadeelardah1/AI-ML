#  Week 4 — Day 2: Cross-Validation

**BinX Tech · AI & Machine Learning Internship Program**

---

## Overview

This notebook is the Day 2 deliverable of Week 4. It builds directly on Day 1 (the three-way train/validation/test split) and introduces **k-fold cross-validation** as a more reliable way to evaluate a model than trusting a single validation split.

By the end of this notebook, the reader will be able to:

- Explain how k-fold cross-validation produces a reliable performance estimate
- Run cross-validation with `cross_val_score` and interpret the mean and standard deviation
- Explain why stratified k-fold matters for classification
- Apply all of the above hands-on, on the Titanic dataset (`train_and_test2.csv`), comparing results against the Day 1 single-split baseline

| | |
|---|---|
| **Dataset** | `train_and_test2.csv` (Titanic passengers — same file as Day 1) |
| **Target** | `Survived` (renamed from `2urvived`) |
| **Model** | `RandomForestClassifier` (carried over from Day 1) |
| **Core tools** | `cross_val_score`, `StratifiedKFold`, `train_test_split` |
| **Cells** | 37 total — 10 code cells, 27 markdown cells |
| **Status** |  Executed end-to-end with no errors |

---

##  Table of Contents

| # | Section | What it covers |
|---|---|---|
| 0 | [Setup](#0-setup) | Importing Pandas, NumPy & Scikit-learn; library versions |
| 1 | [What Cross-Validation Does](#1-what-cross-validation-does) | Why a single validation split can be lucky/unlucky; the k-fold fix |
| 2 | [How k-Fold Works (k = 5)](#2-how-k-fold-works) | The fold-rotation table — what trains vs. what validates each round |
| 3 | [`cross_val_score` — Mean & Standard Deviation](#3-cross_val_score) | Running cross-validation in one call; reading mean vs. std |
| 4 | [Stratified k-Fold for Classification](#4-stratified-k-fold) | Why plain k-fold can distort imbalanced classes; the stratified fix |
| 5 | [Common Mistakes to Avoid](#5-common-mistakes) | Five pitfalls: leakage into the test set, ignoring std, missing `random_state`, etc. |
| 6 | [Quick Reference — Cheat Sheet](#6-quick-reference) | One table with every key line of code from the lesson |
| 7 | [Hands-On Lab — Cross-Validating a Model](#7-hands-on-lab) | The graded lab: 4 steps + 1 reflection question, run on real data |
| 8 | [Best Practices & Reproducibility](#8-best-practices) | Checklist for doing cross-validation correctly every time |
| 9 | [Summary — What I Learned Today](#9-summary) | Five-bullet recap of the day |

---

##  Section 7 in Detail — Hands-On Lab

The lab reuses the Day 1 Titanic model and dataset, split with the same `random_state=42` discipline:

| Step | Task |
|---|---|
| 7.0 | Load `train_and_test2.csv`, rename target column to `Survived`, select features |
| 1 | Rebuild the Day 1 three-way split; run 5-fold `cross_val_score` on the training set only |
| 2 | Report the mean and standard deviation of the fold scores |
| 3 | Compare the cross-validated mean against the Day 1 single-split score |
| 4 | Confirm stratified folds preserve the survival-rate balance; explain why it matters |


> The test set carved off in Day 1 is **never touched** in this notebook — cross-validation runs on `X_train`/`y_train` only, keeping the Day 1 evaluation discipline intact.

---

##  How to Run

1. Place `train_and_test2.csv` in the same folder as this notebook.
2. Open `Day2.ipynb` in Jupyter and run all cells top to bottom (`Kernel → Restart & Run All`).
3. Requires: `pandas`, `numpy`, `matplotlib`, `scikit-learn`.

---

##  Related Files

- `Day1.ipynb` — Train/Validation/Test Splits (prerequisite notebook)
- `train_and_test2.csv` — shared Titanic dataset used across Week 4
- `week_4__Resources.pdf` / `BinX_Tech_AI___ML_Internship_Week4.pdf` — source curriculum this notebook is built from
