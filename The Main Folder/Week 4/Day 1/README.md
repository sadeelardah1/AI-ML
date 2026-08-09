# Week 4 — Day 1: Train / Validation / Test Splits

**BinX Tech · AI & Machine Learning Internship Program · Phase 2: Evaluation, Tuning & Pipelines**
**Week 4 of 10 · Day 1 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day1.ipynb`](#how-to-use-day1ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. The Problem With a Single Test Set](#1-the-problem-with-a-single-test-set)
   - [2. The Three-Way Split](#2-the-three-way-split)
   - [3. Creating the Split in Code](#3-creating-the-split-in-code)
   - [4. Why This Isn't Always Enough](#4-why-this-isnt-always-enough)
7. [Hands-On Lab — Building a Three-Way Split](#hands-on-lab--building-a-three-way-split)
8. [Tools Used](#tools-used)
9. [Best Practices & Reproducibility](#best-practices--reproducibility)
10. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Week 4 is the second week of Phase 2 — the point where a model that simply *runs* becomes a model
you can *trust*. Day 1 opens the week by exposing a hidden trap in the Week 3 workflow: repeatedly
checking a single test set while tuning quietly turns that test score dishonest. The fix is the
**three-way split** — train, validation, and test — with a strict rule about when each is touched.

## Learning Objectives

By the end of Day 1, you should be able to:

- **Explain** why a validation set is needed in addition to a test set.
- **Create** a correct three-way split in Scikit-learn.
- **Explain** why tuning against the test set produces misleading results.

## Key Topics

- The problem with tuning against a single test set
- The three-way split: train, validation, test
- Creating a three-way split in code
- Why one validation set can still mislead (motivating cross-validation)

## Files in This Folder

| File | Description |
|---|---|
| `Day1.ipynb` | The full, detailed Day 1 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs. |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |
| `train_and_test2.csv` | The Titanic-style dataset from Week 3, reused here as the Week 3 dataset for the three-way split lab. |

## How to Use `Day1.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to implement it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** section is a fast reference you can come back to for the rest of the program.
6. Make sure `train_and_test2.csv` is in the same folder as the notebook before running the
   Hands-On Lab cells.
7. Fill in the **"Your turn"** Markdown cell at the end of the lab with your own explanation of
   why tuning against the test set would be a mistake.

## Lesson Summary

### 1. The Problem With a Single Test Set
In Week 3, the data was split into train and test — honest, as long as the test set is checked
only once. But repeatedly checking the test set while tuning a model quietly fits your decisions
to that specific test set, so the test score stops being an honest estimate of real-world
performance.

### 2. The Three-Way Split
The professional solution is three sets, each with one job:

| Set | Purpose | When It's Used |
|---|---|---|
| Training set | The model learns its parameters from this | During `.fit()` |
| Validation set | Tune choices (model, hyperparameters, features) against this | During development and tuning |
| Test set | Final, one-time, honest performance estimate | Once, at the very end — never touched during tuning |

The rule is strict: the test set is opened exactly once, after every decision is final.

### 3. Creating the Split in Code
A three-way split is done with two calls to `train_test_split`: first carve off the test set,
then split the remaining data into train and validation.
```python
from sklearn.model_selection import train_test_split

# 1) hold out 20% as the final test set
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2) split the rest into train (75%) and validation (25%)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42)
```
Typical proportions: 60% train / 20% validation / 20% test.

### 4. Why This Isn't Always Enough
A single validation set has its own weakness: if it happens to be an unusual slice of the data,
tuning decisions end up based on luck rather than signal. On smaller datasets especially, one
validation split can be misleading — exactly the problem cross-validation (Day 2) solves.

## Hands-On Lab — Building a Three-Way Split

Using the **Titanic passenger dataset** (`train_and_test2.csv`) from Week 3, the lab builds and
uses a correct three-way split end to end:

1. **Take** the Week 3 dataset and create a 60/20/20 train/validation/test split with a fixed
   `random_state`.
2. **Train** a random forest on the training set and tune `max_depth` by checking the validation
   set only.
3. **Evaluate** the final model on the test set exactly once and report the F1-score.
4. **Explain**, in Markdown, what would go wrong if the test set had been used for tuning instead.

## Tools Used

Scikit-learn (`train_test_split`, `RandomForestClassifier`) • Pandas • Jupyter Notebook

## Best Practices & Reproducibility

- Always carve off the test set first, before doing anything else with the remaining data.
- Fix `random_state=42` on both split calls, for a fully reproducible three-way split.
- Make every tuning decision using the validation set, never the test set.
- Check the test set exactly once, after every decision is already final.
- On small datasets, remember that even a good validation score can be a matter of luck.

## Where This Leads Next

Day 1's three-way split is the foundation for the rest of Week 4. **Day 2** replaces the single
validation set with **k-fold cross-validation**, giving a far more stable performance estimate by
averaging over several validation splits instead of trusting just one.


