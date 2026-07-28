# Week 2 — Day 3: Linear Algebra for ML

**BinX Tech · AI & Machine Learning Internship Program · Phase 1 → 2 Transition**
**Week 2 of 10 · Math Foundations & EDA · Day 3 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Week2_Day3.ipynb`](#how-to-use-Day3ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. Why Linear Algebra Is the Language of ML](#1-why-linear-algebra-is-the-language-of-ml)
   - [2. Vectors](#2-vectors)
   - [3. Matrices](#3-matrices)
   - [4. The Dot Product](#4-the-dot-product)
   - [5. Matrix Multiplication](#5-matrix-multiplication)
   - [6. The Shape-Matching Rule](#6-the-shape-matching-rule)
7. [Hands-On Lab — Vectors, Matrices & Predictions](#hands-on-lab--vectors-matrices--predictions)
8. [Deliverables Checklist (Week 2)](#deliverables-checklist-week-2)
9. [Tools Used](#tools-used)
10. [Best Practices & Reproducibility](#best-practices--reproducibility)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 3 of Week 2 completes the mathematical foundations portion of **Phase 1 → 2: Math
Foundations & EDA** with **linear algebra** — the language ML models are literally built from.
Every dataset is a matrix, every model's parameters are vectors or matrices, and every
prediction a model makes is fundamentally a dot product. Today builds that understanding from
the ground up, ending with a deliberately broken example that turns a common, intimidating error
into a well-understood, fixable mistake.

## Learning Objectives

By the end of Day 3, you should be able to:

- Represent data samples as **vectors** and datasets as **matrices**.
- Compute a **dot product** and explain why it is central to model prediction.
- Perform **matrix multiplication** and reason about resulting shapes.

## Key Topics

- Why linear algebra is the language of ML
- Vectors: one sample's features
- Matrices: a full dataset (samples × features)
- The dot product and how models predict with it
- Matrix multiplication and the shape-matching rule

## Files in This Folder

| File | Description |
|---|---|
| `Week2_Day3.ipynb` | The full, detailed Day 3 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs, including a deliberately triggered and explained `ValueError`. |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |

## How to Use `Week2_Day3.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to compute it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** (Section 8) is a fast reference you can come back to for the rest of the program.

## Lesson Summary

### 1. Why Linear Algebra Is the Language of ML
Every dataset in ML is a matrix: rows are samples, columns are features. Every model's
parameters are vectors or matrices, and training is a sequence of matrix operations — this is
literally what a model is doing internally.

### 2. Vectors
A vector is an ordered list of numbers — usually one data sample's features:
```python
v = np.array([25, 50000, 3])   # a customer: age, income, tenure
```

### 3. Matrices
A matrix is a 2D grid of numbers — a full dataset, where each row is a sample and each column is
a feature. Its shape is `(rows, columns) = (samples, features)`.
```python
X = np.array([[25, 50000, 3], [40, 80000, 10], [33, 62000, 5]])
X.shape   # (3, 3): 3 samples, 3 features
```

### 4. The Dot Product
Multiplies two vectors element-by-element and sums the result — the single most important
operation in ML, exactly how a linear model computes one prediction:
```python
prediction = np.dot(features, weights) + bias
```

### 5. Matrix Multiplication
Applies the dot product across a whole matrix at once — one prediction per sample, in a single
operation, no loop required:
```python
predictions = X @ w + bias
```

### 6. The Shape-Matching Rule
An `(m × n)` matrix times an `(n × p)` matrix gives an `(m × p)` matrix — **the inner dimensions
must match**. This is why shape mismatches are the most common bug in ML code. The notebook
deliberately triggers a `ValueError` from a mismatched weight vector, then walks through reading
the error message and fixing it.

## Hands-On Lab — Vectors, Matrices & Predictions

The lab builds a tiny "linear model" from scratch, predicting exam scores for three interns from
three features (`hours_studied`, `previous_score`, `hours_slept`):

1. **Represent** three data samples as a `(3 × 3)` NumPy matrix.
2. **Compute a dot product by hand**, for one sample against a weight vector, then **verify** it
   matches `np.dot()`.
3. **Use matrix multiplication (`@`)** to produce predictions for all three samples in one step.
4. **Deliberately trigger a shape-mismatch error** with a wrong-sized weight vector, read the
   `ValueError` message, and explain in Markdown exactly why it occurred and how to fix it.

## Deliverables Checklist (Week 2)

By the end of Week 2, submit the following to your mentor and GitHub repository:

- [x] A descriptive-statistics notebook computing and interpreting central tendency and spread for a real dataset *(Day 1)*
- [x] A probability notebook with simulations for coin flips, a normal distribution, and a conditional-probability check *(Day 2)*
- [x] **A linear-algebra notebook demonstrating vectors, matrices, the dot product, and matrix multiplication for prediction** ← this is `Week2_Day3.ipynb`
- [ ] A univariate EDA notebook with distributions, box plots, and documented outlier handling
- [ ] The complete Week 2 EDA notebook (statistics + univariate + bivariate + correlation) with a data-storytelling narrative
- [ ] All Week 2 notebooks committed to the intern's GitHub repository with clear commit messages

## Tools Used

NumPy • Jupyter Notebook

## Best Practices & Reproducibility

- Always print `.shape` for every array involved **before** multiplying — catching a mismatch
  early saves debugging time later.
- Know which axis represents samples and which represents features; comment it if it isn't obvious.
- Remember the shape rule as a simple phrase: *"the inner numbers must match, the outer numbers survive."*
- When something looks wrong in ML code, check shapes first — the vast majority of bugs trace
  back to a shape mismatch somewhere upstream.

## Where This Leads Next

Day 3's linear algebra foundation is what makes **Week 3's linear and logistic regression**
models fully understandable — their predictions are exactly `X @ weights + bias`, the same
formula built from scratch in today's Hands-On Lab. Together with Day 1 (statistics) and Day 2
(probability), this completes the math toolkit needed before Week 2 shifts into full
**Exploratory Data Analysis on Days 4 and 5**.




