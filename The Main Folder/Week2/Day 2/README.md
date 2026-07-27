# Week 2 — Day 2: Probability & Distributions

**BinX Tech · AI & Machine Learning Internship Program · Phase 1 → 2 Transition**
**Week 2 of 10 · Math Foundations & EDA · Day 2 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Week2_Day2.ipynb`](#how-to-use-week2_day2ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. Probability Basics](#1-probability-basics)
   - [2. Core Rules](#2-core-rules)
   - [3. Conditional Probability](#3-conditional-probability)
   - [4. Bayes' Theorem](#4-bayes-theorem)
   - [5. Common Distributions](#5-common-distributions)
7. [Hands-On Lab — Probability & Distributions in Code](#hands-on-lab--probability--distributions-in-code)
8. [Deliverables Checklist (Week 2)](#deliverables-checklist-week-2)
9. [Tools Used](#tools-used)
10. [Best Practices & Reproducibility](#best-practices--reproducibility)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 2 of Week 2 continues **Phase 1 → 2: Math Foundations & EDA** with **probability** — the
mathematics of uncertainty, and the language ML models speak whenever they output a prediction
like "85% likely to churn." Today covers the core rules of probability, conditional probability,
Bayes' theorem, and the three probability distributions you will meet constantly throughout the
program.

## Learning Objectives

By the end of Day 2, you should be able to:

- Apply the **complement**, **addition**, and **multiplication** rules of probability.
- Explain **conditional probability** and **Bayes' theorem**, and where they appear in ML.
- Recognize the **normal**, **binomial**, and **uniform** distributions.

## Key Topics

- Probability basics: favorable outcomes over total outcomes
- Core rules: complement, addition, multiplication
- Conditional probability `P(A|B)`
- Bayes' theorem: prior, likelihood, posterior
- Common distributions: normal, binomial, uniform

## Files in This Folder

| File | Description |
|---|---|
| `Week2_Day2.ipynb` | The full, detailed Day 2 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs (including all simulation histograms). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |

## How to Use `Week2_Day2.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to compute or simulate it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs (including the distribution charts) without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** (Section 7) is a fast reference you can come back to for the rest of the program.

## Lesson Summary

### 1. Probability Basics
Probability quantifies uncertainty, always a number between 0 (impossible) and 1 (certain):
```python
P(event) = favorable_outcomes / total_outcomes
```

### 2. Core Rules

| Rule | Statement | Meaning |
|---|---|---|
| Complement | `P(not A) = 1 - P(A)` | The chance something does NOT happen |
| Addition | `P(A or B) = P(A) + P(B) - P(A and B)` | The chance of either event |
| Multiplication (independent) | `P(A and B) = P(A) × P(B)` | Both happen, when unrelated |

### 3. Conditional Probability
`P(A | B)` is the probability of A given that B has already happened — the foundation of nearly
all predictive modeling ("given these features, what is the probability of churn?"):
```python
P(A | B) = P(A and B) / P(B)
```

### 4. Bayes' Theorem
Reverses a conditional probability, combining a **prior** belief with **evidence** (likelihood)
to get an updated **posterior** belief:
```python
P(A | B) = ( P(B | A) × P(A) ) / P(B)
```
The notebook works through a classic medical-testing example, showing the surprising result that
a "95% accurate" test can still leave only a ~16% real chance of having a rare disease after a
positive result — a powerful demonstration of why the base rate (prior) matters so much.

### 5. Common Distributions

| Distribution | Describes | Example |
|---|---|---|
| Normal (Gaussian) | Symmetric, bell-shaped data clustered around a mean | Human heights, measurement error |
| Binomial | Number of successes in a fixed number of yes/no trials | Number of heads in 10 coin flips |
| Uniform | Every outcome equally likely | A fair die roll |

The normal distribution matters most — many statistical methods and ML assumptions rely on data
being approximately normal, which is one of the first things EDA checks for (Day 4).

## Hands-On Lab — Probability & Distributions in Code

The lab confirms today's theory with simulation — probability theory predicts what *should*
happen on average, and simulation verifies it actually does:

1. **Simulate 10,000 coin flips** with NumPy and confirm the proportion of heads approaches 0.5.
2. **Sample from a normal distribution** with `np.random.normal` and plot its histogram to
   confirm the bell shape (simulated exam scores, mean=75, std=8).
3. **Compute a conditional probability by hand** (a dice-rolling scenario) and **verify it with a
   200,000-roll simulation** — the hand calculation and the simulated result match almost exactly.
4. **Document each result** in plain language, explaining what it demonstrates.

## Deliverables Checklist (Week 2)

By the end of Week 2, submit the following to your mentor and GitHub repository:

- [x] A descriptive-statistics notebook computing and interpreting central tendency and spread for a real dataset *(Day 1)*
- [x] **A probability notebook with simulations for coin flips, a normal distribution, and a conditional-probability check** ← this is `Week2_Day2.ipynb`
- [ ] A linear-algebra notebook demonstrating vectors, matrices, the dot product, and matrix multiplication
- [ ] A univariate EDA notebook with distributions, box plots, and documented outlier handling
- [ ] The complete Week 2 EDA notebook (statistics + univariate + bivariate + correlation) with a data-storytelling narrative
- [ ] All Week 2 notebooks committed to the intern's GitHub repository with clear commit messages

## Tools Used

NumPy • Matplotlib • Jupyter Notebook

## Best Practices & Reproducibility

- Always fix a random seed wherever randomness is used: `np.random.seed(42)`.
- Double-check whether events are truly **independent** before using the simple multiplication rule.
- Be careful never to confuse `P(A|B)` with `P(B|A)` — they are not the same, and mixing them up
  is one of the most common statistical errors.
- When a probability calculation feels uncertain, **simulate** the scenario with NumPy to
  sanity-check the formula-based answer, exactly as done in the Hands-On Lab.

## Where This Leads Next

Day 2's probability foundation feeds directly into **Day 3 (Linear Algebra for ML)**, where
vectors and matrices become the objects ML models actually compute with, and later into
**Day 4–5 (EDA)**, where recognizing distribution shapes (normal vs. skewed) guides how outliers
and data transformations are handled. Conditional probability and Bayes' theorem also resurface
directly in classification models (like Naive Bayes) later in the program.

---

*Prepared by BinX Tech · Palestine | Nablus*
