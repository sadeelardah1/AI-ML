# Week 2 — Day 1: Descriptive Statistics

**BinX Tech · AI & Machine Learning Internship Program · Phase 1 → 2 Transition**
**Week 2 of 10 · Math Foundations & EDA · Day 1 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Week2_Day1.ipynb`](#how-to-use-week2_day1ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. Why Statistics Comes Before Modeling](#1-why-statistics-comes-before-modeling)
   - [2. Measures of Central Tendency](#2-measures-of-central-tendency)
   - [3. Mean vs. Median with Outliers](#3-mean-vs-median-with-outliers)
   - [4. Measures of Spread](#4-measures-of-spread)
   - [5. Percentiles, Quartiles & the IQR](#5-percentiles-quartiles--the-iqr)
   - [6. Pandas `.describe()`](#6-pandas-describe)
7. [Hands-On Lab — Describing a Real Dataset](#hands-on-lab--describing-a-real-dataset)
8. [Deliverables Checklist (Week 2)](#deliverables-checklist-week-2)
9. [Tools Used](#tools-used)
10. [Best Practices & Reproducibility](#best-practices--reproducibility)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 1 of Week 2 opens **Phase 1 → 2: Math Foundations & EDA**, the mathematical bridge between
plain data handling (Week 1) and machine learning. Today's topic is **descriptive statistics** —
the vocabulary used to describe a dataset precisely: where it's centered, how spread out it is,
and how outliers distort each measure differently. This vocabulary underpins every Exploratory
Data Analysis (EDA) and every model-evaluation step later in the 400-hour program.

## Learning Objectives

By the end of Day 1, you should be able to:

- Compute the **mean**, **median**, and **mode**, and choose the appropriate one for a given dataset.
- Compute and interpret **variance**, **standard deviation**, and the **IQR**.
- Explain how **outliers** affect each measure differently.
- Use Pandas' `.describe()` to get a full statistical summary of a column in one line.

## Key Topics

- Why descriptive statistics precedes modeling
- Central tendency: mean, median, mode
- Spread: range, variance, standard deviation, IQR
- Percentiles and quartiles

## Files in This Folder

| File | Description |
|---|---|
| `Week2_Day1.ipynb` | The full, detailed Day 1 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs. |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |

## How to Use `Week2_Day1.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to compute it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** (Section 8) is a fast reference you can come back to for the rest of the program.

## Lesson Summary

### 1. Why Statistics Comes Before Modeling
A machine learning model is, at its core, a summary of patterns in data. Before judging whether a
model's behavior is reasonable, you first need to describe the data itself — its center, its
spread, its shape.

### 2. Measures of Central Tendency

| Measure | What It Is | When to Use It |
|---|---|---|
| Mean | The arithmetic average | Symmetric data with no extreme outliers |
| Median | The middle value when sorted | Skewed data or data with outliers (more robust) |
| Mode | The most frequent value | Categorical data, or finding the most common value |

### 3. Mean vs. Median with Outliers
The key insight of the day: **the mean is pulled toward outliers, while the median is not.**
```python
data = np.array([10, 12, 12, 13, 100])
np.mean(data)     # 29.4  — pulled up by the outlier
np.median(data)   # 12.0  — unaffected
```

### 4. Measures of Spread

| Measure | Meaning |
|---|---|
| Range | Maximum minus minimum — simple but very sensitive to outliers |
| Variance | The average squared distance from the mean |
| Standard deviation | The square root of variance — spread in the same units as the data |
| IQR (interquartile range) | The range of the middle 50% of the data (Q3 − Q1), robust to outliers |

### 5. Percentiles, Quartiles & the IQR
A percentile marks the value below which a given percentage of the data falls. **Q1** (25th
percentile), **Q2/median** (50th), and **Q3** (75th) split the data into quarters — these are
exactly the values a **box plot** visualizes, which is why they're central to the outlier
detection covered on **Day 4**.
```python
q1, q3 = np.percentile(data, [25, 75])
iqr = q3 - q1
```

### 6. Pandas `.describe()`
A single-line shortcut that computes count, mean, std, min, Q1, median, Q3, and max at once:
```python
df["column"].describe()
```

## Hands-On Lab — Describing a Real Dataset

The lab builds a small sample "interns" dataset with weekly working hours (including one
realistic outlier — a 65-hour week) and runs through the full workflow:

1. **Load** a numeric column (`weekly_hours`) into a Pandas Series.
2. **Compute** its mean, median, mode, standard deviation, and IQR.
3. **Compare** the mean and median, and **justify** which one better represents a "typical"
   value — with the outlier explicitly identified and explained, not ignored.
4. **Summarize** the column's center and spread in a plain-language Markdown paragraph, the way
   a real analysis would communicate findings to a non-technical reader.

## Deliverables Checklist (Week 2)

By the end of Week 2, submit the following to your mentor and GitHub repository:

- [x] **A descriptive-statistics notebook computing and interpreting central tendency and spread for a real dataset** ← this is `Week2_Day1.ipynb`
- [ ] A probability notebook with simulations for coin flips, a normal distribution, and a conditional-probability check
- [ ] A linear-algebra notebook demonstrating vectors, matrices, the dot product, and matrix multiplication
- [ ] A univariate EDA notebook with distributions, box plots, and documented outlier handling
- [ ] The complete Week 2 EDA notebook (statistics + univariate + bivariate + correlation) with a data-storytelling narrative
- [ ] All Week 2 notebooks committed to the intern's GitHub repository with clear commit messages

## Tools Used

NumPy • Pandas • Jupyter Notebook

## Best Practices & Reproducibility

- Always fix a random seed wherever randomness is used: `np.random.seed(42)`.
- Never report a "typical value" (mean or median) without also reporting a spread measure
  (standard deviation or IQR) alongside it.
- Always check whether the mean and median disagree significantly — that gap is itself a signal
  that outliers or skew may be present.
- Investigate outliers before deciding how to handle them — never delete or ignore them silently.

## Where This Leads Next

Day 1's statistics vocabulary is the foundation for the rest of Week 2: **Day 2 (Probability &
Distributions)** builds directly on these ideas, **Day 4 (EDA Part 1)** uses the exact Q1/median/Q3
values from today to detect outliers with box plots, and **Day 5 (EDA Part 2)** assembles
everything into a complete, narrated Exploratory Data Analysis — the template used for every
project through the Phase 3 capstone.

---

*Prepared by BinX Tech · Palestine | Nablus*
