# Week 2 — Day 5: EDA Part 2 — Correlation & Data Storytelling

**BinX Tech · AI & Machine Learning Internship Program · Phase 1 → 2 Transition**
**Week 2 of 10 · Math Foundations & EDA · Day 5 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day5.ipynb`](#how-to-use-day5ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. Bivariate Analysis](#1-bivariate-analysis)
   - [2. Correlation & the Correlation Heatmap](#2-correlation--the-correlation-heatmap)
   - [3. Correlation Is Not Causation](#3-correlation-is-not-causation)
   - [4. The Pairplot](#4-the-pairplot)
   - [5. Data Storytelling](#5-data-storytelling)
7. [Hands-On Lab — Complete EDA Notebook](#hands-on-lab--complete-eda-notebook)
8. [Deliverables Checklist (Week 2 — Complete)](#deliverables-checklist-week-2--complete)
9. [Tools Used](#tools-used)
10. [Best Practices & Reproducibility](#best-practices--reproducibility)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 5 closes **Week 2: Math Foundations & EDA** with the second half of Exploratory Data
Analysis — **bivariate analysis** and **correlation**. Where Day 4 examined variables one at a
time, today examines relationships *between* variables, then assembles everything from the
entire week (statistics, probability-informed thinking, linear algebra, univariate analysis, and
now correlation) into one complete, narrated EDA notebook — the template used for the EDA stage
of every project through the Phase 3 capstone.

## Learning Objectives

By the end of Day 5, you should be able to:

- Perform **bivariate analysis** with scatter plots and grouped box plots.
- Compute and interpret a **correlation matrix** and **heatmap**.
- Assemble a **complete, narrated EDA notebook** on a real dataset.

## Key Topics

- Bivariate analysis: scatter plots, grouped box plots
- Correlation and the correlation heatmap
- Correlation is not causation
- The pairplot for scanning relationships
- Data storytelling: turning analysis into a narrative

## Files in This Folder

| File | Description |
|---|---|
| `Day5.ipynb` | The full, detailed Day 5 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs (scatter plots, grouped box plots, a correlation heatmap, and a pairplot). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |

## How to Use `Day5.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to compute or visualize it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs (including all charts) without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** (Section 7) is a fast reference you can come back to for the rest of the program.

## Lesson Summary

### 1. Bivariate Analysis
Examines the relationship between two variables:
```python
sns.scatterplot(data=df, x="age", y="income")          # two numeric variables
sns.boxplot(data=df, x="category", y="income")          # numeric variable across categories
```

### 2. Correlation & the Correlation Heatmap
Correlation measures how strongly two numeric variables move together, on a scale from -1 to +1:
```python
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
```

### 3. Correlation Is Not Causation
Two variables moving together does not mean one causes the other — EDA identifies
relationships, it never proves cause. The classic "ice cream sales vs. shark attacks" example
illustrates how a hidden **confounding variable** (hot weather) can explain both.

### 4. The Pairplot
`sns.pairplot()` plots every numeric variable against every other in a single grid, with
distributions on the diagonal — a fast way to scan an entire dataset's relationships at once.

### 5. Data Storytelling
EDA produces charts, but the deliverable is **understanding, communicated clearly**. A finding
nobody understands is not a finding — every chart should be paired with a plain-language
explanation of what it shows and what it implies.

## Hands-On Lab — Complete EDA Notebook

Using the same sample "interns" dataset from Day 4 (rebuilt with the identical random seed, plus
a genuine relationship deliberately added between hours worked and stipend), the lab assembles a
full EDA:

1. **Scatter plots and grouped box plots** for the most important variable pairs.
2. **Correlation matrix and annotated heatmap** for all numeric variables.
3. **Identifying the strongest relationships** and reasoning about what they might mean for a
   future model — including an explicit caution against assuming causation.
4. **Assembling the complete EDA** — statistics, outlier checks, and correlation — into one
   printed summary, followed by a full written **data-storytelling narrative**.
5. **Commit instructions** for pushing the finished notebook to GitHub.

## Deliverables Checklist (Week 2 — Complete)

By the end of Week 2, submit the following to your mentor and GitHub repository:

- [x] A descriptive-statistics notebook computing and interpreting central tendency and spread for a real dataset *(Day 1)*
- [x] A probability notebook with simulations for coin flips, a normal distribution, and a conditional-probability check *(Day 2)*
- [x] A linear-algebra notebook demonstrating vectors, matrices, the dot product, and matrix multiplication *(Day 3)*
- [x] A univariate EDA notebook with distributions, box plots, and documented outlier handling *(Day 4)*
- [x] **The complete Week 2 EDA notebook (statistics + univariate + bivariate + correlation) with a data-storytelling narrative** ← this is `Day5.ipynb`
- [ ] All Week 2 notebooks committed to the intern's GitHub repository with clear commit messages

## Tools Used

Seaborn • Pandas • Matplotlib • Jupyter Notebook • Git & GitHub

## Best Practices & Reproducibility

- Always fix a random seed wherever randomness is used: `np.random.seed(42)` — the **same** seed
  as Day 4, so both notebooks describe the exact same dataset.
- Look at the scatter plot, not just the correlation number — a non-linear pattern can hide from
  a correlation coefficient entirely.
- Check for outliers before trusting a correlation value — they can drastically distort it.
- Never state or imply causation from a correlation alone.
- End every EDA with a short, plain-language narrative — the charts support the story; they
  aren't the story themselves.

## Where This Leads Next

With Week 2 complete, all five math and EDA foundations — statistics, probability, linear
algebra, and a full two-part EDA — are in place. This is the exact template that will be reused
for every dataset explored for the rest of the 400-hour program, right through **Week 3's linear
and logistic regression models** (which literally compute `X @ weights + bias`, the formula built
from scratch on Day 3) and all the way to the **Phase 3 capstone project**.

---

*Prepared by BinX Tech · Palestine | Nablus*
