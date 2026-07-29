# Week 2 — Day 4: EDA Part 1 — Distributions & Outliers

**BinX Tech · AI & Machine Learning Internship Program · Phase 1 → 2 Transition**
**Week 2 of 10 · Math Foundations & EDA · Day 4 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day4.ipynb`](#how-to-use-week2_day4ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. What EDA Is and Why It Comes First](#1-what-eda-is-and-why-it-comes-first)
   - [2. Seaborn — Statistical Visualization](#2-seaborn--statistical-visualization)
   - [3. Univariate Analysis](#3-univariate-analysis)
   - [4. Outlier Detection with the IQR Method](#4-outlier-detection-with-the-iqr-method)
7. [Hands-On Lab — Univariate EDA on a Real Dataset](#hands-on-lab--univariate-eda-on-a-real-dataset)
8. [Deliverables Checklist (Week 2)](#deliverables-checklist-week-2)
9. [Tools Used](#tools-used)
10. [Best Practices & Reproducibility](#best-practices--reproducibility)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 4 of Week 2 begins the **Exploratory Data Analysis (EDA)** portion of the week, bringing
together everything learned so far — statistics (Day 1), probability (Day 2), and math
foundations (Day 3) — into a hands-on, visual investigation of a dataset. Today introduces
**Seaborn** and focuses on **univariate analysis**: understanding one variable at a time through
distributions and outlier detection, before moving to relationships between variables on Day 5.

## Learning Objectives

By the end of Day 4, you should be able to:

- Explain why EDA is a required first step before modeling.
- Perform univariate analysis using Seaborn **histograms**, **box plots**, and **count plots**.
- Detect outliers using the **IQR method** and decide how to handle them.

## Key Topics

- What EDA is and why it comes first
- Seaborn for statistical visualization
- Univariate analysis: histogram, box plot, count plot, KDE
- Outlier detection with the IQR method

## Files in This Folder

| File | Description |
|---|---|
| `Day4.ipynb` | The full, detailed Day 4 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs (histograms, box plots, count plots, and KDE charts). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |

## How to Use `Day4.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to visualize it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs (including all charts) without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** (Section 6) is a fast reference you can come back to for the rest of the program.

## Lesson Summary

### 1. What EDA Is and Why It Comes First
Exploratory Data Analysis is the systematic process of understanding a dataset before modeling —
not optional preliminary work, but the step where data problems (missing values, outliers,
skewed distributions) are caught before they silently corrupt every result downstream.

### 2. Seaborn — Statistical Visualization
Seaborn is built on Matplotlib and specializes in statistical plots, working directly with Pandas
DataFrames:
```python
import seaborn as sns
sns.histplot(data=df, x="age")
```

### 3. Univariate Analysis

| Plot | Seaborn Function | Reveals |
|---|---|---|
| Histogram | `sns.histplot()` | The shape of a numeric variable's distribution |
| Box plot | `sns.boxplot()` | Median, quartiles, and outliers at a glance |
| Count plot | `sns.countplot()` | The frequency of each category in a categorical variable |
| KDE plot | `sns.kdeplot()` | A smoothed estimate of the distribution's shape |

### 4. Outlier Detection with the IQR Method
The standard rule flags any value below `Q1 - 1.5×IQR` or above `Q3 + 1.5×IQR` as a potential
outlier — exactly what a box plot draws as points beyond its whiskers:
```python
Q1 = df["income"].quantile(0.25)
Q3 = df["income"].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df["income"] < Q1 - 1.5*IQR) | (df["income"] > Q3 + 1.5*IQR)]
```
**An outlier is a question, not a verdict** — investigate whether it's real or an error before
deciding to keep, cap, or remove it. Never delete outliers silently.

## Hands-On Lab — Univariate EDA on a Real Dataset

Using a sample "interns" dataset (200 rows, with three deliberately injected outliers), the lab
runs a complete univariate EDA pass:

1. **Histograms** for every numeric variable (`age`, `weekly_hours`, `monthly_stipend`), with KDE overlays.
2. **Box plots** for the same variables, visually confirming all three injected outliers.
3. **IQR-based outlier flagging** in code, with a documented decision for each outlier found —
   including one judged a likely **data-entry error** (a stipend nearly 4× the typical value) and
   two judged plausible real values worth keeping.
4. **Count plot** for the categorical `track` variable, revealing clear **class imbalance**.
5. A written, plain-language summary of what each distribution reveals.

## Deliverables Checklist (Week 2)

By the end of Week 2, submit the following to your mentor and GitHub repository:

- [x] A descriptive-statistics notebook computing and interpreting central tendency and spread for a real dataset *(Day 1)*
- [x] A probability notebook with simulations for coin flips, a normal distribution, and a conditional-probability check *(Day 2)*
- [x] A linear-algebra notebook demonstrating vectors, matrices, the dot product, and matrix multiplication *(Day 3)*
- [x] **A univariate EDA notebook with distributions, box plots, and documented outlier handling** ← this is `Week2_Day4.ipynb`
- [ ] The complete Week 2 EDA notebook (statistics + univariate + bivariate + correlation) with a data-storytelling narrative
- [ ] All Week 2 notebooks committed to the intern's GitHub repository with clear commit messages

## Tools Used

Seaborn • Pandas • Matplotlib • Jupyter Notebook

## Best Practices & Reproducibility

- Always fix a random seed wherever randomness is used: `np.random.seed(42)`.
- Plot **every** numeric column with a histogram/KDE and box plot — don't skip columns that "look fine."
- Plot **every** categorical column with a count plot, and note any class imbalance.
- For every flagged outlier, write down the decision (keep / cap / remove) **and why** — a silent
  deletion is one of the easiest ways to accidentally bias a later model.

## Where This Leads Next

Day 4's univariate foundation — understanding each variable individually — sets up **Day 5's
bivariate analysis and correlation**, where the focus shifts to relationships *between*
variables, culminating in the complete, narrated **Week 2 EDA notebook**: the template used for
the EDA stage of every project through the Phase 3 capstone.

---

*Prepared by BinX Tech · Palestine | Nablus*
