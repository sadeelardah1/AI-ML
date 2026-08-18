# Week 5 — Day 3: Dimensionality Reduction with PCA

**BinX Tech · AI & Machine Learning Internship Program · Phase 2 → Phase 3 Transition**
**Week 5 of 10 · Day 3 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day3.ipynb`](#how-to-use-day3ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. The Curse of Dimensionality](#1-the-curse-of-dimensionality)
   - [2. What PCA Does](#2-what-pca-does)
   - [3. Explained Variance](#3-explained-variance)
   - [4. When (and When Not) to Use PCA](#4-when-and-when-not-to-use-pca)
7. [Hands-On Lab — Reducing Dimensions with PCA](#hands-on-lab--reducing-dimensions-with-pca)
8. [Lab Results](#lab-results)
9. [Tools Used](#tools-used)
10. [Best Practices & Reproducibility](#best-practices--reproducibility)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 3 moves from **clustering** (Days 1-2) to **dimensionality reduction**. Real datasets often have far more
features than can be plotted or reasoned about directly. **PCA (Principal Component Analysis)** compresses
many features into a handful of new axes — principal components — that capture as much of the data's
variance as possible. The day closes by running PCA on the full **heart.csv** dataset (every clinical
feature, not just the five numeric ones from Day 2), to see how much a real, mixed numeric-and-categorical
dataset can actually be compressed.

## Learning Objectives

- Explain the curse of dimensionality and why reduction helps.
- Apply PCA to reduce a dataset's dimensions.
- Interpret explained variance and choose how many components to keep.

## Key Topics

- The curse of dimensionality
- What PCA does: principal components and variance
- Explained variance ratio
- Choosing the number of components (e.g. 95% variance)
- When (and when not) to use PCA

## Files in This Folder

| File | Description |
|---|---|
| `Day3.ipynb` | The full, detailed Day 3 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs (a cumulative explained-variance plot and a 2D PCA scatter plot colored by diagnosis). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |
| `heart.csv` | The same real heart-failure dataset from Day 2 (918 patients, 11 clinical features + a `HeartDisease` diagnosis column), reused here for the PCA lab. |

## How to Use `Day3.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note**, **Important**, or **Tip** box explaining
   *why* a concept matters before showing *how* to implement it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can also just
   read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. Make sure `heart.csv` is in the same folder as the notebook before running the Hands-On Lab cells.
6. The notebook's colors follow a colorblind-safe palette (the Okabe–Ito palette), and every colored box is
   also labeled in words (Note / Tip / Important / Goal), so meaning never depends on color alone.

## Lesson Summary

### 1. The Curse of Dimensionality
Real datasets often have dozens or hundreds of features. High dimensionality causes real problems: data
becomes sparse, distances lose meaning, models overfit more easily, and you cannot visualize beyond three
dimensions. Dimensionality reduction compresses many features into a few, keeping as much information as
possible.

### 2. What PCA Does
PCA finds new axes — **principal components** — that capture the directions of greatest variance in the
data. The first component captures the most variance, the second the next most, and so on.
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)   # PCA requires scaling
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

### 3. Explained Variance
The key output is the explained variance ratio — how much of the data's total information each component
keeps. A common rule is to keep enough components to retain about 95% of the variance.
```python
print(pca.explained_variance_ratio_)          # variance kept per component
print(pca.explained_variance_ratio_.sum())    # total variance retained
```

### 4. When (and When Not) to Use PCA

| Use | What It Does For You |
|---|---|
| Speed & stability | Fewer input features means faster training and a more stable model |
| Reduce overfitting | Removing redundant / correlated features reduces noise a model could latch onto |
| Visualization | Reducing data to 2D or 3D lets you actually plot and look at it |

The trade-off: the new components are combinations of the original features, so they lose the direct
interpretability the original columns had.

## Hands-On Lab — Reducing Dimensions with PCA

Using the **heart-failure dataset** (`heart.csv`), the lab reduces **all 11 input features** — not just the
5 numeric ones used for clustering — down to a handful of principal components:

1. **Clean the data**: the same `0`-as-missing issue from Day 2 (in `RestingBP` and `Cholesterol`) is
   fixed with median imputation before anything else.
2. **One-hot encode** the categorical columns (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`,
   `ST_Slope`) so every feature is numeric — this is what makes the dataset genuinely high-dimensional.
3. **Scale** the encoded dataset with `StandardScaler`.
4. **Fit PCA** keeping all components, and plot the **cumulative explained variance** against the number of
   components.
5. **Choose** the number of components that retains ~95% of the variance, and justify it in Markdown.
6. **Reduce to 2 components**, plot a 2D scatter, and color the points by the known `HeartDisease` label
   (used only afterward, for interpretation — never for the PCA fit itself).
7. **Document**, in Markdown, what the reduction preserved and what it cost.

## Lab Results

On this run: the dataset's 11 original input features expand to **15 numeric columns** after one-hot
encoding. Reaching the standard **95% explained-variance threshold required 13 of those 15 components** —
almost no compression at all. Reducing all the way down to just **2 components for plotting retained only
about 33% of the total variance**.

This is an intentionally realistic (and useful) result, not a disappointing one: it shows that **PCA
compresses best on continuous, correlated measurements**, and much less on one-hot-encoded categorical
flags, which tend to be largely independent of one another. The 2D scatter plot, colored by diagnosis,
shows the two `HeartDisease` groups **overlapping heavily** rather than separating into clean clusters —
confirming that 2 components are not enough to capture what distinguishes a diagnosis here. For an actual
classification task, keeping far more components (or skipping PCA entirely and using the original features,
as in Weeks 3-4) would be the better choice; today's 2D reduction is best understood as a visualization
tool, not a finished feature set.

## Tools Used

Scikit-learn (`PCA`, `StandardScaler`) • Pandas • NumPy • Matplotlib • Jupyter Notebook

## Best Practices & Reproducibility

- Always scale features with `StandardScaler` before PCA — it is a variance-based method, so scale matters
  even more here than in clustering.
- Check the cumulative explained-variance plot before picking a component count — do not default to "2,
  because I can plot it" unless visualization is genuinely the goal.
- Remember principal components are combinations of the original features — never interpret Component 1 as
  if it were a single original column.
- If a label exists (like `HeartDisease` here), it is fine to color a PCA plot by it *afterward* for
  interpretation — PCA itself never sees the label, exactly as with clustering in Days 1-2.
- Fix a random seed (`random_state=42`) wherever PCA or any other randomized step is used, so results are
  reproducible for your mentor.

## Where This Leads Next

Day 3 completes the "compress and understand" half of unsupervised learning: clustering finds groups,
PCA finds the directions that matter most. **Day 4** moves to **t-SNE** — a reduction technique built purely
for visualization — and **anomaly detection** with Isolation Forest, addressing data points that do not fit
any group at all.
