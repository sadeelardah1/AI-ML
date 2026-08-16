# Week 5 — Day 1: Unsupervised Learning & K-Means

**BinX Tech · AI & Machine Learning Internship Program · Phase 2 → Phase 3 Transition**
**Week 5 of 10 · Day 1 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day1.ipynb`](#how-to-use-day1ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. Supervised vs. Unsupervised Learning](#1-supervised-vs-unsupervised-learning)
   - [2. What Clustering Does](#2-what-clustering-does)
   - [3. K-Means, Step by Step](#3-k-means-step-by-step)
   - [4. Choosing k: the Elbow Method](#4-choosing-k-the-elbow-method)
   - [5. Choosing k: the Silhouette Score](#5-choosing-k-the-silhouette-score)
7. [Hands-On Lab — K-Means Clustering](#hands-on-lab--k-means-clustering)
8. [Troubleshooting](#troubleshooting)
9. [Tools Used](#tools-used)
10. [Best Practices & Reproducibility](#best-practices--reproducibility)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Week 5 is the bridge between Phase 2 and Phase 3: after four weeks of supervised learning
(predicting a known target), Day 1 opens **unsupervised learning** — finding structure in data
that has no labels at all. The day centers on **K-Means clustering**, the most widely used
clustering algorithm, and the two standard ways to choose how many clusters to use.

## Learning Objectives

By the end of Day 1, you should be able to:

- **Explain** unsupervised learning and how it differs from supervised learning.
- **Run** K-Means clustering and interpret the resulting clusters and centroids.
- **Choose** the number of clusters `k` using the elbow method and silhouette score.

## Key Topics

- Supervised vs. unsupervised learning
- What clustering does
- K-Means: the centroid-assignment loop
- Choosing k: the elbow method
- Choosing k: the silhouette score
- Scaling before clustering

## Files in This Folder

| File | Description |
|---|---|
| `Day1.ipynb` | The full, detailed Day 1 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs (elbow plot, silhouette comparison, and a 2D cluster scatter plot). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |
| `customer_segmentation.csv` | The dataset used throughout the lesson and lab — customer records with `Age`, `Annual_Income_k`, and `Spending_Score` (no target column, since clustering needs none). |

## How to Use `Day1.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note**, **Important**, or **Tip** box
   explaining *why* a concept matters before showing *how* to implement it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. Make sure `customer_segmentation.csv` is in the same folder as the notebook before running the
   Hands-On Lab cells.
6. Fill in the **"Your turn"** Markdown cell at the end of the lab with your own reading of the
   `cluster_summary` table.

## Lesson Summary

### 1. Supervised vs. Unsupervised Learning
In supervised learning (Weeks 3–4), every training example had a known correct answer (the
target `y`). Unsupervised learning works on data with no labels at all — there is no `y`. Instead
of predicting a known answer, the goal is to discover hidden structure the data contains on its
own: natural groupings, underlying dimensions, or unusual points.

| | Supervised | Unsupervised |
|---|---|---|
| Data | Has labels (X and y) | No labels (X only) |
| Goal | Predict the known target | Discover hidden structure |
| Examples | Regression, classification | Clustering, dimensionality reduction, anomaly detection |
| Evaluation | Compare prediction to true label | No ground truth — uses internal metrics and judgment |

### 2. What Clustering Does
Clustering groups data points so that points in the same group are similar to each other and
different from points in other groups — answering questions like *"what natural customer segments
exist?"* with no pre-defined segments.

### 3. K-Means, Step by Step
K-Means partitions data into a chosen number of clusters (`k`) by repeating:
1. Place `k` cluster centers ("centroids"), initially at random.
2. Assign each point to its nearest centroid.
3. Move each centroid to the mean position of the points assigned to it.
4. Repeat steps 2–3 until the centroids stop moving.

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = km.fit_predict(X)
print(km.cluster_centers_)   # the final centroid positions
```
**Always scale features before clustering** — K-Means uses distance, so an unscaled large-range
feature (like income) would dominate a small-range one (like age).

### 4. Choosing k: the Elbow Method
Runs K-Means for a range of `k` values and plots the inertia (total within-cluster distance)
against `k`. Inertia always falls as `k` rises, but the rate of improvement drops sharply at the
right `k` — creating an "elbow" in the plot.
```python
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    inertias.append(km.inertia_)
plt.plot(range(1, 11), inertias, marker="o")
```

### 5. Choosing k: the Silhouette Score
A more quantitative check: measures how well each point sits inside its own cluster versus the
nearest other cluster, on a scale from -1 to +1. A higher average silhouette score means
better-defined clusters — used to confirm (or override) the elbow's suggestion.
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, labels)
```

## Hands-On Lab — K-Means Clustering

Using the **customer segmentation dataset** (`customer_segmentation.csv`), the lab runs a
complete K-Means workflow:

1. **Load and scale** the dataset (numeric features only) with `StandardScaler`.
2. **Run K-Means** for `k` from 1 to 10 and plot inertia to find the elbow.
3. **Compute the silhouette score** for the top candidate `k` values and pick the best.
4. **Fit the final model** with the chosen `k` and visualize the clusters on a 2D scatter plot.
5. **Interpret** what each cluster represents — a "Your turn" cell to name each segment in plain
   language using the `cluster_summary` table.

## Troubleshooting

- **`FileNotFoundError` on `customer_segmentation.csv`** — the CSV must sit in the same folder as
  the notebook.
- **`UserWarning: KMeans is known to have a memory leak on Windows with MKL...`** — this is a
  harmless warning from scikit-learn on Windows/Anaconda, not an error; the notebook still runs
  and produces correct results. To silence it, add this to the very first cell (before any other
  imports) and then **restart the kernel** so the setting takes effect:
  ```python
  import os
  os.environ["OMP_NUM_THREADS"] = "1"

  import warnings
  warnings.filterwarnings("ignore", category=UserWarning)
  ```

## Tools Used

Scikit-learn (`KMeans`, `StandardScaler`, `silhouette_score`) • Pandas • Matplotlib • Jupyter Notebook

## Best Practices & Reproducibility

- Always scale features with `StandardScaler` before running K-Means — it is entirely
  distance-based.
- Fix `random_state=42` (and `n_init=10`) so clustering results are reproducible.
- Never pick `k` from the elbow plot alone — confirm the choice with the silhouette score.
- Interpret clusters by their **feature averages**, not by the arbitrary cluster label number.
- Keep the elbow and silhouette sweeps in the notebook as evidence for the `k` that was chosen.

## Where This Leads Next

Day 1 establishes K-Means as the baseline clustering method. **Day 2** introduces **DBSCAN** and
**hierarchical clustering** — two alternatives that do not require choosing `k` in advance and
handle irregularly shaped clusters and noise more gracefully, then compares all three methods on
the same dataset.

