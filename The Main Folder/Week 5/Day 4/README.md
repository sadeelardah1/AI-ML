# Week 5 — Day 4: t-SNE & Anomaly Detection

**BinX Tech · AI & Machine Learning Internship Program · Phase 2 → Phase 3 Transition**
**Week 5 of 10 · Day 4 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day4.ipynb`](#how-to-use-day4ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. t-SNE for Visualization](#1-t-sne-for-visualization)
   - [2. PCA vs. t-SNE](#2-pca-vs-t-sne)
   - [3. What Anomaly Detection Is](#3-what-anomaly-detection-is)
   - [4. Isolation Forest](#4-isolation-forest)
7. [Hands-On Lab — Visualization & Anomaly Detection](#hands-on-lab--visualization--anomaly-detection)
8. [Lab Results](#lab-results)
9. [Tools Used](#tools-used)
10. [Best Practices & Reproducibility](#best-practices--reproducibility)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 4 closes out the visualization and detection half of unsupervised learning. **t-SNE** takes the
high-dimensional heart-disease dataset from Day 3 and lays it out in 2D based on local neighborhoods rather
than global variance — often revealing cluster shapes that PCA's straight-line compression cannot.
**Isolation Forest** then scans the same data for individual patients who do not fit the dense mass of
"normal" cases at all, extending the noise-detection idea first seen with DBSCAN on Day 2.

## Learning Objectives

- Use t-SNE to visualize high-dimensional data and distinguish it from PCA.
- Explain what anomaly detection is and why it is often unsupervised.
- Detect anomalies with Isolation Forest and interpret the flagged points.

## Key Topics

- t-SNE for local-structure visualization
- PCA vs. t-SNE: when to use each
- What anomaly detection is
- Isolation Forest and the contamination parameter
- Anomaly detection and clustering overlap

## Files in This Folder

| File | Description |
|---|---|
| `Day4.ipynb` | The full, detailed Day 4 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs (a t-SNE scatter plot, a side-by-side PCA vs. t-SNE comparison, an anomaly-highlighted plot, and inspection of two flagged patients). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |
| `heart.csv` | The same real heart-failure dataset from Days 2-3 (918 patients, 11 clinical features + a `HeartDisease` diagnosis column), reused here for the t-SNE and anomaly-detection lab. |

## How to Use `Day4.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note**, **Important**, or **Tip** box explaining
   *why* a concept matters before showing *how* to implement it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can also just
   read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. Make sure `heart.csv` is in the same folder as the notebook before running the Hands-On Lab cells.
6. The notebook uses a colorblind-safe palette (the IBM Design Library palette) — a different color set
   from Days 2 and 3, but built on the same rule: every colored box is also labeled in words (Note / Tip /
   Important / Goal), so meaning never depends on color alone.

## Lesson Summary

### 1. t-SNE for Visualization
t-SNE (t-distributed Stochastic Neighbor Embedding) is a dimensionality-reduction technique built
specifically for visualization. Unlike PCA, which preserves global variance, t-SNE preserves local
neighborhoods — it keeps points that were close together in high dimensions close together in 2D.
```python
from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)
```

### 2. PCA vs. t-SNE

| | PCA | t-SNE |
|---|---|---|
| Preserves | Global structure / variance | Local neighborhoods |
| Main use | Compression + visualization | Visualization only |
| Speed | Fast | Slow on large data |
| Axes meaning | Interpretable directions | No meaningful axes — only relative position |

### 3. What Anomaly Detection Is
Anomaly detection finds data points that differ significantly from the norm — fraud, defects, system
failures, or errors. It is often unsupervised because anomalies are rare and rarely pre-labeled.

### 4. Isolation Forest
Isolation Forest randomly partitions the data and measures how few splits it takes to isolate each point —
points isolated quickly are flagged as anomalies.
```python
from sklearn.ensemble import IsolationForest
iso = IsolationForest(contamination=0.05, random_state=42)
preds = iso.fit_predict(X_scaled)
# -1 = anomaly, 1 = normal
```

## Hands-On Lab — Visualization & Anomaly Detection

Using the same prepared **heart-failure dataset** (`heart.csv`, all 11 input features, one-hot encoded and
scaled, as in Day 3):

1. **Recreate Day 1-2's clusters**: run K-Means with `k` chosen by silhouette score, to have cluster labels
   for coloring today's plots.
2. **Apply t-SNE** to reduce the dataset to 2D and plot it, colored by K-Means cluster.
3. **Compare** the t-SNE plot side by side with a fresh PCA plot on the same data, and note what each
   reveals.
4. **Run Isolation Forest** and report how many points were flagged as anomalies.
5. **Inspect two flagged points**, compare them against the dataset's median values, and hypothesize why
   they were flagged — documented in Markdown.

## Lab Results

On this run: K-Means (silhouette-selected) again settled on **k=2**, matching Day 2's clustering result, so
today's plots are colored by the same two-group split. Side by side, the **PCA plot spreads points across a
wide, continuous smear** — consistent with Day 3's finding that 2 components only captured about a third of
the total variance — while the **t-SNE plot pulls same-cluster points into visibly tighter local groups**,
since it optimizes for neighborhood closeness rather than overall spread.

**Isolation Forest flagged 46 patients as anomalies out of 918** (contamination = 0.05, as configured).
Inspecting two of the flagged patients showed neither had any single wildly extreme value; instead, each
combined several unusual readings at once relative to the dataset medians — exactly the kind of pattern
Isolation Forest is designed to catch by isolating points through a *combination* of features rather than
any one outlying column.

## Tools Used

Scikit-learn (`TSNE`, `IsolationForest`, `KMeans`, `PCA`, `StandardScaler`) • Pandas • NumPy • Matplotlib •
Jupyter Notebook

## Best Practices & Reproducibility

- Always scale features before t-SNE or Isolation Forest, exactly as with clustering and PCA.
- Fix `random_state` for t-SNE — its layout is stochastic, and a fixed seed is the only way to get a
  reproducible plot to share with a mentor.
- Never read distances *between* t-SNE clusters as meaningful — only relative neighborhoods matter.
- Choose `contamination` deliberately, based on domain knowledge of how common anomalies actually are, not
  as an arbitrary default.
- Inspect flagged anomalies individually before acting on them — a flagged point is unusual, not
  automatically wrong.

## Where This Leads Next

Day 4 completes the unsupervised-learning toolkit for Week 5: clustering (Days 1-2), dimensionality
reduction for modeling (Day 3), and visualization plus anomaly detection (Day 4). **Day 5** closes Phase 2
and opens Phase 3 — selecting a capstone project type and completing Sprint 1 planning with mentor sign-off.
