# Week 5 — Unsupervised Learning & Capstone Kickoff

**BinX Tech · AI & Machine Learning Internship Program**
**Phase 2 → Phase 3 Transition · Week 5 of 10 · 5 Days · 40 Hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Week Structure](#week-structure)
3. [Day 1 — Unsupervised Learning & K-Means](#day-1--unsupervised-learning--k-means)
4. [Day 2 — DBSCAN & Hierarchical Clustering](#day-2--dbscan--hierarchical-clustering)
5. [Day 3 — Dimensionality Reduction with PCA](#day-3--dimensionality-reduction-with-pca)
6. [Day 4 — t-SNE & Anomaly Detection](#day-4--t-sne--anomaly-detection)
7. [Day 5 — Capstone Project: Cardiac Patient Monitoring System](#day-5--capstone-project-cardiac-patient-monitoring-system)
8. [Datasets Used](#datasets-used)
9. [Tools Used Across the Week](#tools-used-across-the-week)
10. [How to Run Everything](#how-to-run-everything)
11. [Key Takeaways](#key-takeaways)

---

## Overview

Week 5 is the bridge between Phase 2 (supervised learning, Weeks 3–4) and Phase 3 (capstone project work).
The first four days build a complete unsupervised-learning toolkit — clustering, dimensionality reduction,
visualization, and anomaly detection — all applied to real datasets with no target labels used during
training. Day 5 closes the phase by applying everything learned across Weeks 1–4 to a full, end-to-end
supervised project: a heart-disease prediction system built and evaluated the way a real project would be.

## Week Structure

| Day | Topic | Dataset |
|---|---|---|
| Day 1 | Unsupervised learning & K-Means clustering | `customer_segmentation.csv` |
| Day 2 | DBSCAN & hierarchical clustering | `heart.csv` |
| Day 3 | Dimensionality reduction with PCA | `heart.csv` |
| Day 4 | t-SNE visualization & anomaly detection (Isolation Forest) | `heart.csv` |
| Day 5 | Capstone project — full supervised ML pipeline | `heart.csv` |

---

## Day 1 — Unsupervised Learning & K-Means

**Files:** `Day1.ipynb`, `README.md`, `customer_segmentation.csv`

Introduces unsupervised learning as a contrast to the supervised work of Weeks 3–4: instead of predicting a
known target, the goal is to discover hidden structure in unlabeled data.

**Covered:**
- Supervised vs. unsupervised learning
- What clustering does
- K-Means step by step (centroid placement, assignment, update, repeat)
- Choosing `k` with the elbow method (inertia vs. `k`)
- Choosing `k` with the silhouette score (-1 to +1, higher is better-defined clusters)

**Hands-on lab:** scales the customer segmentation dataset, sweeps `k` from 1–10 to find the elbow, confirms
the choice with silhouette scores, fits the final K-Means model, visualizes the clusters in 2D, and
interprets what each cluster represents in plain language.

**Key rule:** always scale features before clustering, since K-Means is entirely distance-based.

---

## Day 2 — DBSCAN & Hierarchical Clustering

**Files:** `Day2.ipynb`, `README.md`, `heart.csv`

Introduces two alternatives to K-Means that need no `k` chosen in advance and handle irregular shapes and
noise more gracefully.

**Covered:**
- Why K-Means isn't always enough (needs `k` in advance, assumes round clusters, forces every point into a
  group)
- **DBSCAN**: groups points by density, discovers the number of clusters automatically, and explicitly
  flags outliers as noise (label `-1`). Controlled by `eps` (neighbor distance) and `min_samples`.
- **Hierarchical clustering**: builds a full tree of nested clusters (a dendrogram) by repeatedly merging
  the two closest points/clusters; can be cut at any height to get any number of clusters.
- A comparison table of when to use K-Means vs. DBSCAN vs. hierarchical clustering, and common mistakes to
  avoid with each.

**Hands-on lab:** runs all three methods on the heart-failure dataset's five numeric clinical features
(`Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`), after imputing zero-as-missing values in
`RestingBP` and `Cholesterol`. Result: K-Means (k=2) had the best silhouette score and is the recommended
baseline; DBSCAN flagged 201 points as noise, showing the data does not form perfectly dense, well-separated
groups; hierarchical clustering's five-cluster cut is more useful for exploration than for a clean
segmentation.

---

## Day 3 — Dimensionality Reduction with PCA

**Files:** `Day3.ipynb`, `README.md`, `heart.csv`

Moves from clustering to compressing many features into a handful of new axes that capture as much of the
data's variance as possible.

**Covered:**
- The curse of dimensionality (sparsity, meaningless distances, easier overfitting, no visualization beyond
  3D)
- What PCA does: principal components ranked by variance captured
- Explained variance ratio and the common rule of retaining ~95% of variance
- When PCA helps (speed, reduced overfitting, visualization) and its cost (loss of direct feature
  interpretability)

**Hands-on lab:** one-hot encodes all 11 clinical features into 15 numeric columns, scales them, and fits
PCA. Reaching the 95% variance threshold required 13 of the 15 components — almost no compression, because
one-hot categorical flags are largely independent of one another. Reducing to 2 components for plotting
retained only about 33% of the variance, and the resulting scatter plot (colored by `HeartDisease`) showed
heavy overlap between the two diagnosis groups — confirming 2D PCA here is a visualization tool, not a
finished feature set for classification.

---

## Day 4 — t-SNE & Anomaly Detection

**Files:** `Day4.ipynb`, `README.md`, `heart.csv`

Closes the visualization and detection half of the week's unsupervised toolkit.

**Covered:**
- **t-SNE**: a visualization-only reduction technique that preserves local neighborhoods rather than global
  variance, often revealing cluster shapes PCA cannot
- PCA vs. t-SNE comparison (global structure vs. local neighborhoods, speed, axis interpretability)
- What anomaly detection is and why it is usually unsupervised
- **Isolation Forest**: isolates points via random partitioning; points isolated in fewer splits are flagged
  as anomalies, controlled by the `contamination` parameter

**Hands-on lab:** recreates Day 1–2's K-Means clusters (k=2) for coloring, applies t-SNE to the same
one-hot-encoded, scaled features, and compares it side by side with a fresh PCA plot. The t-SNE plot pulled
same-cluster points into visibly tighter local groups, while PCA showed a wide, continuous smear. Isolation
Forest (contamination = 0.05) flagged 46 of 918 patients as anomalies; inspection showed flagged patients had
no single extreme value but combined several unusual readings at once relative to the dataset's medians.

---

## Day 5 — Capstone Project: Cardiac Patient Monitoring System

**Folder:** `cardiac-patient-monitoring/` (see its own `README.md` for full detail)

A complete, end-to-end **supervised** machine learning project — the phase's capstone — predicting whether a
patient has heart disease from 11 clinical measurements. Deliberately scoped to classical Scikit-learn
models only (no clustering, no deep learning), as a full application of Weeks 1–4.

**Workflow:**
1. Understand the data (918 patients, 11 features, 1 target).
2. Clean it — remove one invalid `RestingBP = 0` row, treat `Cholesterol = 0` (~19% of rows) as missing for
   median imputation, check for duplicates.
3. Explore it — target is balanced (44.71% / 55.29%); chest pain type, exercise-induced angina, and ST
   slope are strong predictors; `Oldpeak` and `MaxHR` are the strongest numeric correlates.
4. Split 80/20, stratified, `random_state=42`.
5. Build a single leakage-free Scikit-learn pipeline (median imputation, one-hot encoding, standard
   scaling).
6. Train five models: Logistic Regression, Decision Tree, Random Forest, SVM (RBF), KNN (k=15).
7. Compare fairly on accuracy, precision, recall, F1, and ROC-AUC on the held-out test set.
8. Read the confusion matrix to understand the trade-off between false positives and false negatives.
9. Confirm rankings with 5-fold cross-validation.
10. Compare ROC-AUC across all five models.
11. Test an engineered feature (age bands) — result was mixed, so it was rejected in favor of raw `Age`.
12. Select the final model.

**Result:** Random Forest was selected as the final model — 89.13% accuracy, 90.20% precision, 90.20%
recall, 90.20% F1, 92.91% ROC-AUC on unseen data. SVM (RBF) is documented as the strongest alternative,
with the best ROC-AUC (94.33%) and the fewest missed disease cases, useful if minimizing false negatives is
the priority. The final pipeline is saved to `models/random_forest_pipeline.joblib` for reuse without
retraining.

> This project is educational only. It is not a medical tool and must never be used for real diagnosis,
> treatment decisions, or emergency care.

---

## Datasets Used

| Dataset | Used In | Description |
|---|---|---|
| `customer_segmentation.csv` | Day 1 | Customer records: `Age`, `Annual_Income_k`, `Spending_Score` — no target column |
| `heart.csv` | Days 2, 3, 4, 5 | Heart Failure Prediction dataset: 918 patients, 11 clinical features, `HeartDisease` target (0/1) |

## Tools Used Across the Week

Scikit-learn (`KMeans`, `DBSCAN`, hierarchical clustering via SciPy, `PCA`, `TSNE`, `IsolationForest`,
`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`, `KNeighborsClassifier`,
`StandardScaler`, `Pipeline`) • SciPy • Pandas • NumPy • Matplotlib • Jupyter Notebook

## How to Run Everything

1. Open each day's folder and launch its notebook in Jupyter Notebook, VS Code, or Google Colab.
2. Keep each notebook's CSV file in the same folder as that notebook before running any lab cells.
3. Run cells top to bottom with `Shift + Enter`; all notebooks are already fully executed, so outputs can
   also be read without re-running anything.
4. For Day 5, follow the separate setup steps in `cardiac-patient-monitoring/README.md` (virtual
   environment, `requirements.txt`, and running the two notebooks in order).

## Key Takeaways

- Unsupervised learning finds structure in data with no labels — clustering groups similar points,
  dimensionality reduction compresses features, anomaly detection flags what does not fit.
- No single clustering method wins on every dataset; the right choice depends on the data's shape, checked
  by comparing methods side by side.
- PCA compresses continuous, correlated measurements far better than one-hot-encoded categorical flags.
- t-SNE is for visualization only — never read distances between t-SNE clusters as meaningful.
- A trustworthy supervised project compares multiple models fairly, checks results with cross-validation
  and confusion matrices rather than a single metric, and picks a final model based on evidence, not
  assumption.
