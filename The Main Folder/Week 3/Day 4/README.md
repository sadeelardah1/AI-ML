# Week 3 — Day 4: Trees, Forests, SVMs & k-NN

**BinX Tech · AI & Machine Learning Internship Program · Phase 2: Classical Machine Learning**
**Week 3 of 10 · Day 4 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day4.ipynb`](#how-to-use-day4ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. Decision Trees](#1-decision-trees)
   - [2. Random Forests](#2-random-forests)
   - [3. Support Vector Machines (SVM)](#3-support-vector-machines-svm)
   - [4. k-Nearest Neighbors (k-NN)](#4-k-nearest-neighbors-k-nn)
   - [5. Comparing Models Fairly](#5-comparing-models-fairly)
7. [Hands-On Lab — Model Comparison](#hands-on-lab--model-comparison)
8. [Tools Used](#tools-used)
9. [Best Practices & Reproducibility](#best-practices--reproducibility)
10. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 4 of Week 3 introduces four new classifiers — **decision trees**, **random forests**, **SVMs**,
and **k-NN** — and, more importantly, teaches the professional habit of comparing multiple models
**fairly** on the same train/test split and the same metric, instead of guessing which algorithm
"should" work best.

## Learning Objectives

By the end of Day 4, you should be able to:

- **Train and interpret** decision trees and random forests, including feature importances.
- **Train** SVM and k-NN classifiers and explain how each makes decisions.
- **Compare** multiple classifiers fairly on the same train/test split and metric.

## Key Topics

- Decision trees: rule-based, interpretable, prone to overfitting
- Random forests: ensembles and feature importances
- Support Vector Machines and the margin
- k-Nearest Neighbors
- Comparing models fairly (no free lunch)

## Files in This Folder

| File | Description |
|---|---|
| `Day4.ipynb` | The full, detailed Day 4 lesson notebook — explanations, worked examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs (an F1-score comparison chart and a feature-importance chart). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |
| `train_and_test2.csv` | The same Titanic-style dataset from Day 3, reused here so all four models are compared on identical data. |

## How to Use `Day4.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note** or **Goal** box explaining
   *why* a concept matters before showing *how* to implement it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** (Section 7) is a fast reference you can come back to for the rest of the program.
6. Make sure `train_and_test2.csv` is in the same folder as the notebook before running the
   Hands-On Lab cells.

## Lesson Summary

### 1. Decision Trees
Predicts by asking a sequence of yes/no questions about the features, splitting the data at each
step until it reaches a decision. Interpretable, but a deep tree can overfit — memorizing the
training data instead of generalizing:
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)
```

### 2. Random Forests
An ensemble that trains many decision trees on random subsets of the data and features, then
averages their votes. This "wisdom of the crowd" fixes the single tree's overfitting problem and
is often a strong default choice:
```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print(model.feature_importances_)   # which features mattered most
```

### 3. Support Vector Machines (SVM)
Finds the decision boundary that maximizes the margin — the widest possible gap between the two
classes. The "kernel trick" lets it draw curved boundaries too. Powerful on small-to-medium,
high-dimensional data; scales poorly to very large datasets:
```python
from sklearn.svm import SVC
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)
```

### 4. k-Nearest Neighbors (k-NN)
The simplest classifier: to predict a new point's class, it looks at the `k` closest training
points and takes a majority vote. It learns nothing during training — all the work happens at
prediction time:
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
```

### 5. Comparing Models Fairly

| Model | Strength | Watch Out For |
|---|---|---|
| Decision Tree | Interpretable rules | Overfitting if too deep |
| Random Forest | Strong, reliable default | Less interpretable than one tree |
| SVM | Powerful in high dimensions | Slow on large datasets |
| k-NN | Simple, intuitive | Slow and weaker as data grows |

No single algorithm is best for every problem (the "no free lunch" principle). The professional
approach is to train several models on the same train/test split and compare them fairly on the
same metric.

## Hands-On Lab — Model Comparison

Reusing the **Titanic passenger dataset** (`train_and_test2.csv`) and the same train/test split
from Day 3, the lab runs a complete model-comparison workflow:

1. **Train** a decision tree, random forest, SVM, and k-NN on the same train/test split.
2. **Evaluate** all four with the same metrics (accuracy, precision, recall, F1) and assemble the
   results into one comparison table and chart.
3. **Report** the random forest's top feature importances and interpret them.
4. **Identify** the best-performing model for this dataset and explain, in Markdown, why it
   likely won.

## Tools Used

Scikit-learn (tree, ensemble, svm, neighbors) • Pandas • Matplotlib • Jupyter Notebook

## Best Practices & Reproducibility

- Always compare models on the exact same train/test split and `random_state`.
- Use the same metric for every model in a comparison — mixing metrics makes comparisons meaningless.
- Limit decision tree depth (`max_depth`) to reduce overfitting risk.
- Remember that SVM and k-NN are distance-based — feature scaling (Day 5) often improves them
  significantly.
- Never assume one algorithm is "best" without testing it on this specific dataset.

## Where This Leads Next

Day 4's model-comparison habit — training multiple candidates fairly and picking a winner by
evidence, not assumption — is exactly the mindset **Day 5**'s end-to-end mini-project formalizes
into a complete pipeline: EDA → preprocessing (with proper feature scaling) → modeling →
evaluation against a baseline.

