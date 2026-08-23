# Week 6 — Day 1: Sprint 1 Planning & Neural Network Architecture

**BinX Tech · AI & Machine Learning Internship Program · Phase 3 — Sprint 1**
**Project: Cardiac Patient Monitoring System**
**Week 6 of 10 · Day 1 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [How This Connects to Week 5, Day 5](#how-this-connects-to-week-5-day-5)
3. [Learning Objectives](#learning-objectives)
4. [Key Topics](#key-topics)
5. [Files in This Folder](#files-in-this-folder)
6. [How to Use the Notebook](#how-to-use-the-notebook)
7. [Lesson Summary](#lesson-summary)
   - [1. Sprint 1 Planning & the Baseline First](#1-sprint-1-planning--the-baseline-first)
   - [2. Why Deep Learning](#2-why-deep-learning)
   - [3. The Neuron](#3-the-neuron)
   - [4. Layers and Architecture](#4-layers-and-architecture)
   - [5. Weights and Biases Are What's Learned](#5-weights-and-biases-are-whats-learned)
8. [Hands-On Lab — Sprint 1 Kickoff & Baseline](#hands-on-lab--sprint-1-kickoff--baseline)
9. [Lab Results](#lab-results)
10. [Tools Used](#tools-used)
11. [Best Practices & Reproducibility](#best-practices--reproducibility)
12. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 1 opens **Phase 3, Sprint 1** for the already-selected capstone project: the **Cardiac Patient Monitoring
System**. Unlike a generic Sprint 1 kickoff that starts from an unexplored dataset, this project's classical
machine learning phase (Weeks 3–4 material) is already complete — five models were already trained and compared
in `02_supervised_learning_and_model_comparison.ipynb`. Day 1's real job is therefore not to train a baseline
from scratch, but to **formally confirm the Sprint 1 goal and record the existing baseline score** that this
week's neural network must beat, before any deep learning work begins.

## How This Connects to Week 5, Day 5

Week 5, Day 5 was **planning only** — the Cardiac Patient Monitoring project was selected, its problem statement
and Definition of Done were written, and a Sprint 1 backlog was drafted, but no model code was written that day.
Week 6, Day 1 is the **first execution day** of that plan:

| | Week 5, Day 5 | Week 6, Day 1 |
|---|---|---|
| What happened | Planning only — no model code | First real execution of the Sprint 1 backlog |
| Deliverable | Project selection, problem statement, Sprint 1 backlog | Confirmed goal + a recorded, trusted baseline score |
| Model status | No model trained yet | Baseline formally confirmed and recorded |

Because this project's baseline-training backlog task was already completed ahead of schedule (during the
project's own classical ML phase), Day 1 reframes that backlog item as **"confirm and formally record the
existing baseline"** rather than duplicating the training work. The Sprint discipline — goal, backlog, feature
branch, pull request — still applies exactly the same way.

## Learning Objectives

By the end of Day 1, you should be able to:

- **Complete** Sprint 1 planning and confirm the baseline for the Cardiac Patient Monitoring project.
- **Explain** a single neuron as a weighted sum, bias, and activation.
- **Describe** the role of input, hidden, and output layers and what "deep" means.

## Key Topics

- Sprint 1 planning: goal, backlog, and baseline first
- Connecting a prior week's planning to this week's execution
- Why deep learning: unstructured, high-dimensional data
- The neuron: weighted sum + bias + activation (the Week 2 dot product)
- Layers: input, hidden, output
- Weights and biases as the learned parameters

## Files in This Folder

| File | Description |
|---|---|
| `Day1.ipynb` | The Day 1 notebook — explanations, a worked neuron example using this project's own features, a cheat sheet, and the Hands-On Lab, fully executed against the real project data and results. |
| `README.md` | This file. |

This notebook reads directly from `heart_cleaned.csv` and `model_comparison.csv`, which the earlier project
notebooks (`01_data_understanding_and_eda.ipynb` and `02_supervised_learning_and_model_comparison.ipynb`)
already produced.

## How to Use the Notebook

1. Place `Day1.ipynb` in the same folder as `heart_cleaned.csv` and `model_comparison.csv` (both files are
   read using plain, same-folder filenames, not a nested `data/` or `outputs/` path).
2. Open it in Jupyter Notebook, VS Code, or Google Colab.
3. Read top-to-bottom — every section opens with a short **Note**, **Goal**, **Tip**, or **Important** box
   explaining *why* a step matters before showing *how* to do it.
4. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can also just read
   the saved outputs without re-running anything.
5. Make sure `heart_cleaned.csv` and `model_comparison.csv` already exist in that same folder — both are
   produced by the project's earlier notebooks (`01_...` and `02_...`), which should be run first if starting
   from scratch.

## Lesson Summary

### 1. Sprint 1 Planning & the Baseline First
Day 1 opens with formal Sprint Planning: confirm the Sprint 1 goal set in Week 5, break it into backlog tasks,
and commit to the goal. Sprint 1's real objective is to establish a **baseline** — a simple model whose score
every later model, including this week's neural network, must beat. For this project, that baseline already
exists: **Logistic Regression**, trained during the classical ML phase.

### 2. Why Deep Learning
Classical ML — already used successfully in this project — works well on structured, tabular data like
`heart.csv`. Deep learning shines where classical methods struggle: images, text, audio, and other
high-dimensional, unstructured data with complex patterns. Notably, this project's own five-model comparison
already showed a tight accuracy band (82%–89%) across very different classical algorithms — a signal that a
neural network may face real competition to improve on that result.

### 3. The Neuron
A single neuron is exactly the Week 2 dot product, a bias, and an activation function:
```python
# One neuron, conceptually, using two of this project's own scaled features:
# z = (Oldpeak * w1) + (MaxHR * w2) + bias      <- the dot product from Week 2
# output = activation(z)
```
The notebook demonstrates this using `Oldpeak` and `MaxHR` — the two numeric features this project's own EDA
found to have the strongest correlation with `HeartDisease` (+0.404 and -0.401, respectively).

### 4. Layers and Architecture
For this specific project, the eventual neural network's shape is already implied by the existing preprocessing:

| Layer | Role | For This Project |
|---|---|---|
| Input layer | Receives the raw features | 15 nodes — the same numeric + one-hot encoded columns the classical models already use |
| Hidden layer(s) | Learn intermediate representations | To be designed on Day 4 |
| Output layer | Produces the final prediction | 1 node, sigmoid activation — binary `HeartDisease` probability |

### 5. Weights and Biases Are What's Learned
The weights and biases are the network's parameters — the numbers adjusted during training. A fresh network
starts with random weights and makes random predictions; training is the process of nudging these weights until
the predictions become accurate. Everything on Days 2–3 is about how that nudging works.

## Hands-On Lab — Sprint 1 Kickoff & Baseline

Using the **Cardiac Patient Monitoring project's own cleaned dataset and results**, the lab runs a real Sprint 1
kickoff:

1. **Confirm** Sprint 1 planning: restate the sprint goal, adapted to reflect that the baseline task is already
   complete.
2. **Load** the project's existing cleaned dataset (`heart_cleaned.csv`, 917 patients).
3. **Confirm** the existing baseline score by reading it directly from `model_comparison.csv`, rather
   than retraining it.
4. **Document** the feature-branch and pull-request workflow used to commit this confirmation.
5. **Record**, in Markdown, the exact baseline score this week's neural network must beat.

## Lab Results

The confirmed Sprint 1 baseline — **Logistic Regression** — scores:

- **Accuracy: 87.50%**
- **F1 Score: 88.78%**
- **ROC-AUC: 93.90%**

The project's strongest classical model, **Random Forest**, is also kept visible as a stretch benchmark:

- **Accuracy: 89.13%**
- **F1 Score: 90.20%**
- **ROC-AUC: 92.91%**

Both numbers are recorded so that Day 4's Keras neural network can be judged fairly against the standard Week 3
baseline choice (Logistic Regression) and against the best classical result achieved so far (Random Forest).

## Tools Used

Pandas • NumPy • Jupyter/Colab • Git & GitHub

## Best Practices & Reproducibility

- Reuse an already-trustworthy baseline rather than retraining it from scratch — confirming and recording it is
  enough.
- Keep both the simple baseline (Logistic Regression) and the strongest classical model (Random Forest) visible,
  since a neural network should ideally be compared against both.
- Reuse the exact same cleaned dataset and preprocessing logic (`src/preprocessing.py`) for the neural network,
  so any difference in score reflects the model, not the data.
- Still follow full Sprint discipline (goal, backlog, feature branch, pull request) even when a step feels
  "already done" — documentation is part of the deliverable.
- Keep the neuron mental model (dot product + bias + activation) in mind going into Days 2–3 — it prevents deep
  learning from feeling like new math.

## Where This Leads Next

With the Sprint 1 goal confirmed and both a baseline (87.50% accuracy / 0.9390 ROC-AUC) and a stretch benchmark
(89.13% accuracy) locked in, **Day 2** moves from the single neuron to **activation functions, forward
propagation, and loss functions** — the mechanics a full network uses to turn the project's 15 input features
into a `HeartDisease` prediction, one step closer to a model that must actually beat today's recorded scores.


