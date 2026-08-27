# Week 6 — Day 5: Tuning, Evaluation & Sprint Review

Cardiac Patient Monitoring System — Closing Out Sprint 1

## Overview

Day 4 trained a plain Keras network and a regularized one (BatchNorm + Dropout) on the 253,680-row BRFSS dataset, landing close to the Logistic Regression baseline on F1 but ahead on ROC-AUC. Day 5 closes Sprint 1: tune the network systematically, train it efficiently with callbacks, then assemble the evidence for a proper Sprint Review.

## Contents

1. Setup — Importing TensorFlow, Pandas & Scikit-learn
2. Tuning a Neural Network
3. Callbacks: EarlyStopping and Checkpoints
4. Preparing for Sprint Review
5. Sprint Review & Retrospective
6. Common Mistakes to Avoid
7. Hands-On Lab — Sprint 1 Close-Out
8. Best Practices & Reproducibility
9. Summary — What I Learned Today
10. Daily Stand-up

## What Changed From Day 4

- The absolute file path used to load the dataset was replaced with a relative path (`heart_disease_health_indicators_BRFSS2015.csv.zip`), so the notebook now runs on any machine or in Google Colab as long as the CSV sits next to the notebook.
- The first `Dense` layer no longer takes `input_shape`. The model now starts with an explicit `Input(shape=(n_features,))` layer, which removes the Keras `UserWarning` about passing `input_shape`/`input_dim` to a layer.

## Tuning Approach

Tuning is done one variable at a time, in priority order, reusing the best value from each step in the next:

1. **Step 1a — Learning rate**: sweep `[0.01, 0.001, 0.0001]`, everything else at its default.
2. **Step 1b — Width/depth**: sweep hidden layer sizes `[(32, 16), (64, 32), (128, 64)]`, with the learning rate fixed at the best value from Step 1a.
3. **Step 1c — Dropout rate**: sweep `[0.2, 0.3, 0.5]`, with learning rate and width/depth fixed at their best values so far.
4. **Step 1d — Batch size**: sweep `[128, 256, 512]`, with learning rate, width/depth, and dropout rate fixed at their best values so far.

A shared `run_sweep` helper trains one model per candidate value and records the best validation loss and the number of epochs run, so every sweep follows the same one-variable-at-a-time discipline used in Week 4.

The final model is then trained with the full tuned configuration (best learning rate, hidden units, dropout rate, and batch size) using `EarlyStopping` and `ModelCheckpoint`.

## Callbacks

- **EarlyStopping** (`monitor="val_loss"`, `patience=5`, `restore_best_weights=True`) halts training once the validation loss stops improving and restores the weights from the best epoch, not the last one.
- **ModelCheckpoint** (`monitor="val_loss"`, `save_best_only=True`) saves the best model to disk during training as `best_cardiac_model.keras`.

## Loss Curve

The training/validation loss plot marks two distinct points, computed from `val_loss`:

- **Best epoch** — `np.argmin(val_loss)`, the epoch with the lowest validation loss (what `restore_best_weights` keeps).
- **Stopping epoch** — the last epoch actually run before `EarlyStopping` halted training.

These are shown as two separate vertical lines since they are not always the same epoch.

## Model Comparison

The tuned neural network is compared explicitly against a Logistic Regression baseline, both re-fit and evaluated on the same train/val/test split within the same run, using Accuracy, F1 Score, and ROC-AUC. The comparison is reported as-is even where the neural network's edge over the baseline is small.

## Sprint Review

Sprint 1 delivered a confirmed classical baseline (Logistic Regression) and a tuned Keras neural network with EarlyStopping and ModelCheckpoint, both trained and evaluated on the same 253,680-row BRFSS split. The neural network is competitive with the baseline and ahead on ROC-AUC. No backlog items were dropped from Sprint 1.

## Retrospective

- **What went well**: the one-variable-at-a-time tuning approach made it clear which change in each hyperparameter helped, without confusing multiple simultaneous changes.
- **What to improve**: hyperparameter results were tracked manually in plain DataFrames; this does not scale past a handful of experiments.
- **One concrete change for Sprint 2**: log every experiment's configuration and metrics to MLflow (or a simple CSV log) from the very first run, instead of retrofitting tracking after the fact.

## Acceptance Criteria Check

- Notebook runs without errors: yes
- Code committed to the correct feature branch: yes
- Results documented in Markdown: yes
- Metrics logged and compared to baseline: yes
- Pull request approved by mentor: pending, required before merge

## Requirements

- `heart_disease_health_indicators_BRFSS2015.csv.zip` placed in the same directory as the notebook (or uploaded to the same working directory in Colab).
- Python packages: `numpy`, `pandas`, `matplotlib`, `tensorflow`, `scikit-learn`.

