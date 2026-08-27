# Week 6 — Deep Learning & Cardiac Patient Monitoring System

**BinX Tech · AI & Machine Learning Internship Program**
**Phase 3 — Sprint 1**
**Project: Cardiac Patient Monitoring System**
**Week 6 of 10 · 5 Training Days · 8 Hours per Day**

---

## Overview

This week marks the start of Sprint 1 execution for the capstone project selected in Week 5. With the classical machine learning phase (Weeks 3–4) already complete, Week 6 moves into deep learning with neural networks — starting from core concepts and progressing to a fully tuned model built with TensorFlow/Keras, benchmarked against a classical Logistic Regression baseline.

The week consists of five sequential days, each building on the previous one:

| Day | Title | Core Content |
|---|---|---|
| Day 1 | Sprint Planning & Neural Network Architecture | Confirming the sprint goal, recording the baseline, the neuron concept |
| Day 2 | Activations, Forward Propagation & Loss | ReLU, Sigmoid, Softmax, Forward Propagation, Loss Functions |
| Day 3 | Backpropagation, Gradient Descent & Optimizers | Backpropagation, Gradient Descent, Learning Rate, Optimizers |
| Day 4 | Building the First Neural Network with Keras | Sequential model, Dropout, Batch Normalization, baseline comparison |
| Day 5 | Tuning, Evaluation & Sprint Review | Hyperparameter Tuning, Callbacks, Sprint Review, Retrospective |

---

## Project Structure

```
Week 6/
├── Day 1/
│   ├── Day1.ipynb
│   ├── heart_cleaned.csv
│   ├── model_comparison.csv
│   └── README.md
├── Day 2/
│   ├── Day2.ipynb
│   ├── heart_cleaned.csv
│   ├── model_comparison.csv
│   └── README.md
├── Day 3/
│   ├── Day3.ipynb
│   ├── heart_cleaned.csv
│   ├── model_comparison.csv
│   └── README.md
├── Day 4/
│   ├── Day4.ipynb
│   ├── heart_disease_health_indicators_BRFSS2015.csv.zip
│   └── README.md
└── Day 5/
    ├── Day5.ipynb
    ├── best_cardiac_model.keras
    ├── heart_disease_health_indicators_BRFSS2015.csv.zip
    └── README.md
```

---

## Daily Breakdown

### Day 1 — Sprint Planning & Neural Network Architecture

This day formally opens Sprint 1. Since the project's classical machine learning phase was already completed, the task is not to train a new baseline from scratch, but to confirm and formally document the existing baseline as the benchmark the neural network must later beat.

**Key concepts:**
- The neuron as a weighted sum plus a bias, passed through an activation function.
- Layer structure: input layer, hidden layers, output layer.
- Weights and biases as the parameters learned during training.

**Confirmed baseline results (Logistic Regression):**

| Metric | Value |
|---|---|
| Accuracy | 87.50% |
| F1 Score | 88.78% |
| ROC-AUC | 93.90% |

**Best classical model (Random Forest) as a stretch benchmark:**

| Metric | Value |
|---|---|
| Accuracy | 89.13% |
| F1 Score | 90.20% |
| ROC-AUC | 92.91% |

---

### Day 2 — Activations, Forward Propagation & Loss

This day answers two fundamental questions: what allows a network to learn complex, non-linear patterns, and how does it measure how wrong a prediction is?

**Activation functions covered:**

| Function | Output Range | Used For |
|---|---|---|
| ReLU | 0 to +∞ | Hidden layers (default choice) |
| Sigmoid | 0 to 1 | Output layer for binary classification |
| Softmax | 0 to 1 (sums to 1) | Output layer for multi-class classification |
| Tanh | -1 to +1 | Hidden layers when zero-centered output helps |

**Decision made for the Cardiac Monitoring project:** ReLU activations in the hidden layers, a single Sigmoid output neuron, and Binary Cross-Entropy loss, given the binary classification nature of the task.

---

### Day 3 — Backpropagation, Gradient Descent & Optimizers

This day completes the training loop by explaining how an error signal is turned into actual weight updates.

**Key concepts:**
- The full training loop: forward pass → loss computation → backpropagation → weight update.
- Backpropagation and the chain rule.
- Learning rate and its effect on training speed and stability.
- Optimizers: SGD and Adam.
- The concepts of epochs and batches.

**Learning-rate experiment results:**
- Very small learning rate: loss decreases very slowly.
- Appropriate learning rate: loss decreases quickly and steadily.
- Very large learning rate: training becomes unstable and overshoots.

---

### Day 4 — Building the First Neural Network with Keras

This day puts everything covered so far into practice by building the project's first real neural network on the CDC BRFSS 2015 dataset (253,680 rows, 22 columns).

**Regularized model architecture:**

```
Input(shape=(21,))
Dense(64, activation="relu")
BatchNormalization()
Dropout(0.3)
Dense(32, activation="relu")
BatchNormalization()
Dropout(0.2)
Dense(1, activation="sigmoid")
```

- **Optimizer:** Adam (default learning rate ≈ 0.001)
- **Loss:** Binary cross-entropy
- **Class imbalance handling:** `class_weight` (dataset is ~90.6% / 9.4% imbalanced)
- **Split:** 60% train / 20% validation / 20% test (stratified)
- **Scaling:** StandardScaler, fit on training data only

**Comparison results:**

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.7533 | 0.3791 | 0.8470 |
| Keras Neural Network (regularized) | 0.7299 | 0.3660 | 0.8493 |

Because the dataset is heavily imbalanced, accuracy alone is not a reliable metric — F1 Score and ROC-AUC are the primary comparison criteria.

---

### Day 5 — Tuning, Evaluation & Sprint Review

Sprint 1 is closed out on this day through systematic hyperparameter tuning, followed by assembling the evidence needed for a formal Sprint Review.

**Tuning approach (one variable at a time):**

1. Learning rate: sweep [0.01, 0.001, 0.0001]
2. Hidden layer width/depth: sweep [(32, 16), (64, 32), (128, 64)]
3. Dropout rate: sweep [0.2, 0.3, 0.5]
4. Batch size: sweep [128, 256, 512]

At each step, the best value from the previous step is fixed before moving to the next.

**Training callbacks:**
- **EarlyStopping:** halts training once validation loss stops improving, restoring the best weights.
- **ModelCheckpoint:** automatically saves the best model during training (`best_cardiac_model.keras`).

**Sprint Review summary:**
Sprint 1 delivered a confirmed classical baseline (Logistic Regression) and a tuned Keras neural network equipped with EarlyStopping and ModelCheckpoint, both trained and evaluated on the same 253,680-row split. The neural network is competitive with the baseline and ahead on ROC-AUC, and no backlog items were dropped.

**Retrospective:**
- What went well: the one-variable-at-a-time tuning approach made it clear which change in each hyperparameter helped.
- What to improve: experiment tracking was done manually via plain DataFrames, which does not scale past a handful of runs.
- Proposed improvement for the next sprint: automatically log every experiment's configuration and metrics using MLflow or a structured CSV log from the start.

---

## Requirements

- Python 3.10+
- Jupyter Notebook / JupyterLab or Google Colab

**Required packages:**

```
tensorflow>=2.15
scikit-learn
pandas
numpy
matplotlib
```

**Running the notebooks:**

1. Place each notebook (`DayX.ipynb`) in the same folder as its corresponding data files.
2. Open the notebook in Jupyter Notebook, VS Code, or Google Colab.
3. Run all cells top to bottom (Shift + Enter), or simply read the saved outputs without re-running anything.

---

## Tools Used This Week

Pandas • NumPy • Matplotlib • TensorFlow / Keras • Scikit-learn • Jupyter Notebook • Git & GitHub

---

## Best Practices Followed

- Reuse an already-trustworthy baseline rather than retraining it from scratch when one already exists.
- Keep both the simple baseline and the strongest classical model visible as comparison points.
- Use the same cleaned dataset and preprocessing logic across all models for a fair comparison.
- Follow full sprint discipline (goal, backlog, feature branch, pull request) even when a step feels already done.
- Fix random seeds to ensure reproducible results.
- Always pair the output activation with the matching loss function (Sigmoid with Binary Cross-Entropy, Softmax with Categorical Cross-Entropy).
- Tune hyperparameters one variable at a time to clearly interpret the effect of each change.
- Use EarlyStopping and ModelCheckpoint to avoid overfitting and retain the best model version.

---

## Sprint 1 Acceptance Criteria

- [x] Notebooks run end-to-end without errors
- [x] Code committed to a dedicated feature branch
- [x] Results documented in Markdown
- [x] Metrics logged and compared against the baseline

---

## Summary & Next Steps

Week 6 successfully closed out Sprint 1 of Phase 3, moving from the theoretical foundations of neural networks to a fully tuned and evaluated Keras model benchmarked against a classical baseline. Results show competitive performance from the neural network, with a slight edge in ROC-AUC. Sprint 2 is expected to focus on improved experiment tracking, deeper architectures, or additional preprocessing techniques aimed at improving F1 Score specifically.
