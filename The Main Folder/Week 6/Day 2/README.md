# Day 2 — Activations, Forward Propagation & Loss

**BinX Tech · AI & Machine Learning Internship Program**
**Project: Cardiac Patient Monitoring System**
**Day 2 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day2.ipynb`](#how-to-use-day2ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. Why Activation Functions Matter](#1-why-activation-functions-matter)
   - [2. Common Activation Functions](#2-common-activation-functions)
   - [3. Forward Propagation](#3-forward-propagation)
   - [4. The Loss Function](#4-the-loss-function)
7. [Hands-On Lab — Activations & the Forward Pass](#hands-on-lab--activations--the-forward-pass)
8. [Tools Used](#tools-used)
9. [Best Practices & Reproducibility](#best-practices--reproducibility)
10. [Daily Stand-up](#daily-stand-up)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 2 answers two questions that follow immediately from Day 1's building blocks (neurons,
weights, layers): what makes a stack of layers able to learn curved, complex patterns instead of
just a straight line, and how does the network know when a prediction is wrong? The answers —
**activation functions** and **loss functions** — are the two ingredients that turn a pile of
weighted sums into a trainable neural network.

## Learning Objectives

By the end of Day 2, you should be able to:

- **Explain** why non-linear activations are essential.
- **Choose** the correct activation for hidden and output layers.
- **Describe** forward propagation and select the right loss function for a task.

## Key Topics

- Why activations introduce non-linearity
- Common activations: ReLU, sigmoid, softmax, tanh
- Choosing activations by layer and task
- Forward propagation: computing a prediction
- Loss functions: MSE, binary/categorical cross-entropy

## Files in This Folder

| File | Description |
|---|---|
| `Day2.ipynb` | The full, detailed Day 2 lesson notebook — explanations, worked examples, and the Hands-On Lab, fully executed with outputs (activation function plots and a hand-computed forward pass with loss). |
| `README.md` | This file — a plain-language overview of the whole lesson, for anyone browsing the repository. |

## How to Use `Day2.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section opens with a short **Note**, **Important**, or **Tip** box
   explaining *why* a concept matters before showing *how* to implement it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. This notebook needs no external dataset — every example uses small, self-contained NumPy arrays.

## Lesson Summary

### 1. Why Activation Functions Matter
Without an activation function, a neural network — no matter how many layers — collapses into a
single linear model, because a stack of linear operations is still linear. The activation function
introduces non-linearity, which is what lets the network learn complex, curved patterns.

### 2. Common Activation Functions

| Function | Output Range | Use It For |
|---|---|---|
| ReLU | 0 to +∞ (negatives → 0) | Hidden layers — the default choice, fast and effective |
| Sigmoid | 0 to 1 | Output layer for binary classification (a probability) |
| Softmax | 0 to 1, sums to 1 | Output layer for multi-class classification (class probabilities) |
| Tanh | -1 to +1 | Hidden layers when zero-centered output helps |

The practical rule: use ReLU in hidden layers by default, and choose the output activation by the
task — sigmoid for binary, softmax typically for multi-class, and none (linear) for regression. For binary classification, a single sigmoid output is the standard and simpler choice; a two-output softmax formulation is possible but usually unnecessary.

### 3. Forward Propagation
Forward propagation is the network making a prediction: data enters the input layer and flows
forward, each layer computing its weighted sums and activations, until the output layer produces a
prediction.
```python
# layer1 = ReLU(dot(X, W1) + b1)
# layer2 = ReLU(dot(layer1, W2) + b2)
# output = sigmoid(dot(layer2, W3) + b3)
```

### 4. The Loss Function
After a forward pass, the loss function measures how wrong the prediction was by comparing it to
the true label. The right loss depends on the task:

| Task | Loss Function |
|---|---|
| Regression | Mean Squared Error (MSE) |
| Binary classification | Binary cross-entropy |
| Multi-class classification | Categorical cross-entropy |

The output activation and loss function are always chosen as a matched pair: sigmoid + binary
cross-entropy, softmax + categorical cross-entropy, linear + MSE.

## Hands-On Lab — Activations & the Forward Pass

1. **Plot** ReLU, sigmoid, and tanh over a range of inputs to see how each transforms values.
2. **Decide** the correct output activation and loss function for the **Cardiac Patient Monitoring System**
   (`HeartDisease` binary classification) and justify both.
3. **Compute**, by hand in NumPy, one full forward pass for a tiny 2-layer network on a sample
   input, then pair the resulting prediction with a binary cross-entropy loss against a true label.
4. **Document** the choices and the forward-pass result in Markdown.

## Tools Used

NumPy • Matplotlib • Jupyter Notebook

## Best Practices & Reproducibility

- Always use a non-linear activation between layers — never stack purely linear layers.
- Default to ReLU for hidden layers unless there is a specific reason to choose otherwise.
- Always pair the output activation with the matching loss function.
- Fix `np.random.seed(...)` when initializing weights for a worked example, so results are
  reproducible.
- Clip predictions away from exactly 0 or 1 before computing cross-entropy loss, to avoid
  `log(0)` errors.
- When implementing softmax manually, subtract the maximum value before exponentiation and normalize along the last axis for numerical stability and batch-safe behavior.


## Daily Stand-up

- **Completed:** Plotted ReLU, sigmoid, and tanh; selected ReLU + sigmoid + binary cross-entropy for the Cardiac Patient Monitoring System; completed and documented a manual 2-layer forward pass and BCE loss.
- **Next:** Day 3 — backpropagation, gradient descent, learning-rate experiments, and preparation for the mentor code/notebook review.
- **Blockers:** None identified in today's notebook; continue watching tensor shapes and numerical stability when moving from NumPy examples to framework code.

## Where This Leads Next

Day 2 established how a network computes a prediction (forward propagation) and measures how wrong
it was (the loss). **Day 3** covers backpropagation — how that loss flows backward through the
network to actually update the weights, which is what training a neural network means.


