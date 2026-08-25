# Week 6 — Day 3: Backpropagation, Gradient Descent & Optimizers

**BinX Tech · AI & Machine Learning Internship Program · Phase 3 — Sprint 1**  
**Project: Cardiac Patient Monitoring System**  
**Day 3 of 5 · 8 hours**

---

## Overview

Day 3 explains the part of neural-network training that turns an error signal into parameter updates. Day 2 established the forward pass and binary cross-entropy loss; Day 3 completes the training loop by covering backpropagation, the chain rule, gradient descent, learning rates, optimizers, epochs, and batches.

The hands-on lab uses a small self-contained NumPy network to make the mechanics visible before Day 4 moves to the project network in TensorFlow/Keras. The synthetic lab data are not patient records and are not used as the project model.

## Connection to the Cardiac Patient Monitoring System

The Phase 3 project is a binary-classification system. The architecture choices established on Day 2 remain unchanged:

- Hidden layers: ReLU
- Output layer: one sigmoid neuron
- Loss: binary cross-entropy

Day 3 explains how gradients for those layers are computed and how an optimizer uses them. Day 4 will apply the same principles when training the project neural network in Keras.

## Learning Objectives

By the end of Day 3, the notebook demonstrates the ability to:

- Describe the four-step neural-network training loop.
- Explain gradient descent and the role of the learning rate.
- Explain backpropagation conceptually and why the chain rule is required.
- Distinguish backpropagation from the optimizer.
- Explain optimizers, epochs, and batches.
- Compare learning-rate behavior using reproducible loss curves.
- Prepare the current project notebook for the mid-sprint mentor review workflow.

## Key Topics

- Training loop: forward pass → loss → backpropagation → update
- Gradient descent and parameter updates
- Learning-rate behavior
- Backpropagation and the chain rule
- SGD and Adam
- Epochs and batches
- Learning-rate experiments and loss-curve interpretation
- Feature branch and pull-request workflow for mentor review

## Files

| File | Description |
|---|---|
| `Day3.ipynb` | Fully executed Day 3 notebook with explanations, worked examples, the learning-rate lab, daily stand-up, and mentor-review checklist. |
| `README.md` | Project-oriented summary of the Day 3 work and its connection to Sprint 1. |

## Hands-On Lab

The lab follows four stages:

1. Describe the full training loop in Markdown.
2. Train the same tiny NumPy network at three learning rates and compare the loss curves.
3. Explain backpropagation and the chain rule in plain language.
4. Prepare the notebook for the mid-sprint GitHub pull-request and mentor-review workflow.

The learning-rate experiment deliberately keeps the data, initialization, architecture, and number of epochs fixed. Only the learning rate changes, making the comparison interpretable.

## Lab Results

The three learning-rate runs demonstrate three distinct behaviors:

- A very small learning rate reduces the loss only slightly.
- An appropriate rate for the toy full-batch network reduces the loss rapidly and consistently.
- A very large rate overshoots and becomes unstable.

The exact numeric rates in the toy NumPy experiment are not intended as Keras/Adam defaults. Day 4 should start Adam from a conventional value such as `0.001` and tune only when the training curves justify it.

## Accessibility and Visual Design

The notebook uses a Day 3-specific violet and burnt-orange palette rather than the Day 2 palette. The design was built with accessibility in mind:

- Text labels carry meaning; color is never the only indicator.
- Learning-rate curves use different line styles in addition to different colors.
- Plot colors are selected from a color-vision-friendly palette.
- Callout backgrounds are transparent and use inherited text color so they remain readable in light and dark notebook themes.
- Section headings use high-contrast foreground/background combinations.

## Reproducibility

- NumPy random seeds are fixed.
- All learning-rate runs use the same dataset and initial weights.
- Sigmoid and binary cross-entropy implementations include numerical-safety guards.
- The notebook is executed top-to-bottom and saved with outputs and execution counts.
- Hyperparameter comparisons change one variable at a time.


## Daily Stand-up

The notebook includes a compact Day 3 stand-up section covering:

- Completed work
- Next work
- Current blockers
- Mentor-review status

This keeps the notebook aligned with the Sprint 1 workflow rather than treating the day as an isolated lesson.

## Where This Leads Next

Day 4 moves from transparent NumPy mechanics to TensorFlow/Keras. The next notebook should build, compile, train, and evaluate the Cardiac Patient Monitoring System neural network, inspect training and validation curves, add dropout and/or batch normalization, and compare test performance to the Day 1 baseline.
