# Cardiac Patient Monitoring System — Deep Learning Intro (Sprint 1)

**BinX Tech · AI & Machine Learning Internship Program — Week 6, Phase 3: Sprint 1**

A neural network baseline for predicting heart disease / heart attack risk from the CDC BRFSS 2015 health survey, built with TensorFlow/Keras and benchmarked against a classical Logistic Regression baseline.

---

## Project Overview

| | |
|---|---|
| **Sprint** | Phase 3 — Sprint 1 (Week 6 of 10) |
| **Goal** | Build, train, and evaluate a first neural network for the Phase 3 capstone project |
| **Dataset** | `heart_disease_health_indicators_BRFSS2015.csv` (253,680 rows · 22 columns) |
| **Target** | `HeartDiseaseorAttack` (binary classification) |
| **Framework** | TensorFlow / Keras (Sequential API) |
| **Baseline** | Scikit-learn Logistic Regression |

The project follows the standard BinX Tech sprint cycle: Sprint Planning → Daily Stand-ups → Mentor Code Review → Sprint Review → Retrospective.

---

## Repository Structure

```
.
├── Day4.ipynb          # Main notebook — model build, train, tune, evaluate
├── data/
│   └── heart_disease_health_indicators_BRFSS2015.csv
└── README.md          
```

---

## Model Architecture

**Baseline (regularized) network:**

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
- **Scaling:** `StandardScaler`, fit on training data only

---

## Results

| Model | Accuracy | F1 Score | ROC-AUC |
|---|---|---|---|
| Logistic Regression (baseline) | 0.7533 | 0.3791 | 0.8470 |
| Keras Neural Network (regularized) | 0.7299 | 0.3660 | 0.8493 |

Note: Because the dataset is heavily imbalanced (~90.6% negative), accuracy alone is not a reliable metric — F1-score and ROC-AUC are the primary comparison criteria.

---

## Getting Started

### Prerequisites

- Python 3.10+
- Jupyter Notebook / JupyterLab or Google Colab

### Installation

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

**`requirements.txt`**
```
tensorflow>=2.15
scikit-learn
pandas
numpy
matplotlib
```

### Running the notebook

1. Place `heart_disease_health_indicators_BRFSS2015.csv` in the project folder.
2. Launch Jupyter:
   ```bash
   jupyter notebook Day4.ipynb
   ```
3. Run all cells top to bottom (Kernel → Restart & Run All).

---

## Methodology

1. **Baseline** — A fresh Logistic Regression model is trained on this dataset's own split (not reused from earlier, smaller datasets) to give a fair comparison point.
2. **Plain network** — A simple 2-hidden-layer Dense network is trained and its loss/accuracy curves are diagnosed for over/underfitting.
3. **Regularized network** — Batch Normalization and Dropout are added to stabilize training and reduce overfitting; validation curves are compared against the plain network.
4. **Evaluation** — The regularized model is scored on the held-out test set and compared against the baseline using Accuracy, F1-score, and ROC-AUC.

---

## Sprint 1 Acceptance Criteria

- [x] Notebook runs end-to-end without errors
- [x] Code committed to a feature branch
- [x] Pull request reviewed and approved by mentor
- [x] Results documented in Markdown
- [x] Metrics logged and compared against baseline

---



