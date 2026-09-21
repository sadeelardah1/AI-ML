# BinX Tech AI & Machine Learning Internship Portfolio

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?logo=scikitlearn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployment-FF4B4B?logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Weeks%201--9-Completed-success)

> A structured portfolio documenting my hands-on journey through Python, data analysis, mathematics for machine learning, classical ML, unsupervised learning, deep learning, NLP, computer vision, model evaluation, explainability, APIs, dashboards, and public deployment.

---

## Table of Contents

- [Program Overview](#program-overview)
- [Learning Journey](#learning-journey)
- [Repository Structure](#repository-structure)
- [Week 1 — Python & Data Science Foundations](#week-1--python--data-science-foundations)
- [Week 2 — Math Foundations & Exploratory Data Analysis](#week-2--math-foundations--exploratory-data-analysis)
- [Week 3 — Supervised Machine Learning](#week-3--supervised-machine-learning)
- [Week 4 — Model Validation, Tuning & Pipelines](#week-4--model-validation-tuning--pipelines)
- [Week 5 — Unsupervised Learning & Capstone Kickoff](#week-5--unsupervised-learning--capstone-kickoff)
- [Week 6 — Deep Learning Fundamentals & Sprint 1](#week-6--deep-learning-fundamentals--sprint-1)
- [Week 7 — Deep Learning Architectures](#week-7--deep-learning-architectures)
- [Week 8 — NLP, Computer Vision, Evaluation & Explainability](#week-8--nlp-computer-vision-evaluation--explainability)
- [Week 9 — Model Deployment & Project Polish](#week-9--model-deployment--project-polish)
- [Final Deployed Project — IMDb Sentiment Analysis](#final-deployed-project--imdb-sentiment-analysis)
- [Selected Results](#selected-results)
- [Technical Skills](#technical-skills)
- [Tools & Technologies](#tools--technologies)
- [How to Run the Repository](#how-to-run-the-repository)
- [Key Engineering Practices](#key-engineering-practices)
- [Challenges & Lessons Learned](#challenges--lessons-learned)
- [Future Improvements](#future-improvements)
- [Final Reflection](#final-reflection)

---

## Program Overview

This repository contains the work completed during **Weeks 1–9 of the BinX Tech AI & Machine Learning Internship Program**.

The work progresses from foundational Python programming to complete machine-learning systems that can be trained, evaluated, explained, exposed through APIs, presented through an interactive interface, and deployed publicly.

The overall progression is:

```text
Python Foundations
        ↓
NumPy + Pandas + Visualization
        ↓
Statistics + Probability + Linear Algebra
        ↓
Exploratory Data Analysis
        ↓
Supervised Machine Learning
        ↓
Validation + Tuning + Pipelines
        ↓
Unsupervised Learning
        ↓
Deep Learning
        ↓
CNNs + RNNs + LSTMs + Transformers
        ↓
NLP + Computer Vision
        ↓
Evaluation + Explainability
        ↓
Model Serialization
        ↓
FastAPI
        ↓
Streamlit
        ↓
Public Deployment
```

---

## Learning Journey

| Week | Main Focus | Main Outcome |
|---|---|---|
| **Week 1** | Python & Data Science Foundations | Reproducible Python environment, NumPy, Pandas, Matplotlib |
| **Week 2** | Math Foundations & EDA | Statistics, probability, linear algebra, univariate/bivariate EDA |
| **Week 3** | Supervised Learning | Regression, classification, model comparison, end-to-end ML workflow |
| **Week 4** | Validation & Tuning | Train/validation/test, cross-validation, bias-variance, GridSearchCV, pipelines |
| **Week 5** | Unsupervised Learning | K-Means, DBSCAN, hierarchical clustering, PCA, t-SNE, anomaly detection |
| **Week 6** | Deep Learning Fundamentals | Neural networks, backpropagation, TensorFlow/Keras, tuning and callbacks |
| **Week 7** | Deep Learning Architectures | CNNs, transfer learning, RNN/LSTM, attention and Transformers |
| **Week 8** | Applied NLP/CV & Evaluation | Text preprocessing, TF-IDF, OpenCV, error analysis, SMOTE, SHAP |
| **Week 9** | Deployment & MLOps | Serialization, FastAPI, Streamlit, Render deployment, repository polish |

---

## Repository Structure

A clean top-level organization for this work is:

```text
AI-ML-Internship/
│
├── Week 1/
│   ├── Day 1/
│   ├── Day 2/
│   ├── Day 3/
│   ├── Day 4/
│   └── Day 5/
│
├── Week 2/
│   ├── Day 1/
│   ├── Day 2/
│   ├── Day 3/
│   ├── Day 4/
│   └── Day 5/
│
├── Week 3/
├── Week 4/
├── Week 5/
├── Week 6/
├── Week 7/
├── Week 8/
├── Week 9/
│
└── README.md
```

Each week contains its notebooks, source files, datasets or dataset references, daily documentation, and supporting artifacts.

---

# Week 1 — Python & Data Science Foundations

**Goal:** build a reliable Python workflow and develop the core data-handling skills required for machine learning.

### Day-by-Day

| Day | Focus | Main Work |
|---|---|---|
| **Day 1** | Environment Setup & Jupyter | Python/pip verification, virtual environment, dependencies, Jupyter workflow, Git |
| **Day 2** | Python Fundamentals | Data types, control flow, functions, list comprehensions, OOP basics |
| **Day 3** | NumPy | Arrays, shapes, indexing, slicing, vectorization, Boolean masks, broadcasting |
| **Day 4** | Pandas | Series/DataFrames, CSV loading, filtering, cleaning, `groupby`, aggregation |
| **Day 5** | Matplotlib & Integration | Line/scatter/bar/histogram plots and a NumPy → Pandas → visualization mini-workflow |

### Highlights

- Created a reproducible Python environment.
- Learned to combine Markdown, code, and outputs inside Jupyter notebooks.
- Used NumPy vectorized operations instead of manual loops.
- Practiced data cleaning and aggregation with Pandas.
- Built clear visualizations with Matplotlib.
- Used Git/GitHub as part of the development workflow.

### Core Pipeline

```text
Load Data
   ↓
Inspect
   ↓
Clean
   ↓
Process
   ↓
Analyze
   ↓
Visualize
```

---

# Week 2 — Math Foundations & Exploratory Data Analysis

**Goal:** understand the mathematical ideas behind machine learning and apply them through exploratory data analysis.

### Day-by-Day

| Day | Focus | Main Work |
|---|---|---|
| **Day 1** | Descriptive Statistics | Mean, median, mode, variance, standard deviation, quartiles, IQR |
| **Day 2** | Probability & Distributions | Probability rules, conditional probability, Bayes' theorem, common distributions |
| **Day 3** | Linear Algebra | Vectors, matrices, dot products, matrix multiplication, shape reasoning |
| **Day 4** | EDA Part 1 | Histograms, box plots, count plots, KDE, outlier detection with IQR |
| **Day 5** | EDA Part 2 | Bivariate analysis, correlation, heatmaps, pairplots, data storytelling |

### Key Concepts

```text
Statistics
    +
Probability
    +
Linear Algebra
    ↓
Exploratory Data Analysis
    ↓
Better Modeling Decisions
```

### Important Lessons

- Outliers can strongly affect the mean and variance.
- EDA should happen before model training.
- Correlation is useful for detecting relationships, but **correlation does not imply causation**.
- Matrix shapes must align correctly for mathematical operations.
- Visual analysis helps reveal skew, anomalies, class imbalance, and relationships before modeling.

---

# Week 3 — Supervised Machine Learning

**Goal:** move from analysis into predictive modeling using Scikit-learn.

### Day-by-Day

| Day | Focus | Main Work |
|---|---|---|
| **Day 1** | Supervised Learning & Scikit-learn API | Features `X`, target `y`, regression vs classification, train/test split |
| **Day 2** | Linear Regression | Training, predictions, coefficients, MAE, RMSE, R², baseline comparison |
| **Day 3** | Logistic Regression | Classification, confusion matrix, precision, recall, F1, ROC-AUC |
| **Day 4** | Model Families | Decision Trees, Random Forests, SVMs, k-NN, fair model comparison |
| **Day 5** | End-to-End Mini-Project | EDA, preprocessing, multiple models, baseline, evaluation, conclusion |

### Standard Scikit-learn Workflow

```python
model = Model(...)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

But the complete workflow became:

```text
Problem Definition
       ↓
EDA
       ↓
Preprocessing
       ↓
Train/Test Split
       ↓
Baseline
       ↓
Train Models
       ↓
Evaluate
       ↓
Compare
       ↓
Document Result
```

### Week 3 Mini-Project

The final mini-project compared:

- Majority-class `DummyClassifier`
- Logistic Regression
- Random Forest

The **Random Forest** produced the strongest F1-score on the saved project run, demonstrating the value of non-linear models when relationships between features are more complex.

---

# Week 4 — Model Validation, Tuning & Pipelines

**Goal:** improve the reliability of model evaluation and learn how to tune models without leaking information from the test set.

### Day-by-Day

| Day | Focus | Main Work |
|---|---|---|
| **Day 1** | Train / Validation / Test | Three-way splitting and final-test discipline |
| **Day 2** | Cross-Validation | More stable model estimation across multiple folds |
| **Day 3** | Bias vs Variance | Diagnosing underfitting and overfitting |
| **Day 4** | Feature Engineering & Tuning | Feature engineering, hyperparameters, `GridSearchCV` |
| **Day 5** | Scikit-learn Pipelines | Full preprocessing + model pipeline tuned as one object |

### Diagnostic Examples

A deliberately overfit tree showed approximately:

```text
Train F1 ≈ 0.97
Validation F1 ≈ 0.50
→ Large generalization gap
```

A deliberately underfit decision stump showed approximately:

```text
Train F1 ≈ 0.58
Validation F1 ≈ 0.56
→ Both scores are low
```

### Hyperparameter Tuning

A Random Forest grid search produced:

```text
Best parameters:
max_depth = 10
n_estimators = 200

Untuned F1 ≈ 0.515
Tuned F1   ≈ 0.517
```

The improvement was small, which is an important real-world lesson: **tuning does not guarantee a dramatic improvement**.

The final full-pipeline exercise produced:

```text
Cross-validated F1 ≈ 0.547

Held-out test:
Baseline F1 ≈ 0.420
Tuned F1    ≈ 0.435
```

---

# Week 5 — Unsupervised Learning & Capstone Kickoff

**Goal:** discover structure without labels and expand the ML toolkit beyond supervised prediction.

### Day-by-Day

| Day | Focus | Main Work |
|---|---|---|
| **Day 1** | K-Means | Clustering, scaling, elbow method, silhouette score |
| **Day 2** | DBSCAN & Hierarchical Clustering | Density-based clustering, noise detection, dendrograms |
| **Day 3** | PCA | Dimensionality reduction and explained variance |
| **Day 4** | t-SNE & Isolation Forest | Non-linear visualization and anomaly detection |
| **Day 5** | Unsupervised Review + Capstone Kickoff | Integrated learning and heart-disease project evaluation |

### Clustering Comparison

On the heart-disease dataset:

```text
K-Means:
Chosen k = 2

DBSCAN:
Clusters = 2
Noise points = 201 / 918

Hierarchical Clustering:
Selected cut produced 5 clusters
```

K-Means silhouette scores included:

```text
k=2 → 0.218
k=3 → 0.178
k=4 → 0.181
k=5 → 0.187
k=6 → 0.164
```

So `k=2` was the strongest K-Means choice among the tested values.

### Heart-Disease Project Result

A full supervised comparison on the heart-disease dataset selected **Random Forest** as the final model:

| Metric | Random Forest |
|---|---:|
| Accuracy | **89.13%** |
| Precision | **90.20%** |
| Recall | **90.20%** |
| F1 Score | **90.20%** |
| ROC-AUC | **92.91%** |

An RBF SVM achieved a higher ROC-AUC (**94.33%**) and was documented as a strong alternative when minimizing false negatives is especially important.

---

# Week 6 — Deep Learning Fundamentals & Sprint 1

**Goal:** understand how neural networks learn and apply those concepts in TensorFlow/Keras.

### Day-by-Day

| Day | Focus | Main Work |
|---|---|---|
| **Day 1** | Sprint Planning & Neural Architecture | Baseline-first planning, neurons, layers, architecture |
| **Day 2** | Activations, Forward Pass & Loss | ReLU/sigmoid concepts, predictions, binary cross-entropy |
| **Day 3** | Backpropagation & Optimization | Chain rule, gradients, gradient descent, learning rate, optimizers |
| **Day 4** | TensorFlow/Keras Project | Build, regularize, train, and evaluate a neural network |
| **Day 5** | Tuning & Sprint Review | Learning-rate/architecture/dropout/batch-size sweeps, callbacks, retrospective |

### Deep Learning Training Loop

```text
Input
  ↓
Forward Propagation
  ↓
Prediction
  ↓
Loss
  ↓
Backpropagation
  ↓
Gradients
  ↓
Optimizer Update
  ↓
Repeat
```

### Cardiac Patient Monitoring System

Day 4 used the **CDC BRFSS 2015** health dataset:

```text
253,680 rows
22 columns
Binary target: HeartDiseaseorAttack
```

The regularized Keras architecture used:

```text
Input(21)
  ↓
Dense(64, ReLU)
  ↓
Batch Normalization
  ↓
Dropout(0.3)
  ↓
Dense(32, ReLU)
  ↓
Batch Normalization
  ↓
Dropout(0.2)
  ↓
Dense(1, Sigmoid)
```

The dataset was highly imbalanced, so F1 and ROC-AUC were treated as more informative than accuracy alone.

### Day 4 Comparison

| Model | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Logistic Regression | 0.7533 | 0.3791 | 0.8470 |
| Regularized Neural Network | 0.7299 | 0.3660 | **0.8493** |

The neural network was competitive with the classical baseline and slightly stronger on ROC-AUC.

### Day 5 Engineering Improvements

- One-variable-at-a-time hyperparameter sweeps
- `EarlyStopping`
- `ModelCheckpoint`
- Validation-loss monitoring
- Best-epoch restoration
- Reproducible relative paths
- Sprint Review and Retrospective

---

# Week 7 — Deep Learning Architectures

**Goal:** move beyond dense neural networks into specialized architectures for images and sequences.

### Day-by-Day

| Day | Focus | Main Work |
|---|---|---|
| **Day 1** | CNN Foundations | Convolution, kernels, feature maps, stride, padding, parameter sharing |
| **Day 2** | CNNs & Transfer Learning | CNN from scratch, augmentation, MobileNetV2, fine-tuning |
| **Day 3** | RNNs & LSTMs | Sequential text, embeddings, recurrent memory, vanishing gradients |
| **Day 4** | Attention & Transformers | Self-attention, Transformers, Hugging Face, DistilBERT |
| **Day 5** | Model Advancement & Sprint Review | Controlled MobileNetV2 experiments, threshold analysis, final review |

---

## CNN Project — Skin Lesion Classification

Classes:

```text
Benign
Malignant
```

Dataset:

| Split | Benign | Malignant | Total |
|---|---:|---:|---:|
| Train | 6,289 | 5,590 | 11,879 |
| Test | 1,000 | 1,000 | 2,000 |
| **Total** | **7,289** | **6,590** | **13,879** |

A simple Day 1 convolution demonstration applied a `3 × 3` vertical-edge kernel to a real lesion image.

The notebook used this example to show **parameter sharing**: the same 9 filter values are reused across the image instead of assigning a separate weight to every image position.

### Day 2 Models

1. CNN from scratch
2. CNN + data augmentation
3. MobileNetV2 transfer learning
4. Fine-tuned MobileNetV2

Evaluation included:

- Accuracy
- AUC
- Precision
- Recall
- Classification report
- Confusion matrix
- Training time

### Day 5 Final Controlled Experiment

| Metric | Final Day 5 Result |
|---|---:|
| Accuracy | **0.8375** |
| AUC | **0.9513** |
| Precision | **0.9400** |
| Recall | **0.7210** |

The Day 2 reference had:

```text
Accuracy = 0.8625
AUC      = 0.9499
```

The final experiment slightly improved ranking performance measured by AUC, but accuracy and recall decreased. This was documented rather than hidden—an important model-selection lesson.

---

## Sequential NLP — RNN & LSTM

The IMDb sequence experiment used:

```text
Vocabulary size = 10,000
Max sequence length = 200
Embedding dimension = 64
Batch size = 128
Epochs = 6
```

The saved notebook captured early results for:

| Model | Epoch 1 Validation Accuracy |
|---|---:|
| Non-sequential embedding baseline | 0.8386 |
| Simple RNN | 0.5626 |

**Important:** the saved Day 3 copy had not yet executed the remaining LSTM/optional GRU and final comparison sections, so no final RNN/LSTM winner is claimed here.

---

## Transformers — DistilBERT

Day 4 applied a pre-trained Hugging Face DistilBERT sentiment model to reconstructed IMDb review text.

The default evaluation used a **balanced 1,000-review subset** of the IMDb test data:

| Metric | DistilBERT |
|---|---:|
| Accuracy | **0.8850** |
| Precision | **0.9365** |
| Recall | **0.8260** |
| F1 | **0.8778** |
| AUC | **0.9593** |

> The Transformer result should not be treated as a perfectly equal benchmark against Day 3 models unless both are evaluated on the same test population. Day 3 used the full 25,000-review test set, while this default DistilBERT run used 1,000 balanced reviews.

---

# Week 8 — NLP, Computer Vision, Evaluation & Explainability

**Goal:** apply reusable preprocessing and evaluation principles across text, images, and highly imbalanced tabular data.

### Day-by-Day

| Day | Focus | Main Work |
|---|---|---|
| **Day 1** | NLP Preprocessing | HTML cleanup, normalization, tokenization, stopwords, negation handling, lemmatization |
| **Day 2** | Text Representation | Bag of Words concepts, TF-IDF, embeddings, Logistic Regression baseline |
| **Day 3** | Computer Vision Preprocessing | OpenCV, resize, BGR→RGB, normalization, augmentation |
| **Day 4** | Integration & Error Analysis | Reusable prediction pipeline, confusion matrix, misclassification analysis |
| **Day 5** | Evaluation & Explainability | Imbalance-aware metrics, SMOTE, threshold tuning, SHAP |

---

## IMDb Sentiment Pipeline

Day 1 built a task-aware preprocessing workflow:

```text
Raw Review
    ↓
HTML Cleanup
    ↓
Lowercasing
    ↓
Negation Normalization
    ↓
Tokenization
    ↓
Filtering
    ↓
Stopword Removal
(negations preserved)
    ↓
Lemmatization
    ↓
Clean Review
```

### TF-IDF Baseline

A subset of IMDb reviews was transformed into:

```text
5,000 reviews
×
5,000 TF-IDF features
```

The resulting matrix had:

```text
Sparsity = 98.4101%
```

The Logistic Regression baseline achieved:

```text
Accuracy = 85.6%
```

Classification report:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Negative | 0.87 | 0.83 | 0.85 |
| Positive | 0.84 | 0.88 | 0.86 |

This pipeline became the foundation of the model deployed in Week 9.

---

## Computer Vision Preprocessing

The reusable OpenCV workflow was:

```text
Read Image
   ↓
Resize
   ↓
BGR → RGB
   ↓
float32
   ↓
Normalize
   ↓
Model-ready Image
```

The work also covered:

- Edge detection
- Data augmentation
- Transfer-learning preprocessing rules
- Training/serving consistency
- End-to-end prediction functions
- Misclassification inspection
- Distinguishing data issues from model weaknesses

---

## Credit Card Fraud Detection

Day 5 focused on the fact that **accuracy can be misleading on highly imbalanced data**.

Evaluation therefore emphasized:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC

The workflow compared a baseline Logistic Regression model with a version trained after **SMOTE** resampling, then explored **decision-threshold tuning**.

The notebook also introduced **SHAP** for:

- Global feature importance
- Local prediction explanations

---

# Week 9 — Model Deployment & Project Polish

**Goal:** turn a trained machine-learning model into a usable public application.

The deployed project is:

# IMDb Sentiment Analysis

```text
Movie Review
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorizer
     ↓
Logistic Regression
     ↓
Positive / Negative
     ↓
Confidence
```

### Day-by-Day

| Day | Focus | Main Work |
|---|---|---|
| **Day 1** | Model Serialization & MLOps | Save/load model and vectorizer, reusable preprocessing, reproducibility |
| **Day 2** | FastAPI | REST API, Pydantic validation, `/predict`, Swagger testing |
| **Day 3** | Streamlit | Interactive UI, text input, prediction, confidence visualization |
| **Day 4** | Public Deployment | Requirements, cloud environment debugging, Render hosting |
| **Day 5** | Repository Polish | Documentation, Definition of Done, Sprint Review, retrospective |

---

## Model Serialization

Instead of training during every request, the trained objects were saved and reused:

```text
tfidf_vectorizer.joblib
sentiment_model.joblib
preprocessing.py
```

This keeps inference fast and prevents training-serving inconsistencies.

---

## FastAPI Serving Layer

The API pipeline is:

```text
Incoming Review
      ↓
Pydantic Validation
      ↓
preprocess_to_string()
      ↓
vectorizer.transform()
      ↓
model.predict()
      ↓
JSON Response
```

The API was tested using FastAPI's interactive Swagger documentation.

---

## Streamlit Interface

The user-facing app includes:

- Movie-review text area
- Predict button
- Positive/Negative result
- Prediction confidence
- Confidence progress bar

Prediction flow:

```text
User Review
    ↓
preprocess_to_string()
    ↓
TF-IDF
    ↓
Logistic Regression
    ↓
Prediction
    ↓
predict_proba()
    ↓
Prediction + Confidence
```

---

## Public Deployment

The Streamlit project was deployed successfully on **Render**.

**Live Demo:**  
https://ai-ml-s33x.onrender.com

Deployment work included:

- `requirements.txt`
- Python runtime compatibility
- Cloud file paths
- NLTK resources
- Serialized model artifacts
- Public application testing

One important deployment issue was Python/package compatibility. The cloud build was moved to **Python 3.12.12** after a newer Python environment caused a `pyarrow` installation problem.

---

# Final Deployed Project — IMDb Sentiment Analysis

## Problem

Automatically determine whether an IMDb movie review expresses **positive** or **negative** sentiment.

## End-to-End Architecture

```text
                 TRAINING
                    │
IMDb Reviews        │
     ↓              │
Preprocessing       │
     ↓              │
TF-IDF Fit          │
     ↓              │
Logistic Regression │
     ↓              │
Saved Artifacts ────┘
     │
     ├── tfidf_vectorizer.joblib
     └── sentiment_model.joblib

                 SERVING
                    │
User Review          │
     ↓               │
Shared Preprocessing │
     ↓               │
Saved TF-IDF         │
     ↓               │
Saved Model          │
     ↓               │
Prediction           │
     ↓               │
Confidence           │
     ↓               │
FastAPI / Streamlit  │
     ↓               │
Render Public URL ───┘
```

## Model

```text
Text Representation: TF-IDF
Classifier: Logistic Regression
Accuracy: 85.6%
```

## Why This Approach?

**TF-IDF** was a strong choice because it is:

- Fast
- Sparse and memory efficient
- Easy to interpret
- Strong as a classical NLP baseline

**Logistic Regression** works well with sparse high-dimensional text features and provides class probabilities through `predict_proba()`.

## Limitations

- TF-IDF has limited semantic understanding.
- Sarcasm and subtle context can be difficult.
- Current sentiment workflow is English-only.
- No production model monitoring.
- No prediction-history storage.
- Explainability can be expanded further.

## Future Work

- BERT or another Transformer-based model
- Multilingual sentiment classification
- SHAP/feature explanations in the deployed UI
- Automated tests
- CI/CD
- Model monitoring
- Improved application UI
- Prediction logging and analytics

---

# Selected Results

| Project / Experiment | Result |
|---|---|
| Week 4 Overfit Tree | Train F1 ≈ 0.97 vs Validation F1 ≈ 0.50 |
| Week 4 Tuned Random Forest | F1 ≈ 0.517 vs baseline ≈ 0.515 |
| Week 5 Heart-Disease Random Forest | Accuracy 89.13%, F1 90.20%, ROC-AUC 92.91% |
| Week 5 Heart-Disease SVM Alternative | ROC-AUC 94.33% |
| Week 6 BRFSS Logistic Regression | Accuracy 0.7533, F1 0.3791, AUC 0.8470 |
| Week 6 Regularized Neural Network | Accuracy 0.7299, F1 0.3660, AUC 0.8493 |
| Week 7 Skin Lesion Day 5 | Accuracy 0.8375, AUC 0.9513 |
| Week 7 DistilBERT | Accuracy 0.8850, F1 0.8778, AUC 0.9593 on 1,000 balanced reviews |
| Week 8 IMDb TF-IDF + Logistic Regression | Accuracy 85.6%, F1 0.85/0.86 |
| Week 9 Deployment | Public Streamlit application deployed on Render |

---

# Technical Skills

## Python & Data

- Python fundamentals
- Functions and OOP basics
- NumPy
- Pandas
- Data cleaning
- Data aggregation
- Matplotlib
- Seaborn

## Mathematics & Analysis

- Descriptive statistics
- Probability
- Bayes' theorem
- Linear algebra
- Exploratory Data Analysis
- Outlier detection
- Correlation analysis
- Data storytelling

## Classical Machine Learning

- Linear Regression
- Logistic Regression
- Decision Trees
- Random Forests
- Support Vector Machines
- k-Nearest Neighbors
- Baselines
- Train/test/validation splitting
- Cross-validation
- Feature engineering
- Hyperparameter tuning
- GridSearchCV
- Scikit-learn Pipelines

## Unsupervised Learning

- K-Means
- Elbow method
- Silhouette score
- DBSCAN
- Hierarchical clustering
- PCA
- t-SNE
- Isolation Forest

## Deep Learning

- Neural-network architecture
- Forward propagation
- Activation functions
- Binary cross-entropy
- Backpropagation
- Gradient descent
- Optimizers
- TensorFlow / Keras
- Dropout
- Batch Normalization
- EarlyStopping
- ModelCheckpoint

## Computer Vision

- OpenCV
- Image resizing
- BGR/RGB conversion
- Normalization
- Data augmentation
- CNNs
- Convolution
- Pooling
- Transfer learning
- MobileNetV2
- Fine-tuning

## Natural Language Processing

- Text cleaning
- Tokenization
- Stopword handling
- Negation preservation
- Lemmatization
- TF-IDF
- Word embeddings concepts
- RNN
- LSTM
- Attention
- Transformers
- Hugging Face
- DistilBERT

## Evaluation & Explainability

- Accuracy
- MAE
- RMSE
- R²
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion matrix
- Threshold tuning
- Error analysis
- SMOTE
- SHAP

## Deployment & Engineering

- Git / GitHub
- Model serialization with Joblib
- Reusable preprocessing modules
- FastAPI
- Pydantic
- Swagger UI
- Streamlit
- Render
- Dependency management
- Reproducibility
- Sprint Review
- Retrospective
- Technical documentation

---

# Tools & Technologies

| Category | Tools |
|---|---|
| Language | Python |
| Notebook Environment | Jupyter Notebook, Google Colab |
| Data | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow, Keras |
| NLP | NLTK, TF-IDF, Hugging Face Transformers |
| Computer Vision | OpenCV |
| Imbalanced Learning | SMOTE / imbalanced-learn |
| Explainability | SHAP |
| Model Persistence | Joblib |
| Backend/API | FastAPI, Pydantic, Uvicorn |
| Frontend/Demo | Streamlit |
| Deployment | Render |
| Version Control | Git, GitHub |

---

# How to Run the Repository

Because each week contains different projects and dependencies, use the README inside the relevant day folder for the most precise instructions.

A common setup is:

```bash
git clone <repository-url>
cd <repository-folder>

python -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install the dependencies required by the selected project.

For notebook-based work:

```bash
jupyter notebook
```

For the Week 9 FastAPI service:

```bash
uvicorn main:app --reload
```

For the Week 9 Streamlit app:

```bash
streamlit run app.py
```

> Run commands from the folder containing the corresponding `main.py` or `app.py`.

---

# Key Engineering Practices

The training emphasized several practices that apply beyond a single model:

### 1. Prevent Data Leakage

```text
Split first
→ fit preprocessing on training data only
→ apply learned transformations to validation/test data
```

### 2. Always Use a Baseline

A metric has little meaning without a reference point.

### 3. Keep the Test Set Final

Model selection and tuning should use training/validation data. The held-out test set should be used only for final evaluation.

### 4. Use the Right Metric

Examples:

```text
Regression:
MAE / RMSE / R²

Balanced Classification:
Accuracy + F1 + confusion matrix

Imbalanced Classification:
Precision + Recall + F1 + ROC-AUC + PR-AUC
```

### 5. Reuse the Exact Training Preprocessing

The deployed system must use the fitted vectorizer and the same text preprocessing used during training.

Correct:

```python
clean_review = preprocess_to_string(review)
X = vectorizer.transform([clean_review])
```

Incorrect:

```python
vectorizer.fit_transform([review])
```

### 6. Reproducibility Matters

- Fixed random seeds where appropriate
- Pinned/recorded dependencies
- Relative paths
- Saved model artifacts
- Documented experiments
- Version control

### 7. Document Negative Results Too

Not every tuned model improves every metric. Week 7 is an example where AUC increased slightly while accuracy and recall decreased. Reporting this clearly is part of professional ML work.

---

# Challenges & Lessons Learned

## Model Development

- Accuracy alone can hide poor minority-class performance.
- High training performance does not guarantee good generalization.
- A larger neural network is not automatically better than a classical model.
- Hyperparameter tuning can produce only small gains.
- Different metrics may favor different models.

## Data

- Raw text requires consistent preprocessing.
- Images need consistent size, color order, and numeric range.
- Class imbalance changes how model performance should be measured.
- High-dimensional datasets benefit from dimensionality reduction and visualization.

## Deployment

Challenges included:

- Missing dependencies
- Serialized file paths
- NLTK resources in the cloud
- Python/package compatibility
- Repository structure for deployment

These were addressed through:

- Clean deployment folders
- `requirements.txt`
- Relative paths
- Explicit runtime version
- Cloud logs
- Reusable preprocessing
- Saved model/vectorizer artifacts

---

# Future Improvements

The next technical improvements I would prioritize are:

1. Replace the classical IMDb baseline with a fine-tuned Transformer.
2. Add multilingual sentiment support.
3. Add SHAP or token-level explanations to the public application.
4. Add automated unit/integration tests.
5. Add CI/CD for deployment.
6. Add model monitoring and prediction drift tracking.
7. Centralize experiment tracking.
8. Improve the public Streamlit interface.
9. Create a unified root `requirements` or environment strategy where practical.
10. Continue testing models on larger and more realistic datasets.

---

# Final Reflection

Across Weeks 1–9, the work moved from learning how to write and organize Python code to building and deploying complete machine-learning workflows.

The biggest progression was learning that machine learning is not only:

```text
model.fit()
```

A professional workflow includes:

```text
Problem Understanding
        ↓
Data Inspection
        ↓
EDA
        ↓
Preprocessing
        ↓
Baseline
        ↓
Model Development
        ↓
Validation
        ↓
Error Analysis
        ↓
Explainability
        ↓
Reproducibility
        ↓
Deployment
        ↓
Documentation
```

The final IMDb Sentiment Analysis deployment connects these stages into one usable product: a trained model, reusable preprocessing pipeline, API, interactive interface, and public deployment.

---

## Live Demo

**IMDb Sentiment Analysis:**  
https://ai-ml-s33x.onrender.com

---

## Repository Status

```text
Week 1   Python & Data Foundations
Week 2   Math Foundations & EDA
Week 3   Supervised Learning
Week 4   Validation & Tuning
Week 5   Unsupervised Learning
Week 6   Deep Learning Fundamentals
Week 7   Deep Learning Architectures
Week 8   NLP / CV / Evaluation
Week 9   Deployment & Project Polish
```

---

<p align="center">
  <b>From Python fundamentals to a publicly deployed machine-learning application.</b>
</p>
