# Week 9 - Day 1: Sprint 4 Planning, Model Serialization & MLOps

## Overview

This notebook is part of **Week 9 - Sprint 4: Model Deployment** of the BinXTech AI & Machine Learning Internship.

The goal of Day 1 is to prepare the trained **IMDb Sentiment Analysis** model for deployment.  
Instead of retraining the model every time, the trained model and its preprocessing objects are saved to disk, loaded again, and tested to make sure they can reproduce predictions correctly.

---

## Day 1 Objectives

By the end of this day, I should be able to:

- Define the Sprint 4 deployment goal and backlog.
- Understand why deployment is important in a machine learning project.
- Understand the concept of model serialization.
- Save and load a trained model using `joblib`.
- Save the TF-IDF vectorizer used during training.
- Reuse the same preprocessing pipeline during deployment.
- Avoid training-serving skew.
- Verify that the loaded model can reproduce a known prediction.
- Understand the importance of reproducibility.
- Use pinned dependencies, fixed random seeds, and experiment tracking concepts.
- Prepare the project for FastAPI and Streamlit in the next days.

---

## Project Used

### IMDb Sentiment Analysis

The model used in this notebook was developed during Week 8.

The pipeline is:

```text
Raw Movie Review
        |
        v
Text Preprocessing
        |
        v
TF-IDF Vectorizer
        |
        v
Logistic Regression Model
        |
        v
Positive / Negative Sentiment
```

The trained model is a **Logistic Regression** classifier and the text representation is created using **TF-IDF**.

---

## Project Structure

```text
Week 9/
|
+-- Day 1/
|   |
|   +-- Day1.ipynb
|   +-- preprocessing.py
|   +-- README.md
|   +-- requirements.txt
|
Week 8/
|
+-- Day 2/
    |
    +-- sentiment_model.joblib
    +-- tfidf_vectorizer.joblib
    +-- IMDB_Dataset_Cleaned_Day1.csv
```

---

## Model Serialization

Serialization means saving a trained machine learning object to disk so that another application can load and use it later without retraining.

The TF-IDF vectorizer and the trained model are saved using `joblib`.

```python
import joblib

joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
joblib.dump(model, "sentiment_model.joblib")
```

They can later be loaded using:

```python
loaded_vectorizer = joblib.load("tfidf_vectorizer.joblib")
loaded_model = joblib.load("sentiment_model.joblib")
```

---

## Why Save the Vectorizer?

The model was trained using TF-IDF features.

This means the deployment application must use the **same trained TF-IDF vectorizer** that was used during training.

Using a different vectorizer could create different feature representations and lead to incorrect predictions.

The correct deployment pipeline is:

```text
User Review
    |
    v
Same Text Preprocessing
    |
    v
Saved TF-IDF Vectorizer
    |
    v
Saved Logistic Regression Model
    |
    v
Prediction
```

---

## Training-Serving Skew

Training-serving skew happens when the data preparation process used during deployment is different from the process used during model training.

To avoid this problem:

- Use the same text cleaning function.
- Use the same saved TF-IDF vectorizer.
- Use the same trained model.
- Keep the deployment environment reproducible.

---

## Preprocessing Module

The file:

```text
preprocessing.py
```

contains the reusable IMDb text preprocessing pipeline.

It performs steps such as:

- HTML cleaning
- Lowercasing
- Negation preservation
- Tokenization
- Stop-word removal
- POS-aware lemmatization

The deployment code can reuse it with:

```python
from preprocessing import preprocess_to_string
```

---

## Prediction Test

After loading the saved model and vectorizer, the notebook verifies that they can still make predictions correctly.

Example workflow:

```python
clean_review = preprocess_to_string(review)

review_vector = loaded_vectorizer.transform([clean_review])

prediction = loaded_model.predict(review_vector)[0]
```

This confirms that the serialized files are ready to be used by another application.

---

## Reproducibility

Reproducibility means that the project can be run again in a consistent environment and produce reliable results.

Important practices used in this sprint include:

### 1. Pinned Dependencies

A `requirements.txt` file records the libraries and versions required by the project.

Example:

```text
scikit-learn
pandas
numpy
joblib
nltk
```

### 2. Fixed Random Seeds

Random seeds help make experiments repeatable.

Example:

```python
RANDOM_STATE = 42
```

### 3. Experiment Tracking

Experiment tracking tools such as MLflow can be used to record:

- Model versions
- Parameters
- Metrics
- Experiment runs

---

## Day 1 Workflow

```text
Sprint Planning
      |
      v
Load Week 8 Model
      |
      v
Save Model
      |
      v
Save TF-IDF Vectorizer
      |
      v
Load Saved Objects
      |
      v
Run Known Prediction
      |
      v
Verify Result
      |
      v
Prepare Reproducible Environment
      |
      v
Ready for FastAPI
```

---

## What I Learned

- I learned how to plan Sprint 4 and define the deployment backlog.
- I learned why model deployment is important for turning a trained model into a real application.
- I learned what model serialization means.
- I learned how to save and load machine learning objects using `joblib`.
- I learned why the TF-IDF vectorizer must be saved together with the model.
- I learned why training and deployment must use the same preprocessing pipeline.
- I learned how training-serving skew can affect model predictions.
- I learned how to verify a loaded model using a known prediction.
- I learned the importance of reproducibility in machine learning projects.
- I learned the role of `requirements.txt`, fixed seeds, and experiment tracking.
- I prepared the sentiment analysis model for FastAPI and Streamlit deployment.

---

## Day 1 Deliverables

At the end of Day 1, the project should include:

- Sprint 4 goal and backlog
- Serialized sentiment analysis model
- Serialized TF-IDF vectorizer
- Reusable preprocessing module
- Successful load test
- Successful known prediction test
- Reproducibility setup
- Clean notebook documentation

---

## Next Step

### Day 2 - FastAPI

The next step is to serve the trained sentiment analysis model through a REST API.

The API will:

1. Receive a movie review.
2. Validate the input.
3. Apply the saved preprocessing pipeline.
4. Transform the text using the saved TF-IDF vectorizer.
5. Run the saved Logistic Regression model.
6. Return the sentiment prediction as JSON.

---

## Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- Jupyter Notebook
- Git & GitHub
- MLflow concepts


