# Week 9 - Day 4: Public Deployment

## Overview

This project is part of **Week 9 - Sprint 4: Model Deployment** in the BinXTech AI & Machine Learning Internship.

The goal of Day 4 is to move the working local Streamlit application from Day 3 to a **public hosting platform** so the application can be accessed through a public URL.

The application used in this deployment is the **IMDb Sentiment Analysis** project.

The deployed application uses:

- The saved text preprocessing pipeline
- The saved TF-IDF vectorizer
- The saved Logistic Regression model
- Streamlit for the user interface
- A public hosting platform for deployment

---

## Day 4 Objectives

By the end of Day 4, I learned how to:

- Prepare a Streamlit application for public deployment.
- Understand the difference between a local application and a public application.
- Create and use a correct `requirements.txt`.
- Include the trained model and preprocessing files in the deployment project.
- Deploy a machine learning application to a public hosting platform.
- Test the deployed application with multiple inputs.
- Compare local predictions with deployed predictions.
- Debug common deployment problems.
- Record the public application URL in the project README.

---

## Project Used

### IMDb Sentiment Analysis

The application predicts whether a movie review is:

```text
Positive
```

or:

```text
Negative
```

The model pipeline is:

```text
Raw Movie Review
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorizer
        ↓
Logistic Regression
        ↓
Sentiment Prediction
```

---

## From Local to Public

Before Day 4, the Streamlit application was running locally:

```text
http://localhost:8501
```

This local URL only works on the same computer.

Day 4 moves the application to a public hosting platform:

```text
Local Streamlit App
        ↓
requirements.txt
        ↓
Deployment Repository
        ↓
Hosting Platform
        ↓
Public URL
```

---

## Deployment Files

The deployment folder contains:

```text
Week 9/
└── Day 4/
    ├── Day4.ipynb
    ├── app.py
    ├── preprocessing.py
    ├── sentiment_model.joblib
    ├── tfidf_vectorizer.joblib
    ├── requirements.txt
    └── README.md
```

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- TF-IDF
- Logistic Regression
- Joblib
- NLTK
- Git
- GitHub
- Public Hosting Platform

---

## requirements.txt

The hosting platform creates a fresh environment.

Because of this, the application dependencies must be listed in:

```text
requirements.txt
```

For this project:

```text
streamlit==1.51.0
scikit-learn==1.7.2
joblib==1.5.2
numpy==2.3.5
nltk==3.9.2
```

Pinned versions help make the deployment environment consistent with the environment used locally.

---

## Model and Preprocessing Files

The deployed application requires the same files used locally:

```text
sentiment_model.joblib
tfidf_vectorizer.joblib
preprocessing.py
```

### Model

```text
sentiment_model.joblib
```

contains the trained Logistic Regression model.

### Vectorizer

```text
tfidf_vectorizer.joblib
```

contains the fitted TF-IDF vectorizer.

### Preprocessing

```text
preprocessing.py
```

contains the same text preprocessing used during training.

Using the same preprocessing helps prevent differences between local and deployed predictions.

---

## NLTK Resources

The local computer already contains the required NLTK resources, but a new cloud environment may not.

The deployment application downloads the required resources when it starts.

Example:

```python
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
```

This allows the preprocessing pipeline to work in the cloud environment.

---

## Application File

The main Streamlit application is:

```text
app.py
```

It loads the saved model and vectorizer:

```python
vectorizer = joblib.load(
    BASE_DIR / "tfidf_vectorizer.joblib"
)

model = joblib.load(
    BASE_DIR / "sentiment_model.joblib"
)
```

The application uses relative paths based on the location of `app.py`.

This makes the code more reliable when running on another machine or hosting platform.

---

## Prediction Pipeline

When a user enters a movie review:

```text
User Review
    ↓
Text Preprocessing
    ↓
TF-IDF Transform
    ↓
Logistic Regression
    ↓
Prediction
    ↓
Confidence
```

The application displays:

- The predicted sentiment
- The prediction confidence
- A confidence progress bar

---

## Local Test Before Deployment

Before publishing the application, the Day 4 version was tested locally using:

```powershell
python -m streamlit run app.py
```

The application opened successfully at:

```text
http://localhost:8501
```

---

## Positive Test

Example:

```text
This movie was fantastic, emotional, and beautifully made.
```

Expected prediction:

```text
positive
```

---

## Negative Test

Example:

```text
This movie was boring, terrible, disappointing, and a complete waste of time.
```

Expected prediction:

```text
negative
```

---

## Local vs Deployed Testing

After deployment, the same reviews should be tested again.

The deployed predictions should match the local predictions.

Example comparison:

| Test | Local Result | Deployed Result |
|---|---|---|
| Positive Review | Positive | Positive |
| Negative Review | Negative | Negative |

If the results are different, the first things to check are:

- Preprocessing
- Model files
- Vectorizer files
- Package versions
- File paths

---

## Hosting Options

The Week 9 deployment plan includes platforms such as:

- Hugging Face Spaces
- Render
- Railway

The goal is the same:

```text
Working Local Application
        ↓
Public Hosting
        ↓
Live Public URL
```

---

## Common Deployment Problems

### 1. Missing Package

Example:

```text
ModuleNotFoundError
```

Check:

```text
requirements.txt
```

---

### 2. Model File Not Found

Check that these files are included:

```text
sentiment_model.joblib
tfidf_vectorizer.joblib
```

Also use correct relative file paths.

---

### 3. NLTK Resource Error

Make sure the required NLTK resources are available in the cloud environment.

---

### 4. Different Local and Deployed Predictions

Check:

- Preprocessing consistency
- Dependency versions
- Correct model file
- Correct vectorizer file

---

### 5. Build Failure

Read the deployment build logs carefully.

The logs usually show which package, file, or configuration caused the failure.

---

## What I Learned Today

- I learned the difference between a local application and a publicly deployed application.
- I learned how a hosting platform creates a fresh environment for the application.
- I learned why `requirements.txt` is important for deployment.
- I learned why pinned dependency versions improve reproducibility.
- I learned how to prepare model and preprocessing files for deployment.
- I learned how to make the Streamlit application portable using relative file paths.
- I learned how to prepare NLTK resources for a cloud environment.
- I learned how to test the application before deployment.
- I learned how to compare local and deployed model predictions.
- I learned how to identify common deployment problems.
- I learned that a public URL allows other users to access the machine learning application.

---

## Day 4 Deliverables

By the end of Day 4, the project should include:

- A working Streamlit application.
- A correct `requirements.txt`.
- Serialized model and TF-IDF files.
- Reusable preprocessing code.
- A successful deployment build.
- A public application URL.
- Local and deployed prediction tests.
- The live URL added to the README.

---

## Live Demo

Public URL:

```text
https://ai-ml-s33x.onrender.com/
```





