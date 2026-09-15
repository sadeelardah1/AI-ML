# Week 9 - Day 3: Interactive Streamlit Dashboard

## Overview

This project is part of **Week 9 - Sprint 4: Model Deployment** in the BinXTech AI & Machine Learning Internship.

The goal of Day 3 is to build an interactive **Streamlit dashboard** for the IMDb Sentiment Analysis model so a non-technical user can enter a movie review and receive a sentiment prediction through a simple web interface.

The application reuses the same trained pipeline prepared in the previous days:

```text
Raw Movie Review
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorizer
        ↓
Logistic Regression
        ↓
Positive / Negative
```

---

## Day 3 Objectives

By the end of Day 3, I learned how to:

- Build an interactive Streamlit application.
- Create a simple user interface for a machine learning model.
- Use Streamlit widgets such as `st.text_area()`, `st.button()`, and `st.success()`.
- Load the saved TF-IDF vectorizer and Logistic Regression model.
- Reuse the same preprocessing used during training.
- Run predictions from user input.
- Use `predict_proba()` to calculate model confidence.
- Display a supporting confidence visualization.
- Test the application using positive and negative movie reviews.
- Run the Streamlit application locally.

---

## Project Used

### IMDb Sentiment Analysis

The application predicts whether a movie review has:

```text
Positive Sentiment
```

or

```text
Negative Sentiment
```

The model used is:

- **TF-IDF Vectorizer** for text representation.
- **Logistic Regression** for sentiment classification.

---

## Project Structure

```text
Week 9/
└── Day 3/
    ├── Day3.ipynb
    ├── app.py
    ├── preprocessing.py
    ├── sentiment_model.joblib
    ├── tfidf_vectorizer.joblib
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
- Jupyter Notebook

---

## Why Streamlit?

FastAPI from Day 2 provides an interface mainly for programs and applications.

Streamlit provides a simple interface for people.

```text
FastAPI
Program
   ↓
API Request
   ↓
JSON Response
```

```text
Streamlit
User
   ↓
Text Area
   ↓
Predict Button
   ↓
Visible Result
```

---

## Loading the Saved Model

The application loads the trained model and vectorizer instead of training them again.

```python
import joblib

vectorizer = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("sentiment_model.joblib")
```

The same text preprocessing used during training is also reused:

```python
from preprocessing import preprocess_to_string
```

---

## Streamlit Interface

The main interface contains:

- A title
- A short description
- A text area for the movie review
- A Predict button
- A sentiment result
- A confidence percentage
- A confidence progress bar

Example:

```python
st.title("IMDb Sentiment Analysis")

review = st.text_area(
    "Enter your movie review:"
)

if st.button("Predict"):
    ...
```

---

## Prediction Flow

The Streamlit application follows this pipeline:

```text
User Review
    ↓
preprocess_to_string()
    ↓
TF-IDF Vectorizer
    ↓
Logistic Regression
    ↓
Prediction
    ↓
Confidence
    ↓
Displayed Result
```

---

## Prediction Logic

The user review is cleaned using the same preprocessing pipeline:

```python
clean_review = preprocess_to_string(review)
```

Then it is transformed using the saved TF-IDF vectorizer:

```python
X = vectorizer.transform([clean_review])
```

The sentiment is predicted using the saved model:

```python
prediction = model.predict(X)[0]
```

The model is not retrained during prediction.

---

## Confidence Visualization

The application also calculates the probability of the predicted class:

```python
probabilities = model.predict_proba(X)[0]

class_index = list(model.classes_).index(prediction)

confidence = probabilities[class_index]
```

The confidence is displayed in Streamlit:

```python
st.write(
    f"Confidence: {confidence * 100:.2f}%"
)

st.progress(float(confidence))
```

This provides a simple supporting visualization for the prediction.

---

## Running the Application

Open PowerShell or Terminal inside the Day 3 folder:

```powershell
cd "Week 9\Day 3"
```

Then run:

```powershell
python -m streamlit run app.py
```

The application will normally open at:

```text
http://localhost:8501
```

Keep the terminal open while the application is running.

---

## Positive Test

Example:

```text
This movie was fantastic, emotional, and beautifully made.
```

Expected result:

```text
Prediction: positive
```

The application also displays the model confidence.

---

## Negative Test

Example:

```text
This movie was boring, terrible, disappointing, and a complete waste of time.
```

Expected result:

```text
Prediction: negative
```

---

## Important Serving Rule

The deployment application must use the same preprocessing used during training.

Correct:

```python
clean_review = preprocess_to_string(review)
X = vectorizer.transform([clean_review])
```

Do not use:

```python
vectorizer.fit_transform([review])
```

The vectorizer was already fitted during training.

---

## What Makes the Demo Usable?

The Streamlit application includes:

- A clear title.
- A short explanation of the task.
- A simple text input.
- A visible Predict button.
- A clear sentiment result.
- A supporting confidence visualization.
- A workflow that a first-time user can understand quickly.

---

## Day 3 Deliverables

By the end of Day 3, the project should include:

- A working Streamlit application.
- A movie review text input.
- A Predict button.
- A sentiment prediction.
- A confidence percentage.
- A confidence progress bar.
- Saved model and vectorizer files.
- Reusable preprocessing.
- Positive and negative test cases.

