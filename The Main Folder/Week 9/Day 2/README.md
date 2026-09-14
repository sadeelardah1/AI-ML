# Week 9 - Day 2: Serving the Model with FastAPI

## Overview

This project is part of **Week 9 - Sprint 4: Model Deployment** in the BinXTech AI & Machine Learning Internship.

The goal of Day 2 is to serve the trained **IMDb Sentiment Analysis** model through a REST API using **FastAPI**.

The API receives a movie review, validates the input using **Pydantic**, applies the same preprocessing used during training, transforms the text using the saved **TF-IDF vectorizer**, runs the trained **Logistic Regression** model, and returns the prediction as a JSON response.

---

## Day 2 Objectives

By the end of Day 2, I learned how to:

- Build a FastAPI application for a machine learning model.
- Load a serialized model and preprocessing objects.
- Create a `POST /predict` endpoint.
- Validate incoming requests using Pydantic.
- Apply the same preprocessing used during model training.
- Use the saved TF-IDF vectorizer for inference.
- Return predictions as JSON responses.
- Run the API locally using Uvicorn.
- Test endpoints using FastAPI's automatic `/docs` interface.
- Confirm that invalid input is rejected cleanly.

---

## Project Used

### IMDb Sentiment Analysis

The model was prepared during Week 8 and serialized during Week 9 - Day 1.

The prediction pipeline is:

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

During Day 2, this pipeline is exposed through FastAPI:

```text
Client / Website / Mobile App
            ↓
        POST /predict
            ↓
          FastAPI
            ↓
          Pydantic
            ↓
     preprocessing.py
            ↓
  TF-IDF Vectorizer
            ↓
 Logistic Regression
            ↓
       JSON Response
```

---

## Project Structure

```text
Week 9/
│
├── Day 1/
│   ├── Day1.ipynb
│   └── preprocessing.py
│
└── Day 2/
    ├── Day2.ipynb
    ├── main.py
    ├── preprocessing.py
    ├── sentiment_model.joblib
    ├── tfidf_vectorizer.joblib
    └── README.md
```

---

## Technologies Used

- Python
- FastAPI
- Pydantic
- Uvicorn
- Joblib
- Scikit-learn
- TF-IDF
- Logistic Regression
- NLTK
- Jupyter Notebook
- Swagger UI / FastAPI Docs

---

## Main API File

The API is defined in:

```text
main.py
```

Main imports:

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

from preprocessing import preprocess_to_string
```

---

## Loading the Saved Model

The API loads the serialized model and vectorizer created earlier:

```python
vectorizer = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("sentiment_model.joblib")
```

The model is not retrained inside the API.

---

## Pydantic Input Validation

The expected request structure is defined using Pydantic:

```python
class ReviewInput(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        description="A movie review to classify as positive or negative."
    )
```

A valid request looks like:

```json
{
  "review": "This movie was amazing"
}
```

An invalid request such as:

```json
{
  "age": 25
}
```

is rejected because the required `review` field is missing.

---

## Prediction Endpoint

The main endpoint is:

```text
POST /predict
```

Implementation:

```python
@app.post("/predict")
def predict(data: ReviewInput):
    cleaned_review = preprocess_to_string(data.review)

    X = vectorizer.transform([cleaned_review])

    prediction = model.predict(X)[0]

    return {
        "review": data.review,
        "cleaned_review": cleaned_review,
        "prediction": str(prediction),
    }
```

---

## Prediction Flow

```text
Request
  ↓
Pydantic Validation
  ↓
Text Preprocessing
  ↓
TF-IDF Transform
  ↓
Model Prediction
  ↓
JSON Response
```

---

## Important Serving Rule

The API must use the same preprocessing used during training.

Correct:

```python
cleaned_review = preprocess_to_string(data.review)
X = vectorizer.transform([cleaned_review])
```

Incorrect:

```python
vectorizer.fit_transform([data.review])
```

The saved vectorizer already learned its vocabulary during training.

---

## Running the API

Open PowerShell or Terminal inside the Day 2 folder:

```powershell
cd "Week 9\Day 2"
```

Then run:

```powershell
python -m uvicorn main:app --reload
```

If the server starts correctly, Uvicorn will show:

```text
Uvicorn running on http://127.0.0.1:8000
```

---

## Testing the API

Open FastAPI's automatic documentation:

```text
http://127.0.0.1:8000/docs
```

Then open:

```text
POST /predict
```

Click **Try it out** and test the endpoint.

### Positive Example

```json
{
  "review": "This movie was fantastic and I really enjoyed it."
}
```

Example response:

```json
{
  "review": "This movie was fantastic and I really enjoyed it.",
  "cleaned_review": "movie fantastic really enjoy",
  "prediction": "positive"
}
```

---

## Negative Example

```json
{
  "review": "This movie was boring, terrible, and disappointing."
}
```

The API should return a negative sentiment prediction if the model classifies the review as negative.

---

## Invalid Input Test

Example:

```json
{
  "age": 25
}
```

Pydantic should reject the request before it reaches the prediction pipeline.

---

## Status Code

A successful request returns:

```text
200 OK
```

This means the request was processed successfully.

---

## Local URL

The API is currently running locally at:

```text
http://127.0.0.1:8000
```

The interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

`127.0.0.1` is a localhost address, so it is only available on the same computer while the Uvicorn server is running.

Public deployment will be handled later in Week 9.

---

## Common Mistakes

### 1. Retraining inside the API

Do not use:

```python
model.fit(...)
```

The API should load the already trained model.

### 2. Fitting the vectorizer again

Do not use:

```python
vectorizer.fit_transform(...)
```

Use:

```python
vectorizer.transform(...)
```

### 3. Skipping preprocessing

The review must pass through the same preprocessing pipeline used during training.

### 4. Running Uvicorn from the wrong folder

The terminal should be opened in the folder containing:

```text
main.py
preprocessing.py
sentiment_model.joblib
tfidf_vectorizer.joblib
```


---

## Day 2 Deliverables

By the end of Day 2, the project contains:

- A working FastAPI application.
- A `POST /predict` endpoint.
- Pydantic input validation.
- Saved model loading.
- Saved TF-IDF vectorizer loading.
- Reusable preprocessing.
- JSON prediction responses.
- Local Uvicorn server.
- Successful `/docs` testing.
- Invalid input validation.


