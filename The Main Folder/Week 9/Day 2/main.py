from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib

from preprocessing import preprocess_to_string

vectorizer = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("sentiment_model.joblib")

app = FastAPI(
    title="IMDb Sentiment Analysis API",
    description="Predicts whether a movie review is positive or negative.",
    version="1.0.0",
)

class ReviewInput(BaseModel):
    review: str = Field(
        ...,
        min_length=1,
        description="A movie review to classify as positive or negative."
    )

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "IMDb Sentiment Analysis API is running."
    }

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
