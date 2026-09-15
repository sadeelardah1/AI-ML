import streamlit as st
import joblib

from preprocessing import preprocess_to_string


# Load saved objects
vectorizer = joblib.load("tfidf_vectorizer.joblib")
model = joblib.load("sentiment_model.joblib")


# Page title
st.title("IMDb Sentiment Analysis")

st.write(
    "Enter a movie review below and the model will predict "
    "whether the sentiment is positive or negative."
)


# User input
review = st.text_area(
    "Enter your movie review:"
)


# Prediction button
if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        # Preprocess the review
        clean_review = preprocess_to_string(review)

        # Convert text to TF-IDF features
        X = vectorizer.transform([clean_review])

        # Predict sentiment
        prediction = model.predict(X)[0]

        # Get prediction probabilities
        probabilities = model.predict_proba(X)[0]

        # Find the probability of the predicted class
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index]

        # Display prediction
        st.success(f"Prediction: {prediction}")

        # Display confidence
        st.write(f"Confidence: {confidence * 100:.2f}%")

        # Visualization
        st.progress(float(confidence))