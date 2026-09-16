from pathlib import Path

import joblib
import nltk
import streamlit as st


# Download required NLTK resources
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


from preprocessing import preprocess_to_string


BASE_DIR = Path(__file__).resolve().parent

vectorizer = joblib.load(
    BASE_DIR / "tfidf_vectorizer.joblib"
)

model = joblib.load(
    BASE_DIR / "sentiment_model.joblib"
)

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