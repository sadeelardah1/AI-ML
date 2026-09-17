# Technical Write-Up
## IMDb Sentiment Analysis Deployment Project

## 1. Problem Statement
The goal is to classify IMDb movie reviews as positive or negative and expose the trained model through a public Streamlit application.

## 2. Dataset
IMDb movie review text with binary sentiment labels:
```text
positive
negative
```

## 3. Methodology
```text
Raw Review
    ↓
Text Preprocessing
    ↓
TF-IDF
    ↓
Logistic Regression
    ↓
Prediction
```

### Preprocessing
- HTML cleanup
- Lowercasing
- Negation normalization
- Tokenization
- Alphabetic filtering
- Stopword removal
- Negation preservation
- Lemmatization

### Feature Extraction
TF-IDF converts cleaned text into numerical features. The fitted vectorizer is saved as:
```text
tfidf_vectorizer.joblib
```

### Model
Logistic Regression is used as the classifier and saved as:
```text
sentiment_model.joblib
```

## 4. Results
Documented model accuracy:
```text
Accuracy ≈ 0.856
```

## 5. Deployment
The final application uses:
- Python 3.12.12
- Streamlit
- scikit-learn
- joblib
- NumPy
- NLTK

Public URL:
https://ai-ml-s33x.onrender.com

## 6. Technical Challenges
### Dependency Compatibility
The first deployment used Python 3.14 and failed while installing `pyarrow`. Deploying with Python 3.12.12 solved the compatibility issue.

### File Paths
Deployment files were moved to a clean `deployment/` folder.

### NLTK Resources
Required NLTK resources are downloaded in the cloud environment.

## 7. Limitations
- TF-IDF has limited semantic understanding.
- Sarcasm can be difficult.
- The current model is English-only.
- No model monitoring.
- No prediction history.
- Limited explainability.

## 8. Future Work
- BERT or another transformer
- Multilingual sentiment analysis
- SHAP / feature explanations
- Model monitoring
- Automated testing
- CI/CD
- Better UI

## 9. Conclusion
This project demonstrates an end-to-end ML workflow:
```text
Preprocessing
→ Feature Extraction
→ Model
→ API
→ UI
→ Public Deployment
```
