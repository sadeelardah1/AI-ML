# Week 9 - Day 5: Repository Polish, Definition of Done & Sprint Review

## Overview
Day 5 is the project close-out day for Week 9 / Sprint 4. The goal is to make the project clean, reproducible, documented, deployed, and ready for review.

## Final Project Pipeline
```text
IMDb Dataset
    ↓
Text Preprocessing
    ↓
TF-IDF Vectorization
    ↓
Logistic Regression
    ↓
Model Evaluation
    ↓
Model Serialization
    ↓
FastAPI
    ↓
Streamlit Dashboard
    ↓
Render Deployment
    ↓
Public URL
```

## Repository Polish
The repository should include:
- Clear folder structure
- README files
- `requirements.txt`
- Reproducible notebooks
- Python scripts
- Serialized model artifacts
- Deployment files
- Public deployment URL

Avoid:
- Unused files
- Debug code
- Temporary files
- Hard-coded local paths
- Unnecessary duplicate artifacts

## Public Deployment
https://ai-ml-s33x.onrender.com

## Definition of Done
- Full pipeline: Complete
- Benchmark documented: Complete
- Public deployment: Complete
- Clean repository: Review before final push
- Experiments documented: In notebooks
- Reproducible notebook: Review before final push

## Sprint 4 Review
### Day 1
- Serialized the trained model
- Saved the TF-IDF vectorizer
- Reused preprocessing
- Verified predictions after reload

### Day 2
- Built FastAPI
- Added `/predict`
- Added Pydantic validation
- Tested with Swagger UI

### Day 3
- Built Streamlit dashboard
- Added prediction
- Added confidence visualization
- Tested positive and negative reviews

### Day 4
- Prepared `requirements.txt`
- Prepared deployment files
- Debugged cloud deployment
- Switched deployment runtime to Python 3.12.12
- Deployed successfully on Render

### Day 5
- Repository polish
- Technical write-up
- Definition of Done review
- Sprint Review
- Full-project retrospective

## Full-Project Retrospective
### What Went Well
- The pipeline stayed consistent from training to deployment.
- Model artifacts were successfully reused.
- FastAPI and Streamlit both worked.
- Deployment issues were solved using logs.
- The application is publicly accessible.

### Challenges
- NLTK resources in the cloud
- File paths
- Python 3.14 and `pyarrow` compatibility
- Repository organization for deployment

### Future Work
- Use BERT or another transformer
- Add multilingual sentiment analysis
- Add explainability
- Add automated tests
- Add CI/CD
- Add model monitoring

## What I Learned Today
- How to polish an ML repository
- How to verify a Definition of Done
- How to write a technical summary
- How to run a Sprint Review
- How to write a retrospective
- Why documentation and reproducibility are part of ML engineering

## Final Status
```text
Trained Model
+ Reusable Preprocessing
+ FastAPI API
+ Streamlit Dashboard
+ Public Deployment
+ Documentation
+ Definition of Done
+ Sprint Review
+ Retrospective
```
