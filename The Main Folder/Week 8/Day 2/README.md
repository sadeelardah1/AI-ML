# Day 2 — Text Representation: TF-IDF & Embeddings

This project covers **Week 8 — Day 2** of the AI & Machine Learning internship and focuses on converting cleaned natural-language text into numerical representations that machine learning models can use.

The notebook continues directly from Day 1 by using the cleaned IMDb movie-review dataset and exploring two main text-representation families:

- **Frequency-based representation:** Bag of Words and TF-IDF
- **Semantic representation:** Word Embeddings

The notebook combines clear explanations, small teaching examples, real IMDb data, and a simple sentiment-classification baseline.

---

## Table of Contents

- [Overview](#overview)
- [Learning Objectives](#learning-objectives)
- [Dataset](#dataset)
- [Day 1 to Day 2](#day-1-to-day-2)
- [Topics Covered](#topics-covered)
- [Notebook Workflow](#notebook-workflow)
- [Hands-On Lab](#hands-on-lab)
- [Key Results](#key-results)
- [TF-IDF vs Embeddings](#tf-idf-vs-embeddings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [What I Learned](#what-i-learned)
- [Reference](#reference)

---

## Overview

Day 1 prepared the raw text through preprocessing.

Day 2 answers the next question:

> **How can cleaned text be converted into numbers that a machine learning model can understand?**

The main workflow is:

```text
Clean Text
    |
    v
Text Representation
    |
    +-------------------------+
    |                         |
    v                         v
  TF-IDF               Word Embeddings
    |                         |
    v                         v
Word Importance          Word Meaning
    |                         |
    v                         v
Sparse Vectors           Dense Vectors
```

---

## Learning Objectives

By the end of this notebook, I can:

- Explain why text must be converted into numerical vectors.
- Explain how **Bag of Words** represents documents.
- Distinguish between **Bag of Words** and **Term Frequency**.
- Explain **TF — Term Frequency**.
- Explain **IDF — Inverse Document Frequency**.
- Explain how **TF-IDF** combines frequency and rarity.
- Apply `TfidfVectorizer` to real cleaned text.
- Understand **vocabulary**, **features**, and **sparse matrices**.
- Inspect important TF-IDF terms in movie reviews.
- Explain what **Word Embeddings** represent.
- Understand **dense vectors** and **semantic similarity**.
- Explain the role of **Word2Vec** and **GloVe**.
- Understand why **Contextual Embeddings** are needed.
- Compare **TF-IDF** with **Embeddings**.
- Use TF-IDF features in a simple sentiment-classification model.

---

## Dataset

The notebook uses the cleaned IMDb dataset produced in Day 1:

```text
IMDB_Dataset_Cleaned_Day1.csv
```

### Dataset columns

| Column | Description |
|---|---|
| `review` | Original IMDb movie review |
| `clean_review` | Review after Day 1 preprocessing |
| `sentiment` | `positive` or `negative` label |

### Dataset summary

```text
Rows: 5,000
Columns: 3
Missing values: 0
```

Class distribution:

```text
Positive: 2,519
Negative: 2,481
```

The dataset is nearly balanced.

---

## Day 1 to Day 2

The two days connect as one NLP pipeline:

```text
DAY 1
Raw Text
   |
   v
Tokenization
   |
   v
Cleaning
   |
   v
Stop-word Handling
   |
   v
Lemmatization
   |
   v
Clean Text

DAY 2
Clean Text
   |
   v
Vectorization
   |
   v
Numeric Features
   |
   v
Machine Learning Model
```

A simple way to remember the difference:

> **Day 1 prepares the text. Day 2 represents the text numerically.**

---

## Topics Covered

### 1. From Text to Numbers

Clean text is still text.

A model needs a numerical representation:

```text
movie amazing love
        |
        v
[0.12, 0.00, 0.76, 0.41, ...]
```

This conversion is called **vectorization**.

---

### 2. Bag of Words

Bag of Words represents each document using word counts while ignoring word order.

Example:

```text
D1: movie good
D2: movie bad
D3: movie movie good
```

Vocabulary:

```text
[bad, good, movie]
```

Representation:

| Document | bad | good | movie |
|---|---:|---:|---:|
| D1 | 0 | 1 | 1 |
| D2 | 1 | 0 | 1 |
| D3 | 0 | 1 | 2 |

---

### 3. TF — Term Frequency

TF measures how often a term appears inside one document.

```text
Document: movie good good amazing

movie   -> 1
good    -> 2
amazing -> 1
```

> **TF asks: How frequent is this word in this document?**

---

### 4. IDF — Inverse Document Frequency

IDF measures how rare or common a term is across the collection of documents.

```text
movie       -> common across many reviews -> lower IDF
masterpiece -> more distinctive            -> higher IDF
```

Relationship:

```text
Document Frequency increases -> IDF decreases
Document Frequency decreases -> IDF increases
```

---

### 5. TF-IDF

TF-IDF combines TF and IDF:

```text
TF-IDF = TF x IDF
```

A word receives a stronger score when it is:

- frequent in the current document, and
- relatively rare across the complete dataset.

Example:

```text
movie
-> common across many reviews
-> lower discriminative importance

masterpiece
-> less common
-> more distinctive
-> potentially higher TF-IDF importance
```

---

### 6. Vocabulary, Features & Sparse Matrix

The real IMDb reviews are vectorized using:

```python
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df["clean_review"])
```

The resulting matrix has:

```text
5,000 reviews x 5,000 features
```

Matrix structure:

```text
Rows    = Reviews
Columns = Vocabulary terms
Values  = TF-IDF scores
```

Measured in the notebook:

```text
Total possible values: 25,000,000
Non-zero values:       397,463
Density:               1.5899%
Sparsity:              98.4101%
```

---

### 7. Word Embeddings

TF-IDF represents **importance**.

Embeddings aim to represent **meaning**.

Example:

```text
good -> [0.21, -0.43, 0.87, 0.15, ...]
```

Main difference:

```text
TF-IDF     -> Sparse representation -> Word importance
Embeddings -> Dense representation  -> Semantic meaning
```

---

### 8. Vector Space & Semantic Similarity

In an embedding space, words with similar meanings can be represented by nearby vectors.

```text
good      ~ great      ~ excellent
bad       ~ terrible   ~ awful
```

The notebook also introduces **cosine similarity** to explain how vectors can be compared.

---

### 9. Word2Vec & GloVe

#### Word2Vec

Learns word representations from the contexts in which words appear.

#### GloVe

Learns word representations using global word co-occurrence information.

Both aim to produce:

```text
Word
 |
 v
Dense Vector
 |
 v
Semantic Relationships
```

---

### 10. Contextual Embeddings

A fixed embedding may assign one vector to the word `bank`, even though its meaning changes:

```text
I deposited money in the bank.

We sat beside the river bank.
```

Contextual embeddings allow the representation to depend on the sentence:

```text
bank + financial context -> vector A
bank + river context     -> vector B
```

---

## Notebook Workflow

```text
IMDB_Dataset_Cleaned_Day1.csv
              |
              v
        Load with pandas
              |
              v
         clean_review
              |
              v
      TfidfVectorizer
              |
              v
    5,000 TF-IDF Features
              |
              v
        Sparse Matrix
              |
        +-----+------+
        |            |
        v            v
 Feature Analysis   Sentiment Model
                       |
                       v
              Logistic Regression
```

---

## Hands-On Lab

The practical part begins from:

```text
Section 8 — TF-IDF on the Real IMDb Data
```

The notebook performs the following:

1. Applies TF-IDF to the cleaned reviews.
2. Extracts the learned vocabulary.
3. Measures matrix density and sparsity.
4. Displays a readable subset of the TF-IDF matrix.
5. Finds the highest-weight terms in individual reviews.
6. Compares important terms across positive and negative reviews.
7. Demonstrates semantic vector similarity.
8. Uses TF-IDF features to train a simple sentiment classifier.

---

## Key Results

### TF-IDF representation

```text
Documents: 5,000
Features:  5,000
Sparsity:  98.4101%
```

### Sentiment baseline

Model:

```text
TF-IDF + Logistic Regression
```

Train/test split:

```text
80% Training
20% Testing
```

Result:

```text
Accuracy: 85.6%
```

Classification report:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Negative | 0.87 | 0.83 | 0.85 |
| Positive | 0.84 | 0.88 | 0.86 |

The classifier is used as a **baseline** to demonstrate that TF-IDF vectors can serve directly as machine-learning features.

---

## TF-IDF vs Embeddings

| Feature | TF-IDF | Word Embeddings |
|---|---|---|
| Represents | Word importance | Word meaning |
| Main idea | Frequency + rarity | Semantic relationships |
| Representation | Sparse | Dense |
| Similar-word awareness | No, not directly | Yes |
| Context awareness | No | Contextual embeddings can |
| Typical use | Strong classical baseline | Semantic / deep-learning tasks |

### Main takeaway

> **TF-IDF tells us how important a word is. Embeddings try to represent what the word means.**

---

## Project Structure

Recommended GitHub structure:

```text
Day-2/
├── Day2.ipynb
├── IMDB_Dataset_Cleaned_Day1.csv
├── README.md
└── presentation/
    └── Day2.pptx
```

---

## Requirements

Main Python libraries:

```text
numpy
pandas
matplotlib
scikit-learn
```

Important components used:

```python
CountVectorizer
TfidfVectorizer
train_test_split
LogisticRegression
accuracy_score
classification_report
cosine_similarity
```

---

## How to Run

### 1. Keep the notebook and dataset together

```text
Day2.ipynb
IMDB_Dataset_Cleaned_Day1.csv
```

### 2. Install dependencies if needed

```bash
pip install numpy pandas matplotlib scikit-learn
```

### 3. Open the notebook

You can use:

- Jupyter Notebook
- JupyterLab
- VS Code
- Google Colab

### 4. Run all cells from top to bottom

The notebook will:

- load the cleaned IMDb dataset,
- demonstrate Bag of Words,
- explain TF, IDF, and TF-IDF,
- create the real TF-IDF matrix,
- inspect the vocabulary and important terms,
- explain embeddings and semantic similarity,
- train the sentiment baseline,
- display the evaluation results.

---


## Reference

The notebook follows the practical Python data-analysis workflow used in:


## Final Takeaway

```text
Day 1
Raw Text -> Clean Text

Day 2
Clean Text -> Numeric Representation -> Machine Learning
```

> **TF-IDF represents importance, while embeddings represent meaning.**
