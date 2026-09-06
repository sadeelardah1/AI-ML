# Day 1 — NLP & Text Preprocessing

A professional hands-on notebook for **Week 8 — Day 1** of the BinX Tech AI & Machine Learning Internship.

This day focuses on preparing raw human language for machine learning by building a complete, task-aware text preprocessing pipeline using the **IMDb Dataset of 50K Movie Reviews**.

The notebook explains every concept from first principles, demonstrates each step with code, compares alternative preprocessing approaches, and applies the full pipeline to real sentiment-analysis data.

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Dataset](#dataset)
4. [Notebook Structure](#notebook-structure)
5. [Core Concepts Covered](#core-concepts-covered)
6. [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
7. [Task-Dependent Preprocessing](#task-dependent-preprocessing)
8. [Key Comparisons](#key-comparisons)
9. [Hands-On Implementation](#hands-on-implementation)
10. [Analysis and Validation](#analysis-and-validation)
11. [Project Structure](#project-structure)
12. [Requirements](#requirements)
13. [How to Run](#how-to-run)
14. [Important Design Decisions](#important-design-decisions)
15. [What I Learned Today](#what-i-learned-today)
16. [Day 2 Preview](#day-2-preview)

---

## Overview

Natural Language Processing models cannot directly work with raw human language in the same way people understand it.

Raw text may contain:

- inconsistent capitalization,
- punctuation,
- HTML tags,
- contractions,
- numbers,
- common low-signal words,
- multiple grammatical forms of the same word.

Before text can be represented numerically using techniques such as **TF-IDF** or **word embeddings**, it should first be cleaned and standardized.

The main goal of Day 1 is therefore:

> Transform raw text into clean, consistent, task-aware text while preserving the information that matters to the machine learning task.

---

## Learning Objectives

By the end of Day 1, the notebook demonstrates how to:

- understand the role of NLP preprocessing,
- inspect raw text before transformation,
- tokenize text into smaller units,
- distinguish word tokenization from sub-word tokenization,
- normalize text using lowercasing,
- remove punctuation and selected noise,
- understand when numbers should or should not be removed,
- identify and remove stop words,
- preserve important negation words for sentiment analysis,
- lemmatize words into meaningful base forms,
- compare lemmatization with stemming,
- distinguish useful signal from unnecessary noise,
- build a reusable full preprocessing function,
- apply preprocessing to real IMDb movie reviews,
- compare raw text with cleaned text,
- measure changes in vocabulary size,
- verify that task-critical meaning is preserved.

---

## Dataset

### IMDb Dataset of 50K Movie Reviews

The notebook uses the **IMDb Dataset of 50K Movie Reviews**, a binary sentiment-analysis dataset containing positive and negative movie reviews.

It is especially useful for learning text preprocessing because the reviews contain real-world text characteristics such as:

- upper/lowercase variation,
- punctuation,
- HTML formatting,
- contractions,
- negation,
- ratings,
- different grammatical word forms,
- long natural-language sentences.

The dataset file used by the notebook is:

```text
IMDB Dataset.csv.zip
```

The notebook reads the ZIP file directly, so manual extraction is not required.

---

## Notebook Structure

The notebook is organized into the following sections:

| Section | Topic |
|---|---|
| 0 | Setup — Imports, Reproducibility & NLTK Resources |
| 1 | Day 1 Roadmap & Requirement Coverage |
| 2 | Dataset — IMDb 50K Movie Reviews |
| 3 | NLP & Raw Text |
| 4 | Why Text Needs Preprocessing |
| 5 | Tokenization — Word & Sub-word |
| 6 | Cleaning & Normalization |
| 7 | Stop-word Removal |
| 8 | Lemmatization |
| 9 | Lemmatization vs. Stemming |
| 10 | Task-Dependent Preprocessing |
| 11 | Full Text-Cleaning Pipeline |
| 12 | Apply the Pipeline to IMDb Reviews |
| 13 | Before vs. After Analysis |
| 14 | Signal-Preservation Checks |
| 15 | Hands-On Lab Checklist |
| 16 | What I Learned Today |
| 17 | Day 2 Preview |

---

## Core Concepts Covered

### NLP and Raw Text

**Natural Language Processing (NLP)** is the area of AI concerned with processing and understanding human language.

**Raw text** is text exactly as it appears before any cleaning or transformation.

Example:

```text
WOW!!! This Movie was REALLY good!!! 10/10
```

The goal of preprocessing is not to remove as much information as possible. The goal is to reduce unnecessary variation while preserving useful meaning.

---

### Tokenization

Tokenization splits text into smaller units called **tokens**.

Example:

```text
"The movie was great!"
```

becomes:

```text
["The", "movie", "was", "great", "!"]
```

The notebook also explains the conceptual difference between:

- word tokenization,
- sub-word tokenization.

Sub-word tokenization is commonly used by modern transformer models to handle unfamiliar words by splitting them into smaller known pieces.

---

### Cleaning and Normalization

Cleaning and normalization make text more consistent.

#### Lowercasing

```text
Good
GOOD
good
```

becomes:

```text
good
```

#### Punctuation Removal

```text
great!
```

becomes:

```text
great
```

#### Numbers

Numbers are not automatically considered noise.

For example:

```text
This movie is 10/10
```

The value `10/10` may contain important sentiment information, so number removal should depend on the task.

---

### Stop-word Removal

Stop words are common words that may carry little useful signal for a specific task.

Examples:

```text
the
is
a
```

However, stop-word removal must be task-aware.

For sentiment analysis:

```text
good
```

and:

```text
not good
```

have very different meanings.

Removing `not` can destroy the sentiment signal.

---

### Lemmatization

Lemmatization reduces different word forms into meaningful base forms.

Examples:

```text
running -> run
ran     -> run
cars    -> car
```

The notebook uses **part-of-speech-aware lemmatization** to improve the quality of the resulting base words.

---

## Text Preprocessing Pipeline

The final preprocessing flow used in the notebook is:

```text
Raw IMDb Review
      |
      v
HTML Cleanup
      |
      v
Lowercasing
      |
      v
Negation Normalization
      |
      v
Tokenization
      |
      v
Punctuation / Number Filtering
      |
      v
Task-Aware Stop-word Removal
      |
      v
POS-Aware Lemmatization
      |
      v
Clean Tokens
      |
      v
Clean Text
```

The final reusable preprocessing function combines all Day 1 concepts into one consistent workflow.

---

## Task-Dependent Preprocessing

A major concept in this notebook is that preprocessing should never be applied blindly.

A piece of information may be noise in one NLP task and useful signal in another.

Examples:

| Information | Possible Decision | Why |
|---|---|---|
| `not` | Preserve | Important for sentiment polarity |
| `10/10` | Possibly preserve | Can represent rating information |
| `!!!` | Usually remove | Often punctuation noise |
| Capitalization | Usually normalize | Reduces unnecessary variation |
| HTML tags | Remove | Formatting noise in IMDb reviews |

The most important preprocessing question is:

> Does this step remove noise, or does it remove useful signal?

---

## Key Comparisons

### Tokenization vs. Cleaning

| Tokenization | Cleaning |
|---|---|
| Splits text into smaller units | Removes or normalizes selected content |
| Does not automatically delete punctuation | May remove punctuation |
| Does not convert text into numbers | Still produces text |

---

### Cleaning vs. Normalization

| Cleaning | Normalization |
|---|---|
| Removes selected unwanted content | Makes text more consistent |
| Example: remove punctuation | Example: convert text to lowercase |

---

### Blind vs. Task-Aware Stop-word Removal

Input:

```text
this movie is not good
```

Blind stop-word removal may produce:

```text
movie good
```

Task-aware removal preserves negation:

```text
movie not good
```

---

### Lemmatization vs. Stemming

| Stemming | Lemmatization |
|---|---|
| Mechanically cuts word endings | Produces meaningful base forms |
| May create non-words | Usually produces real words |
| More aggressive | More linguistically informed |
| Weaker meaning preservation | Better meaning preservation |

For this notebook, **lemmatization** is used in the final pipeline.

---

## Hands-On Implementation

The notebook includes practical code for:

- loading the zipped IMDb dataset,
- inspecting dataset shape and columns,
- checking missing values,
- identifying duplicate reviews,
- visualizing sentiment class distribution,
- tokenizing sample text,
- applying lowercasing,
- removing punctuation,
- inspecting stop-word lists,
- preserving sentiment negation,
- comparing stemming and lemmatization,
- building a reusable preprocessing function,
- preprocessing real IMDb reviews,
- creating cleaned review columns,
- comparing raw and cleaned text.

---

## Analysis and Validation

A professional preprocessing workflow should not stop after the code runs.

The notebook validates preprocessing using:

### Review Length Comparison

The number of words before and after preprocessing is compared.

### Vocabulary Comparison

The notebook measures how preprocessing affects the number of unique tokens.

### Most Common Clean Words

The most frequent words after preprocessing are inspected.

### Raw vs. Clean Review Inspection

Real reviews are displayed before and after preprocessing to verify that the output still preserves useful meaning.

### Negation Preservation Check

Reviews containing expressions such as:

```text
not
didn't
wasn't
isn't
```

are inspected to confirm that important sentiment negation survives preprocessing.

---

## Project Structure

Recommended folder structure:

```text
Week 8/
└── Day 1/
    ├── Day1.ipynb
    ├── IMDB Dataset.csv.zip
    └── README.md
```

The notebook is designed to find the dataset ZIP file automatically when it is stored beside the notebook.

---

## Requirements

Main Python libraries used:

```text
pandas
numpy
matplotlib
nltk
```

NLTK resources used:

```text
punkt
punkt_tab
stopwords
wordnet
omw-1.4
averaged_perceptron_tagger
averaged_perceptron_tagger_eng
```

---

## How to Run

### 1. Clone or open the project folder

Make sure the notebook and dataset are stored together:

```text
Day1.ipynb
IMDB Dataset.csv.zip
```

### 2. Open the notebook

You can run it using:

- Jupyter Notebook,
- JupyterLab,
- VS Code,
- Google Colab.

### 3. Run the notebook from top to bottom

The notebook automatically:

1. imports the required libraries,
2. downloads the required NLTK resources,
3. finds the IMDb ZIP file,
4. loads the dataset,
5. applies the preprocessing workflow.

### 4. Working Sample

By default, the notebook can use a smaller reproducible sample for faster experimentation.

Example:

```python
PROCESS_FULL_DATASET = False
SAMPLE_SIZE = 5000
```

To preprocess the full dataset:

```python
PROCESS_FULL_DATASET = True
```

---

## Important Design Decisions

The final preprocessing pipeline uses the following decisions:

| Preprocessing Step | Decision | Reason |
|---|---|---|
| Lowercasing | Applied | Reduces case variation |
| HTML Removal | Applied | HTML is formatting noise |
| Punctuation Removal | Applied | Simplifies the classical text pipeline |
| Number Removal | Default | Can be changed depending on the task |
| Stop-word Removal | Selective | Reduces low-signal words |
| Negation Preservation | Applied | Critical for sentiment analysis |
| Lemmatization | Applied | Reduces word-form variation |
| Stemming | Not used in final pipeline | More aggressive and less meaning-aware |

---

## What I Learned Today

By completing this notebook, I learned:

- what NLP and raw text are,
- why raw text requires preprocessing,
- how tokenization works,
- the difference between word and sub-word tokenization,
- how cleaning and normalization improve text consistency,
- why lowercasing is useful,
- when punctuation and numbers should be removed,
- what stop words are,
- why negation must be preserved in sentiment analysis,
- how lemmatization works,
- the difference between lemmatization and stemming,
- how to distinguish noise from useful signal,
- why preprocessing must depend on the NLP task,
- how to combine all preprocessing steps into one reusable pipeline,
- how to validate that preprocessing did not destroy important meaning.

---

## Day 2 Preview

After Day 1, the text is cleaner and more consistent, but it is still text.

The next step is to convert it into numerical representations that a machine learning model can use.

Day 2 will focus on:

- Bag of Words,
- TF-IDF,
- Word Embeddings,
- Word2Vec,
- GloVe,
- frequency-based vs. semantic text representation.

The transition is:

```text
Day 1
Raw Text
   |
   v
Clean Text

Day 2
Clean Text
   |
   v
Numerical Representation
   |
   v
Machine Learning Model
```

---

## Summary

Day 1 builds the foundation for the rest of the NLP pipeline.

The central lesson is:

> Text preprocessing is not simply about deleting words or symbols. A good preprocessing pipeline removes unnecessary variation while protecting the information that matters to the machine learning task.

The cleaned text produced here is ready for **Day 2 — TF-IDF and Word Embeddings**.
