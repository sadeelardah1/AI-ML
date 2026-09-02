# Week 7 — Day 4: Attention & Transformers

## Overview

This notebook continues the IMDb sentiment-analysis work from Day 3 and introduces **Attention**, **Self-Attention**, and **Transformer-based NLP**.

Instead of building another recurrent model, Day 4 uses a **pre-trained DistilBERT model from Hugging Face** and applies it to the same IMDb sentiment-classification task used in Day 3.

The main goal is to connect the theory of Transformers with a practical implementation and compare the Transformer approach with the Day 3 models.

---

## Learning Objectives

By the end of this notebook, I was able to:

- Explain why RNNs and LSTMs can be limited by step-by-step sequence processing.
- Explain the main idea of Attention and Self-Attention.
- Understand the intuition behind Query, Key, and Value.
- Explain why Transformers need positional information.
- Distinguish between BERT, DistilBERT, and GPT-2.
- Load and use a pre-trained Transformer from Hugging Face.
- Apply DistilBERT to real IMDb reviews.
- Evaluate the model using classification metrics.
- Test how the model reacts to contextual changes such as negation.
- Inspect and visualize real self-attention weights.
- Compare the Transformer with the models implemented in Day 3.

---

## Project Structure

```text
Week 7/
├── Day 3/
│   ├── dataset/
│   │   └── imdb.npz
│   ├── Day3.ipynb
│   └── README.md
│
└── Day 4/
    ├── Day 4.pdf
    ├── Day4.ipynb
    └── README.md
```

Day 4 reuses the IMDb dataset saved during Day 3:

```text
../Day 3/dataset/imdb.npz
```

No duplicate IMDb dataset is created inside the Day 4 folder.

---

## Libraries Used

- Python
- NumPy
- Pandas
- Matplotlib
- TensorFlow / Keras
- PyTorch
- Hugging Face Transformers
- Scikit-learn

Install the main Day 4 dependencies with:

```bash
pip install transformers torch scikit-learn
```

---

## Pre-trained Model

The notebook uses:

```text
distilbert-base-uncased-finetuned-sst-2-english
```

This is a **DistilBERT** checkpoint that is already fine-tuned for English sentiment classification.

It is loaded through the Hugging Face `pipeline` API:

```python
from transformers import pipeline

sentiment_model = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    device=DEVICE,
)
```

The pipeline handles the main inference steps:

```text
Text
↓
Tokenizer
↓
Token IDs
↓
DistilBERT
↓
Self-Attention
↓
Classification Head
↓
POSITIVE / NEGATIVE
```

---

## Practical Work

### 1. Hugging Face Inference

The first experiment verifies that the pre-trained model can classify simple positive, negative, and mixed-context sentences.

Example results from the notebook include high-confidence predictions for both positive and negative examples.

### 2. IMDb Continuation from Day 3

The same IMDb test set from Day 3 is loaded:

- Training reviews: **25,000**
- Test reviews: **25,000**
- Negative test reviews: **12,500**
- Positive test reviews: **12,500**

Because Day 3 stores IMDb reviews as integer sequences, the notebook reconstructs readable review text before passing it to DistilBERT.

### 3. Real IMDb Predictions

DistilBERT is tested on real IMDb reviews and returns:

- True label
- Predicted label
- Confidence score
- Review preview

### 4. Context Stress Test

A controlled experiment checks whether the model reacts to changes in context, especially negation.

Example:

```text
The movie was good.
↓
POSITIVE

The movie was not good.
↓
NEGATIVE
```

This experiment is used to show that the model is responding to contextual information rather than only detecting isolated positive or negative words.

### 5. Transformer Evaluation

For practical CPU execution, the notebook evaluates DistilBERT on a balanced subset of **1,000 IMDb test reviews**:

- 500 negative
- 500 positive

Recorded result:

| Metric | Result |
|---|---:|
| Accuracy | 0.8850 |
| Precision | 0.9365 |
| Recall | 0.8260 |
| F1 Score | 0.8778 |
| AUC | 0.9593 |
| Inference Time | 456.67 s |

The notebook also includes a classification report and confusion matrix.

> Important: Day 3 models were evaluated on the full 25,000-review test set, while the default Day 4 Transformer evaluation uses 1,000 samples. Therefore, the comparison should not be treated as a perfectly equal benchmark unless DistilBERT is also evaluated on all 25,000 test reviews.

---

## Self-Attention Visualization

To go beyond basic `pipeline()` usage, the notebook loads the same DistilBERT checkpoint using:

```python
AutoTokenizer
AutoModelForSequenceClassification
```

The model is configured with:

```python
attn_implementation="eager"
```

This allows the notebook to request and inspect attention weights.

For the sample sentence:

```text
The movie was not good, but the ending was surprisingly excellent.
```

the notebook extracts:

- **6 attention layers**
- A **15 × 15** final attention matrix

The attention heads in the last layer are averaged and visualized as a token-to-token heatmap.

This visualization connects the theoretical concept of Self-Attention with an actual internal output from the Transformer.

---

## Day 3 vs Day 4

The notebook compares the Transformer experiment with the recorded Day 3 results:

| Model | Accuracy | AUC | Evaluation Samples |
|---|---:|---:|---:|
| Non-Sequential Baseline | 0.86172 | 0.938281 | 25,000 |
| LSTM | 0.84380 | 0.923725 | 25,000 |
| Simple RNN | 0.71920 | 0.786096 | 25,000 |
| DistilBERT | 0.88500 | 0.959312 | 1,000 |

The comparison is used primarily to understand the difference between **recurrent sequence modeling** and **pre-trained Transformer-based modeling**.

---

## RNN/LSTM vs Transformer

| Feature | RNN / LSTM | Transformer |
|---|---|---|
| Processing | Step by step | Much more parallel processing across positions |
| Main mechanism | Recurrent hidden state | Self-Attention |
| Long-range relationships | Passed through recurrent updates | Direct interaction between positions |
| Order information | Naturally follows sequence order | Added explicitly through positional information |
| Parallelism | Limited | Much stronger |

The key idea is that RNNs and LSTMs pass information through recurrent memory step by step, while Transformers use Self-Attention so each position can directly weigh other relevant positions in the sequence.

---

## Implementation Decisions

- Reused the same IMDb task from Day 3 to keep the comparison meaningful.
- Selected DistilBERT because the task is text classification and it is lighter than full BERT.
- Used a ready sentiment checkpoint instead of training a Transformer from scratch.
- Added a negation stress test to demonstrate contextual behavior.
- Added an attention heatmap to connect theory with actual model internals.
- Used a balanced 1,000-review subset by default to keep execution practical on CPU.

---

## Limitations

- The DistilBERT checkpoint was already fine-tuned for sentiment analysis, so the notebook mainly performs inference rather than full fine-tuning.
- IMDb text is reconstructed from the integer sequences saved during Day 3, so it is an approximate reconstruction of the original raw text.
- The default Day 4 evaluation uses fewer samples than the Day 3 evaluation.
- Attention weights are useful for inspection but should not be treated as a complete explanation of model reasoning.
- Sentiment models can still struggle with sarcasm, ambiguity, mixed sentiment, or domain-specific language.

---

## What I Learned

During Day 4, I learned how Transformers solve important limitations of recurrent sequence models by using Self-Attention. I also learned how to reuse a pre-trained Transformer through Hugging Face, apply it to a real sentiment-classification task, evaluate its performance, test its sensitivity to context, and inspect its internal attention weights.

The most important practical takeaway was understanding the complete path from raw text to a Transformer prediction:

```text
Text
↓
Tokenizer
↓
Transformer
↓
Self-Attention
↓
Contextual Representation
↓
Classification
```
