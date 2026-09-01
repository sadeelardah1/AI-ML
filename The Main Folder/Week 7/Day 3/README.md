# Day 3 — RNNs & LSTMs for Sequential Data

Week 7, Day 3 of the deep learning track: a hands-on notebook comparing **order-unaware** vs.
**order-aware** neural architectures on a real sequential-text task — IMDb movie review sentiment
classification.

---

## 1. Goal

Build intuition for *why* recurrent architectures exist by directly comparing three models on the
exact same data, split, and training setup:

1. A **non-sequential baseline** (averages word embeddings — ignores order)
2. A **Simple RNN** (hidden-state memory, but prone to vanishing gradients)
3. An **LSTM** (gated memory, designed to preserve longer-range information)



## 2. Dataset

| | |
|---|---|
| **Task** | Binary sentiment classification (positive / negative) |
| **Source** | IMDb movie reviews (Keras built-in dataset, cached locally as `dataset/imdb.npz`) |
| **Vocabulary size** | 10,000 most frequent tokens |
| **Max sequence length** | 200 tokens (shorter reviews zero-padded) |
| **Split** | 20,000 train / 5,000 validation / 25,000 test |
| **Class balance** | 50% positive / 50% negative (train) |

## 3. Notebook Structure

0. Setup — imports, reproducibility (fixed seed), runtime
1. Day 3 roadmap & project context
2. Why order matters in sequential data (toy example)
3. RNNs — hidden state as memory
4. The vanishing-gradient problem (illustrative decay plot)
5. LSTMs & GRUs — gated memory
6. Embeddings for representing text
7. Hands-on dataset — IMDb sentiment (loading, train/val split, padding, decoding, shared training helpers)
8. **Model A** — Non-sequential baseline (`Embedding` → `GlobalAveragePooling1D` → `Dense`)
9. **Model B** — Simple RNN (`Embedding` → `SimpleRNN(64)` → `Dense`)
10. **Model C** — LSTM (`Embedding` → `LSTM(64)` → `Dense`)
11. Optional GRU experiment (`RUN_GRU` flag, off by default)
12. Model comparison & test-set evaluation (validation accuracy, training time, confusion matrix)
13. Order-awareness experiment (same tokens, original vs. reversed order, per-model prediction shift)
14. What I learned today (reflection)

## 4. Setup Used

```
VOCAB_SIZE = 10,000
MAX_LEN    = 200
EMBED_DIM  = 64
BATCH_SIZE = 128
EPOCHS     = 6
Optimizer  = Adam (lr = 1e-3)
Loss       = binary_crossentropy
Metrics    = accuracy, AUC
```

All three models share the same embedding size, batch size, sequence length, loss function, and
train/validation split so the comparison in Section 12 is fair.

## 5. Run Status

This copy of the notebook was executed **up through the Simple RNN training run (Section 9)**.
Early results captured so far:

| Model | Epoch 1 val. accuracy | Notes |
|---|---|---|
| Non-Sequential Baseline | 0.8386 (→ improving each epoch) | Fast (~1–2s/epoch on this run) |
| Simple RNN | 0.5626 (→ improving each epoch) | Much slower (~18–20s/epoch); early epochs near chance level, consistent with the harder optimization RNNs face |

The **LSTM run, the optional GRU run, the full model-comparison table, the confusion matrix, and
the order-awareness experiment (Sections 10–13) have not been executed yet** — running the
remaining cells top to bottom will complete the comparison and populate the "Required comparison
statement" and "What I learned today" reflection cells.

## 6. How to Run

1. Make sure `dataset/imdb.npz` exists next to the notebook (auto-downloaded via Keras on first run if missing).
2. Run all cells top to bottom. `RUN_GRU` is `False` by default — set it to `True` to include the GRU model in Section 11.
3. Section 12 selects the best model **by validation accuracy**, then reports test accuracy/AUC and a confusion matrix for that model only.
4. Fill in the reflection prompts in Sections 12 and 14 after reviewing the results.

**Dependencies:** `numpy`, `pandas`, `matplotlib`, `tensorflow`/`keras`, `scikit-learn`.
