# Week 7: Deep Learning Architectures and Sprint 2

## Program Context

Week 7 is Sprint 2 of the Phase 3 capstone for the BinX Tech AI and Machine Learning Internship
Program. The week is built around two parallel tracks that share the same underlying goal:
matching a neural network architecture to the type of data being modeled, then advancing the
capstone project's core model through a full sprint cycle of planning, experimentation, and
review.

- **Days 1, 2, and 5** follow the capstone project itself: a skin lesion image classifier
  (Benign vs. Malignant), built with a Convolutional Neural Network and MobileNetV2 transfer
  learning.
- **Days 3 and 4** are the program's required reference material on sequence and language
  architectures (RNNs, LSTMs, Attention, and Transformers), applied to a second, independent
  task: IMDb movie review sentiment classification.

## Table of Contents

| Day | Title | Summary | Notebook | README |
|-----|-------|---------|----------|--------|
| Day 1 | Convolutional Neural Networks | Sprint 2 planning; why CNNs suit image data; convolution demonstrated with a real edge-detection filter. | [Day1.ipynb](Day%201/Day1.ipynb) | [README](Day%201/README.md) |
| Day 2 | Building CNNs and Transfer Learning | A full CNN built from scratch, data augmentation, and MobileNetV2 transfer learning, compared on the same dataset and split. | [Day2.ipynb](Day%202/Day2.ipynb) | [README](Day%202/README.md) |
| Day 3 | RNNs and LSTMs for Sequential Data | Non-sequential baseline vs. Simple RNN vs. LSTM, compared on IMDb sentiment classification. | [Day3.ipynb](Day%203/Day3.ipynb) | [README](Day%203/README.md) |
| Day 4 | Attention and Transformers | Pre-trained DistilBERT applied to the same IMDb task, with attention-weight inspection and a comparison to Day 3. | [Day4.ipynb](Day%204/Day4.ipynb) | [README](Day%204/README.md) |
| Day 5 | Advancing the Core Model and Sprint 2 Review | Controlled fine-tuning experiments on the Day 2 CNN, validation-based model selection, final test evaluation, Sprint Review and Retrospective. | [Day5.ipynb](Day%205/Day5.ipynb) | [README](Day%205/README.md) |

Each README link above opens directly to that day's full, standalone documentation, including its
own objectives, dataset details, setup instructions, results, and notes.

## Day-by-Day Summary

### Day 1: Convolutional Neural Networks
Opens Sprint 2 with planning, then introduces convolution conceptually: why a fully connected
network is impractical for image data, and how a small filter, applied through convolution,
detects a pattern such as an edge anywhere in an image. Demonstrated on a real image from the
project's own Benign/Malignant dataset.

### Day 2: Building CNNs and Transfer Learning
Builds a complete CNN using convolution, pooling, and dense layers; applies data augmentation to
reduce overfitting; and introduces transfer learning using MobileNetV2 as a pre-trained feature
extractor, first with a frozen backbone and then with partial fine-tuning. Scratch training,
augmentation, and transfer learning are compared on validation and test metrics, with the
fine-tuned MobileNetV2 model becoming the project's Day 2 reference result (Accuracy 0.8625,
AUC 0.9499).

### Day 3: RNNs and LSTMs for Sequential Data
Steps outside the image project to build intuition for recurrent architectures, comparing a
non-sequential baseline, a Simple RNN, and an LSTM on IMDb movie review sentiment classification,
including a direct illustration of the vanishing-gradient problem and an order-awareness
experiment. This notebook's later sections (the LSTM run, model comparison table, and
order-awareness experiment) were not executed in the delivered copy; see the Day 3 README for the
exact run status.

### Day 4: Attention and Transformers
Continues the Day 3 IMDb task with a pre-trained DistilBERT model from Hugging Face, explaining
self-attention, query/key/value, and positional encoding, then applying DistilBERT to the same
sentiment task, testing its sensitivity to negation, visualizing real attention weights, and
comparing its results against the Day 3 recurrent models.

### Day 5: Advancing the Core Model and Sprint 2 Review
Returns to the capstone project. Runs three controlled fine-tuning experiments on the Day 2 CNN,
selects the best configuration using validation AUC only, evaluates that configuration once on the
test set, and documents the result honestly against the Day 2 reference (Accuracy 0.8375,
AUC 0.9513, a marginal AUC gain alongside a drop in accuracy and recall). Closes with a full
Sprint Review and Retrospective, including one concrete change carried into Sprint 3.

## Repository Structure

```
Week 7/
    README.md
    Day 1/
        Day1.ipynb
        README.md
    Day 2/
        Day2.ipynb
        README.md
    Day 3/
        Day3.ipynb
        README.md
    Day 4/
        Day 4.pdf
        Day4.ipynb
        README.md
    Day 5/
        Day5.ipynb
        README.md
        artifacts/
            day5_experiment_log.csv
            models/
                a_-_day2_control.keras
                b_-_lower_lr.keras
                c_-_deeper_fine-tuning.keras
```

## How to Navigate This Week

1. Start with this file for the overall picture and the day-by-day summary above.
2. Use the Table of Contents to open any day's notebook or README directly.
3. Each day's own README is self-contained: it lists that day's specific objectives, dataset
   requirements, setup steps, and results, so it can be read and run independently once the
   correct dataset is in place.
4. Days 1, 2, and 5 depend on the same image dataset, first introduced in Day 1 and reused without
   duplication in later days; consult the Day 1 README for the dataset layout.
5. Days 3 and 4 depend on the IMDb dataset introduced in Day 3 and reused in Day 4; consult the
   Day 3 README for its dataset layout.

## Notes on Completeness

This week's notebooks are the intern's own delivered work and are documented here exactly as
provided. Day 3, in particular, was executed only through its Simple RNN training run at the time
of this summary; the remaining comparison sections are present in the notebook but have not yet
produced output. See each day's individual README for the precise, authoritative run status and
results.



