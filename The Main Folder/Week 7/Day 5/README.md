# Week 7, Day 5: Advancing the Core Model and Sprint 2 Review

## Project Overview

This notebook is the final deliverable of Sprint 2 for the Phase 3 capstone project: a skin
lesion image classifier that predicts whether a lesion is Benign or Malignant. Day 5 does not
introduce a new architecture. Instead, it takes the CNN with MobileNetV2 transfer learning
developed and evaluated in Day 2, tunes it through a small set of controlled experiments, selects
the best configuration using validation data, evaluates that configuration once on the test set,
and closes Sprint 2 with a documented Review and Retrospective.

## Day 5 Objectives

1. Confirm that the image data type still calls for a CNN with transfer learning, rather than
   switching architecture.
2. Design and run a small number of controlled fine-tuning experiments, changing one variable at
   a time relative to the Day 2 configuration.
3. Select the best experiment using validation performance only, then evaluate that single
   selection on the test set exactly once.
4. Compare the final Day 5 result against the Day 2 reference model, honestly, including the
   possibility of no improvement.
5. Complete a full Sprint Review and Retrospective for Sprint 2.

## What the Notebook Covers

The notebook `Day5.ipynb` is organized into the following sections:

0. Setup: imports, reproducibility, and runtime information (TensorFlow version, GPU
   availability).
1. Sprint 2 goal and the Day 2 evidence used as the starting reference.
2. Confirmation of the core architecture decision (CNN with transfer learning) for this project's
   data type.
3. Loading the image dataset produced and documented in Day 2.
4. Building reproducible TensorFlow datasets, using the same image size, batch size, validation
   split, and random seed as Day 2.
5. The Day 5 experiment plan: three controlled fine-tuning configurations.
6. The MobileNetV2 experiment framework, including the Day 2 data augmentation pipeline and the
   transfer-learning model definition.
7. Running the controlled fine-tuning experiments.
8. The experiment log and validation-based model selection.
9. Final, one-time evaluation of the selected model on the test set.
10. Threshold analysis: how the classification decision threshold affects precision and recall.
11. A comparison table between the Day 2 reference model and the Day 5 result.
12. Sprint Review, documenting the final decision and metrics.
13. Retrospective, reflecting on the sprint and defining one concrete change for Sprint 3.
14. A closing summary of what was learned during Day 5.

Each section states clearly what is being done and why, and the notebook is written so that model
selection decisions and their justifications are visible directly in the Markdown narrative, not
only in the code.

## Experiment Design

Three fine-tuning configurations were tested, changing exactly one variable at a time relative to
a fixed control configuration:

| Experiment | Fine-Tune Layers | Fine-Tune Learning Rate | Dropout | Purpose |
|------------|------------------|--------------------------|---------|---------|
| A, Day 2 Control | 30 | 1e-5 | 0.30 | Reproduce the Day 2 fine-tuning design |
| B, Lower Learning Rate | 30 | 5e-6 | 0.30 | Test gentler fine-tuning |
| C, Deeper Fine-Tuning | 40 | 5e-6 | 0.30 | Test whether adapting more upper layers helps |

All three experiments used the same image size (160 by 160), batch size (32), validation split
(20 percent), preprocessing, and random seed. Only the variable stated in the table above was
changed between experiments.

The experiment with the highest validation AUC was selected as the winner. The test set was not
used at any point during selection.

## Model Selection and Final Results

Experiment A, Day 2 Control, achieved the highest validation AUC among the three experiments and
was selected for final evaluation.

Final test-set results for the selected model:

| Metric | Value |
|--------|-------|
| Accuracy | 0.8375 |
| AUC | 0.9513 |
| Precision | 0.9400 |
| Recall | 0.7210 |

Comparison against the Day 2 fine-tuned MobileNetV2 reference (Accuracy 0.8625, AUC 0.9499):

- AUC improved by 0.0014.
- Accuracy decreased by 0.0250.
- Recall decreased from 0.7950 to 0.7210.

The conclusion documented in the Sprint Review is stated honestly: the Day 5 tuning did not
produce a clear overall improvement over Day 2. Ranking performance, measured by AUC, improved
marginally, while classification accuracy and recall on the Malignant class both decreased. This
result is recorded as valid evidence rather than treated as a failure, consistent with the
notebook's stated principle that a reproducible negative result is still useful evidence.

## Sprint Review Summary

- Project data type: Images.
- Selected architecture: CNN with MobileNetV2 transfer learning.
- Day 2 reference model: Accuracy 0.8625, AUC 0.9499.
- Day 5 experiments conducted: control fine-tuning, lower learning rate, and deeper fine-tuning.
- Selected Day 5 experiment: A, Day 2 Control.
- Final Day 5 metrics: Accuracy 0.8375, AUC 0.9513, Precision 0.9400, Recall 0.7210.
- Result relative to Day 2: AUC improved slightly; accuracy and recall decreased; no clear overall
  improvement was achieved.

## Retrospective Summary

- What went well: transfer learning provided a strong starting point, and the experiments were
  controlled and properly logged.
- What was difficult: fine-tuning multiple CNN configurations is computationally expensive,
  particularly without GPU acceleration.
- What the metrics revealed: the selected Day 5 model achieved a slightly higher AUC than the
  Day 2 model, but its accuracy and recall were lower, showing an improvement in ranking
  performance without a consistent improvement across all evaluation metrics.
- Concrete change for Sprint 3: use the currently selected model as the baseline going forward,
  and focus the next sprint on error analysis and stronger evaluation rather than introducing
  additional architectures.

## Dataset Information

This notebook reuses the image dataset already prepared and documented in Day 2. It is not
duplicated inside the Day 5 folder. The expected project layout is:

```
Week 7/
    Day 2/
        dataset/
            train/
                Benign/
                Malignant/
            test/
                Benign/
                Malignant/
        Day2.ipynb
    Day 5/
        Day5.ipynb
```

Documented dataset counts, carried forward from Day 2: 6,289 Benign and 5,590 Malignant training
images; 1,000 Benign and 1,000 Malignant test images.

## Project Structure

```
Day5/
    Day5.ipynb
    README.md
    artifacts/
        models/
        experiment_log.csv
```

The `artifacts` folder is created by the notebook itself to store trained model files and the
experiment log, so that training does not need to be repeated unnecessarily.

## Requirements

- Python 3.9 or later
- Jupyter Notebook, JupyterLab, VS Code with the Jupyter extension, or Google Colab
- A GPU runtime is recommended but not required; training on CPU is possible but significantly
  slower

## Required Python Libraries

- numpy
- pandas
- matplotlib
- tensorflow
- scikit-learn

Install them with:

```
pip install numpy pandas matplotlib tensorflow scikit-learn
```

## Installation Instructions

1. Ensure the Day 2 project folder, including its `dataset` subfolder, is present alongside this
   Day 5 folder, matching the layout shown in the Dataset Information section.
2. Install the required libraries listed above.

## How to Run the Notebook

1. Open `Day5.ipynb` in Jupyter Notebook, JupyterLab, VS Code, or Google Colab.
2. Run the cells in order, from top to bottom.
3. Section 7 trains real MobileNetV2 models for each experiment and can take a significant amount
   of time on CPU. Once run, the resulting files in `artifacts/models` allow later sections to be
   re-run without repeating training.
4. Model selection in Section 8 is based on validation AUC only. The test set is evaluated once,
   in Section 9, after selection is complete.

## Dataset Path Instructions

The dataset is loaded directly from the Day 2 project folder rather than being duplicated. The
notebook locates the dataset relative to its own location, following the folder layout described
above. If the folder layout differs, adjust the dataset path defined in Section 3 of the notebook
to point at the folder that directly contains the `train` and `test` subfolders.

## Main Concepts Learned

- Confirming an architecture decision rather than repeating architecture selection from scratch.
- Designing controlled experiments that change one variable at a time.
- The distinction between validation-based model selection and a single, final test-set
  evaluation, and why the test set must not be used to choose between experiments.
- Logging experiment configurations and results systematically for later comparison.
- Threshold analysis and its effect on the precision and recall trade-off.
- Documenting a result honestly when tuning does not produce a clear improvement, and treating a
  reproducible negative result as valid evidence.
- Completing a full Sprint Review and Retrospective cycle, including a concrete, actionable change
  for the next sprint.

## Expected Outputs

Running the notebook from top to bottom produces:

- A dataset summary consistent with the documented Day 2 counts.
- An experiment log recording the configuration and validation results of all three fine-tuning
  experiments, saved to `artifacts/experiment_log.csv`.
- A plot comparing validation performance across the three experiments.
- Final test-set metrics, a classification report, and a confusion matrix for the selected model.
- A threshold analysis plot showing precision and recall at different decision thresholds.
- A comparison table between the Day 2 reference model and the final Day 5 result.
- A completed Sprint Review and Retrospective, documented directly in the notebook.


