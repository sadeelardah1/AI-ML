# ============================================================
# Week 4 - Day 3
# Bias-Variance & Diagnosing Model Fit
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score





# 1. Load the dataset
BASE_DIR = Path(__file__).resolve().parent
dataset_path = BASE_DIR / "train_and_test2.csv"
titanic = pd.read_csv(dataset_path)

# Rename the target column for clarity
titanic = titanic.rename(
    columns={"2urvived": "Survived"}
)

# Use the same features from the previous days
features = [
    "Age",
    "Fare",
    "Sex",
    "sibsp",
    "Parch",
    "Pclass"
]

X = titanic[features]
y = titanic["Survived"]
print("Dataset shape:", titanic.shape)
print("Feature shape:", X.shape)
print("Target shape:", y.shape)


# 2. Create train, validation, and test sets
# First split:
# 80% temporary training data
# 20% untouched test data
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Second split:
# 75% of temp -> training
# 25% of temp -> validation
#
# Final result:
# 60% train
# 20% validation
# 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42
)
print("\nDataset split:")
print("Training rows  :", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows      :", len(X_test))


# 3. Helper function for model evaluation
def evaluate_model(name, model):
    """
    Train a model and calculate F1-score on both
    training and validation data.
    """

    # Train the model
    model.fit(X_train, y_train)

    # Predictions on training data
    train_predictions = model.predict(X_train)

    # Predictions on validation data
    val_predictions = model.predict(X_val)

    # Calculate F1-scores
    train_f1 = f1_score(
        y_train,
        train_predictions,
        zero_division=0)

    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0)

    # Difference between training and validation performance
    gap = train_f1 - val_f1
    print(f"\n{name}")
    print("-" * 45)
    print(f"Training F1   : {train_f1:.3f}")
    print(f"Validation F1 : {val_f1:.3f}")
    print(f"Train-Val Gap : {gap:.3f}")
    return train_f1, val_f1, gap


# 4. Step 1 - Deliberately create an overfit model
overfit_model = DecisionTreeClassifier(
    max_depth=None,
    random_state=42
)
overfit_train_f1, overfit_val_f1, overfit_gap = evaluate_model("OVERFIT MODEL", overfit_model)

# 5. Step 2 - Deliberately create an underfit model
underfit_model = DecisionTreeClassifier(
    max_depth=1,
    min_samples_leaf=300,
    random_state=42
)
underfit_train_f1, underfit_val_f1, underfit_gap = evaluate_model("UNDERFIT MODEL", underfit_model)

# 6. Step 3 - Reduce model complexity
fixed_model = DecisionTreeClassifier(max_depth=4, random_state=42)
fixed_train_f1, fixed_val_f1, fixed_gap = evaluate_model("REDUCED-COMPLEXITY MODEL",fixed_model)


# 7. Compare all three models
results = pd.DataFrame({
    "Model": [
        "Overfit",
        "Underfit",
        "Reduced Complexity"
    ],
    "Training F1": [
        overfit_train_f1,
        underfit_train_f1,
        fixed_train_f1
    ],
    "Validation F1": [
        overfit_val_f1,
        underfit_val_f1,
        fixed_val_f1
    ],
    "Train-Val Gap": [
        overfit_gap,
        underfit_gap,
        fixed_gap
    ]
})

print("\nModel Comparison")
print("=" * 60)
print(results.round(3).to_string(index=False))

# 8. Bias-Variance experiment using different tree depths
depths = range(1, 16)
train_scores = []
validation_scores = []
for depth in depths:

    model = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )
    model.fit(
        X_train,
        y_train
    )
    train_predictions = model.predict(
        X_train
    )
    val_predictions = model.predict(
        X_val
    )
    train_f1 = f1_score(
        y_train,
        train_predictions,
        zero_division=0
    )
    val_f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )
    train_scores.append(train_f1)
    validation_scores.append(val_f1)



# 9. Visualize model complexity
plt.figure(figsize=(10, 6))
plt.plot(
    depths,
    train_scores,
    marker="o",
    label="Training F1"
)

plt.plot(
    depths,
    validation_scores,
    marker="o",
    label="Validation F1"
)

plt.xlabel("Decision Tree max_depth")
plt.ylabel("F1-score")
plt.title("Bias-Variance Trade-off")
plt.xticks(list(depths))
plt.legend()
plt.grid(True)
plt.show()


# 10. Find the best validation depth

best_index = np.argmax(validation_scores)
best_depth = list(depths)[best_index]
best_validation_f1 = validation_scores[best_index]
print("\nBest depth in this experiment:")
print("max_depth =", best_depth)
print(f"Validation F1 = {best_validation_f1:.3f}")


# 11. Final diagnosis
print("\nFinal Diagnosis")
print("=" * 60)

print(
    "Overfit model:"
    "\n- Very high training performance"
    "\n- Much lower validation performance"
    "\n- Large train-validation gap"
)

print(
    "\nUnderfit model:"
    "\n- Poor training performance"
    "\n- Poor validation performance"
    "\n- Model is too simple"
)

print(
    "\nReduced-complexity model:"
    "\n- Training and validation scores are much closer"
    "\n- Smaller gap"
    "\n- Better balance between bias and variance"
)

print(
    "\nThe test set was not used during model diagnosis."
)