import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)

# 1. Create the dataset
data = {
    "monthly_usage": [
        5, 8, 10, 12, 15,
        18, 20, 22, 25, 28,
        30, 32, 35, 38, 40,
        42, 45, 48, 50, 55
    ],

    "support_calls": [
        8, 7, 6, 7, 5,
        5, 4, 5, 3, 4,
        2, 3, 2, 2, 1,
        2, 1, 1, 0, 1
    ],

    "satisfaction": [
        1, 2, 2, 2, 3,
        3, 3, 3, 4, 4,
        4, 4, 5, 4, 5,
        5, 5, 5, 5, 5
    ],

    "churn": [
        1, 1, 1, 1, 1,
        1, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)


print("=" * 70)
print("DATASET")
print("=" * 70)

print(df)

print("\nDataset shape:")
print(df.shape)


# 2. Inspect the target distribution
print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print(df["churn"].value_counts())

print("\nTarget percentages:")

print(
    df["churn"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# 3. Separate Features X and Target y
X = df[
    [
        "monthly_usage",
        "support_calls",
        "satisfaction"
    ]
]

y = df["churn"]
print("\n" + "=" * 70)
print("FEATURES X")
print("=" * 70)
print(X.head())
print("\n" + "=" * 70)
print("TARGET y")
print("=" * 70)
print(y.head())
print("\nX shape:", X.shape)
print("y shape:", y.shape)

# 4. Split the dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("\ny_train distribution:")
print(
    y_train.value_counts()
)
print("\ny_test distribution:")
print(
    y_test.value_counts()
)

# 5. Instantiate the Logistic Regression model
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# 6. Fit the model
model.fit(
    X_train,
    y_train
)
print("\nModel training completed successfully.")


# 7. Generate predicted classes
predictions = model.predict(
    X_test
)

print("\n" + "=" * 70)
print("PREDICTED CLASSES")
print("=" * 70)
print(predictions)

# 8. Generate class probabilities
probabilities = model.predict_proba(
    X_test
)

print("\n" + "=" * 70)
print("CLASS PROBABILITIES")
print("=" * 70)
print(probabilities)


# The order of probability columns
print("\nProbability column order:")
print(model.classes_)

# Usually:
# probabilities[:, 0] = Probability of Class 0
# probabilities[:, 1] = Probability of Class 1
# 9. Create a clear results table
results = X_test.copy()
results["Actual Class"] = y_test.values
results["Predicted Class"] = predictions
results["Stay Probability"] = probabilities[:, 0]
results["Churn Probability"] = probabilities[:, 1]
results["Correct Prediction"] = (
    results["Actual Class"]
    == results["Predicted Class"]
)


print("\n" + "=" * 70)
print("PREDICTION RESULTS")
print("=" * 70)
print(
    results.round(3)
)

# 10. Apply the 0.5 threshold manually
churn_probabilities = probabilities[:, 1]

manual_predictions = (
    churn_probabilities >= 0.50
).astype(int)


print("\n" + "=" * 70)
print("MANUAL THRESHOLD PREDICTIONS")
print("=" * 70)
print(manual_predictions)
print("\nPredictions from model.predict():")
print(predictions)
print(
    "\nAre manual predictions equal to model predictions?"
)
print(
    np.array_equal(
        manual_predictions,
        predictions
    )
)


# 11. Display coefficients and intercept
coefficients_table = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})


print("\n" + "=" * 70)
print("MODEL COEFFICIENTS")
print("=" * 70)
print(
    coefficients_table.round(4)
)
print("\nIntercept:")
print(
    round(model.intercept_[0], 4)
)


# 12. Predict one new customer
new_customer = pd.DataFrame({
    "monthly_usage": [9],
    "support_calls": [7],
    "satisfaction": [2]
})
new_prediction = model.predict(
    new_customer
)[0]
new_probabilities = model.predict_proba(
    new_customer
)[0]


print("\n" + "=" * 70)
print("NEW CUSTOMER")
print("=" * 70)
print(new_customer)
print("\nPredicted Class:")
print(new_prediction)
print("\nStay Probability:")
print(
    round(new_probabilities[0], 4)
)
print("\nChurn Probability:")
print(
    round(new_probabilities[1], 4)
)

if new_prediction == 1:
    print("\nFinal Prediction: Customer will churn.")
else:
    print("\nFinal Prediction: Customer will stay.")
    

# 13. Show the weighted sum manually
feature_values = new_customer.iloc[0].values
coefficients = model.coef_[0]
intercept = model.intercept_[0]
weighted_sum = (
    np.dot(
        feature_values,
        coefficients
    )
    + intercept
)

# Sigmoid function
sigmoid_probability = (
    1
    / (
        1
        + np.exp(-weighted_sum)
    )
)


print("\n" + "=" * 70)
print("MANUAL LOGISTIC REGRESSION CALCULATION")
print("=" * 70)
print("Feature values:")
print(feature_values)
print("\nCoefficients:")
print(coefficients)
print("\nIntercept:")
print(intercept)
print("\nWeighted Sum z:")
print(
    round(weighted_sum, 4)
)
print("\nSigmoid Probability:")
print(
    round(sigmoid_probability, 4)
)
print("\nProbability from predict_proba():")
print(
    round(new_probabilities[1], 4)
)


# 14. Final explanation
print("\n" + "=" * 70)
print("FINAL EXPLANATION")
print("=" * 70)
print(
    """
Logistic Regression first calculates a weighted sum:

z = feature1 × weight1
  + feature2 × weight2
  + feature3 × weight3
  + intercept

The weighted sum is then passed through the sigmoid function.

The sigmoid converts the result into a probability between 0 and 1.

If the probability is greater than or equal to 0.5,
the model predicts Class 1.

Otherwise, it predicts Class 0.
"""
)

#ِAccurecy
# 95 customers stay
# 5 customers churn

y_true = (
    [0] * 95
    + [1] * 5
)


# Useless model:
# predicts Stay for every customer

y_pred = [0] * 100


accuracy = accuracy_score(
    y_true,
    y_pred
)


print("Number of customers:", len(y_true))
print(
    "Actual Stay customers:",
    y_true.count(0)
)
print(
    "Actual Churn customers:",
    y_true.count(1)
)
print(
    "Predicted Churn customers:",
    y_pred.count(1)
)
print(
    "\nAccuracy:",
    round(accuracy, 2)
)
print(
    "Accuracy percentage:",
    f"{accuracy:.0%}"
)

#Confusion Matrix 
# 1. Create the dataset

data1 = {
    "monthly_usage": [
        5, 8, 10, 12, 15,
        18, 20, 22, 25, 28,
        30, 32, 35, 38, 40,
        42, 45, 48, 50, 55
    ],

    "support_calls": [
        8, 7, 6, 7, 5,
        5, 4, 5, 3, 4,
        2, 3, 2, 2, 1,
        2, 1, 1, 0, 1
    ],

    "satisfaction": [
        1, 2, 2, 2, 3,
        3, 3, 3, 4, 4,
        4, 4, 5, 4, 5,
        5, 5, 5, 5, 5
    ],

    # 0 = Stay
    # 1 = Churn
    "churn": [
        1, 1, 1, 1, 1,
        1, 0, 1, 0, 0,
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data1)
print("=" * 70)
print("DATASET")
print("=" * 70)
print(df)
print("\nDataset shape:")
print(df.shape)



# 2. Check the target distribution
print("\n" + "=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)
print(df["churn"].value_counts())



# 3. Separate Features X and Target y
X = df[
    [
        "monthly_usage",
        "support_calls",
        "satisfaction"
    ]
]

y = df["churn"]
print("\nFeatures X:")
print(X.head())
print("\nTarget y:")
print(y.head())



# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=13,
    stratify=y
)
print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)



# 5. Create the Logistic Regression model
model = LogisticRegression(
    max_iter=1000
)


# 6. Train the model
model.fit(
    X_train,
    y_train
)
print("\nModel training completed.")



# 7. Generate predictions
predictions = model.predict(
    X_test
)
print("\nActual classes:")
print(y_test.values)
print("\nPredicted classes:")
print(predictions)


# 8. Create results table
results = X_test.copy()
results["Actual Class"] = y_test.values
results["Predicted Class"] = predictions
results["Correct"] = (
    results["Actual Class"]
    == results["Predicted Class"]
)
print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED")
print("=" * 70)
print(results)



# 9. Calculate the Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions
)
print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)
print(cm)


# 10. Extract TN, FP, FN, TP
tn, fp, fn, tp = cm.ravel()
print("\nTrue Negative  TN:", tn)
print("False Positive FP:", fp)
print("False Negative FN:", fn)
print("True Positive  TP:", tp)


# 11. Calculate Accuracy
accuracy = accuracy_score(
    y_test,
    predictions
)
print("\nAccuracy:")
print(round(accuracy, 4))
print(
    "Accuracy Percentage:",
    f"{accuracy:.2%}"
)


# 12. Display the Confusion Matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Stay",
        "Churn"
    ]
)

display.plot()
plt.title("Customer Churn Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()


#Precision & Recall & F1 Score

df = pd.read_csv(r"C:\Users\sadee\OneDrive\Desktop\The Main Folder\The Main Folder\Week 3\Day 3\customer_churn_day3.csv")
print("First 5 rows:")
print(df.head())


# 2. Select Features and Target
# customer_id is only an identifier,
# so we will not use it as a feature.
X = df[
    [
        "tenure_months",
        "monthly_charge",
        "support_calls_3m",
        "late_payments_6m",
        "satisfaction_score",
        "weekly_usage_hours",
        "auto_pay"
    ]
]
y = df["churn"]


# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. Create and train the model
model = LogisticRegression(
    max_iter=1000
)
model.fit(
    X_train,
    y_train
)


# 5. Generate predictions
predictions = model.predict(
    X_test
)


# 6. Confusion Matrix
cm = confusion_matrix(
    y_test,
    predictions
)
print("\nConfusion Matrix:")
print(cm)


# Confusion Matrix order:
#
# [[TN, FP],
#  [FN, TP]]

tn, fp, fn, tp = cm.ravel()
print("\nTN:", tn)
print("FP:", fp)
print("FN:", fn)
print("TP:", tp)


# 7. Calculate Precision, Recall and F1
precision = precision_score(
    y_test,
    predictions
)
recall = recall_score(
    y_test,
    predictions
)
f1 = f1_score(
    y_test,
    predictions
)
print("\nClassification Metrics:")
print(
    "Precision:",
    round(precision, 4)
)
print(
    "Recall:",
    round(recall, 4)
)
print(
    "F1-score:",
    round(f1, 4)
)


# 8. Classification Report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Stay",
            "Churn"
        ],
        zero_division=0
    )
)

#AUC & ROC
# 1. Load the dataset
df = pd.read_csv(r"C:\Users\sadee\OneDrive\Desktop\The Main Folder\The Main Folder\Week 3\Day 3\customer_churn_day3.csv")
print("Dataset loaded successfully.")
print(df.head())


# 2. Select Features and Target
X = df[
    [
        "tenure_months",
        "monthly_charge",
        "support_calls_3m",
        "late_payments_6m",
        "satisfaction_score",
        "weekly_usage_hours",
        "auto_pay"
    ]
]

y = df["churn"]


# 3. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# 4. Create and Train the Model
model = LogisticRegression(
    max_iter=1000
)
model.fit(
    X_train,
    y_train
)


# 5. Get Churn Probabilities
probabilities = model.predict_proba(
    X_test
)


# Column 0 = Stay probability
# Column 1 = Churn probability
churn_probabilities = probabilities[:, 1]
print("\nFirst 5 Churn Probabilities:")
print(
    churn_probabilities[:5]
)


# 6. Calculate AUC Score
auc_score = roc_auc_score(
    y_test,
    churn_probabilities
)
print(
    "\nAUC-ROC Score:",
    round(auc_score, 4)
)


# 7. Calculate ROC Curve Points
fpr, tpr, thresholds = roc_curve(
    y_test,
    churn_probabilities
)



# 8. Plot ROC Curve
plt.figure(figsize=(8, 6))


# ROC curve for our model
plt.plot(
    fpr,
    tpr,
    linewidth=2,
    label=f"Logistic Regression (AUC = {auc_score:.4f})"
)


# Random guessing line
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Guessing (AUC = 0.5)"
)


plt.title("ROC Curve - Customer Churn")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()