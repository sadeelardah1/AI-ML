#2.1 linear Regression 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score)



# 1. Create the dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "score": [48, 54, 61, 63, 71, 74, 82, 85, 91, 96]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)


# 2. Separate Feature X and Target y
# X must be a two-dimensional table
X = df[["study_hours"]]

# y is the target column
y = df["score"]
print("\nFeature X:")
print(X)
print("\nTarget y:")
print(y)



# 3. Create the Linear Regression model
model = LinearRegression()



# 4. Fit the model
# For this introductory visualization, we train on all rows.
# Train/Test Split will be used in the next section.
model.fit(X, y)


# 5. Generate predictions for the existing data
predicted_scores = model.predict(X)
df["predicted_score"] = predicted_scores
print("\nActual and Predicted Values:")
print(df)


# 6. Display coefficient and intercept
coefficient = model.coef_[0]
intercept = model.intercept_
print("\nCoefficient:", coefficient)
print("Intercept:", intercept)



# 7. Display the learned equation
print(
    f"\nRegression Equation: "
    f"Predicted Score = {intercept:.2f} "
    f"+ {coefficient:.2f} × Study Hours"
)



# 8. Predict the score for 7.5 study hours
new_data = pd.DataFrame({
    "study_hours": [7.5]
})

new_prediction = model.predict(new_data)

print(
    "\nPredicted score for 7.5 study hours:",
    round(new_prediction[0], 2)
)


# 9. Draw the actual points and regression line
plt.figure(figsize=(9, 6))

# Actual observations
plt.scatter(
    df["study_hours"],
    df["score"],
    label="Actual Scores",
    s=80
)

# Best-fit regression line
plt.plot(
    df["study_hours"],
    df["predicted_score"],
    label="Best-Fit Line",
    linewidth=2
)

# The new predicted point
plt.scatter(
    new_data["study_hours"],
    new_prediction,
    label="Prediction for 7.5 Hours",
    s=120,
    marker="X"
)

plt.title("Linear Regression: Study Hours vs Score")
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()


#2.2 Training and Predicting
# 1. Create the dataset
data1 = {
    "study_hours": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    ],

    "score": [
        48, 54, 61, 63, 71,
        74, 82, 85, 91, 96,
        98, 103, 109, 112, 118
    ]
}

df = pd.DataFrame(data1)
print("=" * 60)
print("COMPLETE DATASET")
print("=" * 60)
print(df)



# 2. Separate Features X and Target y
X = df[["study_hours"]]
y = df["score"]
print("\nX shape:", X.shape)
print("y shape:", y.shape)


# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("\nX_train:")
print(X_train)
print("\ny_train:")
print(y_train)
print("\nX_test:")
print(X_test)
print("\ny_test:")
print(y_test)


# 4. Instantiate the model
model = LinearRegression()

# 5. Train the model
model.fit(X_train, y_train)
print("\nModel training completed.")


# 6. Generate predictions
predictions = model.predict(X_test)
print("\nPredictions:")
print(predictions)


# 7. Compare actual and predicted values
results = pd.DataFrame({
    "Study Hours": X_test["study_hours"].values,
    "Actual Score": y_test.values,
    "Predicted Score": predictions
})

results["Error"] = (
    results["Actual Score"]
    - results["Predicted Score"]
)

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED")
print("=" * 60)
print(results.round(2))



#2.3Interpreting Coefficients
# Create dataset
data2 = {
    "study_hours": [2, 4, 6, 8, 3, 5, 7, 9],
    "attendance": [60, 55, 80, 75, 90, 65, 85, 70]
}

df = pd.DataFrame(data2)


# Create a continuous target
df["score"] = (
    20
    + 4 * df["study_hours"]
    + 0.5 * df["attendance"]
)


# Separate Features and Target
X = df[
    [
        "study_hours",
        "attendance"
    ]
]

y = df["score"]


# Instantiate
model = LinearRegression()


# Fit
model.fit(X, y)


# Read coefficients and intercept
coefficients_table = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("Dataset:")
print(df)

print("\nCoefficients:")
print(coefficients_table)

print("\nIntercept:")
print(model.intercept_)


# Display learned equation
print(
    "\nLearned Equation:"
)

print(
    f"Predicted Score = "
    f"{model.intercept_:.2f} "
    f"+ {model.coef_[0]:.2f} × Study Hours "
    f"+ {model.coef_[1]:.2f} × Attendance"
)


# Predict a new observation
new_student = pd.DataFrame({
    "study_hours": [6],
    "attendance": [80]
})

prediction = model.predict(new_student)

print(
    "\nPredicted score:",
    round(prediction[0], 2)
)




#Regression Metrics
# 1. Create the dataset
data3 = {
    "study_hours": [
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15
    ],

    "score": [
        48, 54, 61, 63, 71,
        74, 82, 85, 91, 96,
        98, 103, 109, 112, 118
    ]
}

df = pd.DataFrame(data3)



# 2. Separate Features and Target
X = df[["study_hours"]]
y = df["score"]


# 3. Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)



# 4. Create and train the model
model = LinearRegression()

model.fit(
    X_train,
    y_train
)



# 5. Generate predictions
predictions = model.predict(
    X_test
)



# 6. Calculate Regression Metrics
mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)



# 7. Display results
results = pd.DataFrame({
    "Study Hours": X_test["study_hours"].values,
    "Actual Score": y_test.values,
    "Predicted Score": predictions
})

results["Absolute Error"] = abs(
    results["Actual Score"]
    - results["Predicted Score"]
)

print("Actual vs Predicted:")
print(results.round(2))

print("\nRegression Metrics:")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R²:", round(r2, 4))