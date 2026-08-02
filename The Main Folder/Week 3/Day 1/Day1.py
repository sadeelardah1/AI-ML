import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


#Supervised learning
# Create labeled data
data = {
    "study_hours": [2, 4, 6, 8, 3, 7, 5, 9],
    "attendance": [55, 65, 80, 92, 60, 85, 72, 95],
    "assignments": [3, 5, 7, 9, 4, 8, 6, 10],
    "passed": [0, 0, 1, 1, 0, 1, 1, 1]
}

df = pd.DataFrame(data)

# Display the complete dataset
print("Complete Dataset:")
print(df)

# Separate features from target
X = df.drop("passed", axis=1)
y = df["passed"]

print("\nFeatures X:")
print(X)

print("\nTarget y:")
print(y)

# Verify dimensions
print("\nX shape:", X.shape)
print("y shape:", y.shape)

print(
    "Same number of samples:",
    len(X) == len(y)
)


#Regression
# 1. Create the dataset
data1 = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "score": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
}

df = pd.DataFrame(data1)

print("Dataset:")
print(df)

# 2. Separate Features X and Target y
X = df[["study_hours"]]
y = df["score"]

print("\nFeatures X:")
print(X)
print("\nTarget y:")
print(y)


# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nX_train:")
print(X_train)
print("\nX_test:")
print(X_test)



# 4. Instantiate the model
model = LinearRegression()


# 5. Fit the model
model.fit(X_train, y_train)


# 6. Predict the test data
predictions = model.predict(X_test)

results = pd.DataFrame({
    "Study Hours": X_test["study_hours"].values,
    "Actual Score": y_test.values,
    "Predicted Score": predictions
})

print("\nRegression Results:")
print(results)


# 7. Score the model
score = model.score(X_test, y_test)
print("\nR² Score:", score)


# 8. Predict a new value
new_student = pd.DataFrame({
    "study_hours": [7.5]
})

predicted_score = model.predict(new_student)

print(
    "\nPredicted score for 7.5 study hours:",
    predicted_score[0]
)


#Classification 
# 1. Create the dataset
data2 = {
    "study_hours": [
        1, 1.5, 2, 2.5, 3,
        3.5, 4, 4.2, 4.5, 4.8,
        5.2, 5.5, 6, 6.5, 7,
        7.5, 8, 8.5, 9, 10
    ],

    "passed": [
        0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
}

df = pd.DataFrame(data2)

print("Dataset:")
print(df)



# 2. Separate Features X and Target y
X = df[["study_hours"]]
y = df["passed"]

print("\nFeatures X:")
print(X)

print("\nTarget y:")
print(y)



# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nX_train:")
print(X_train)

print("\nX_test:")
print(X_test)



# 4. Instantiate the model
model = LogisticRegression()



# 5. Fit the model
model.fit(X_train, y_train)



# 6. Predict the test data
predictions = model.predict(X_test)

results = pd.DataFrame({
    "Study Hours": X_test["study_hours"].values,
    "Actual Class": y_test.values,
    "Predicted Class": predictions
})

print("\nClassification Results:")
print(results)



# 7. Score the model
score = model.score(X_test, y_test)

print("\nAccuracy:", score)



# 8. Predict a new student
new_student = pd.DataFrame({
    "study_hours": [6]
})

predicted_class = model.predict(new_student)
probabilities = model.predict_proba(new_student)

print(
    "\nPredicted class for 6 study hours:",
    predicted_class[0]
)

print(
    "Probability of failing:",
    probabilities[0][0]
)

print(
    "Probability of passing:",
    probabilities[0][1]
)