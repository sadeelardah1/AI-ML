import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    StratifiedKFold
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score



# Load the dataset
file =r"C:\Users\sadee\OneDrive\Desktop\The Main Folder\The Main Folder\Week 4\Day 2\train_and_test2.csv"
titanic = pd.read_csv(file)
titanic = titanic.rename(columns={"2urvived": "Survived"})
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
print("Dataset shape:", X.shape)



# Create train, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42
)
print("Training rows  :", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows      :", len(X_test))



# Create the Day 1 Random Forest model
model = RandomForestClassifier(
    max_depth=5,
    n_estimators=100,
    random_state=42
)



# Create reproducible stratified 5-fold CV
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)



# Step 1: Run 5-fold cross-validation
cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=skf,
    scoring="f1"
)
print("\nFold scores:", np.round(cv_scores, 3))


# Step 2: Mean and standard deviation
mean_f1 = cv_scores.mean()
std_f1 = cv_scores.std()
print(f"Mean F1 across 5 folds : {mean_f1:.3f}")
print(f"Std. dev. across folds : {std_f1:.3f}")



# Step 3: Compare with the Day 1 validation split
model.fit(X_train, y_train)
validation_predictions = model.predict(X_val)
single_split_f1 = f1_score(
    y_val,
    validation_predictions
)

print(
    f"\nDay 1 single validation-split F1 : "
    f"{single_split_f1:.3f}"
)

print(
    f"Day 2 5-fold cross-validated mean: "
    f"{mean_f1:.3f} (± {std_f1:.3f})"
)


# Step 4: Confirm class balance across stratified folds
print(
    "\nOverall survival rate in y_train:",
    f"{y_train.mean():.1%}"
)

for i, (_, val_idx) in enumerate(
    skf.split(X_train, y_train),
    start=1
):
    fold_rate = y_train.iloc[val_idx].mean()
    print(
        f"Fold {i} survival rate: "
        f"{fold_rate:.1%}")