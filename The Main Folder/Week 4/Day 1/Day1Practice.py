import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

file =r"C:\Users\sadee\OneDrive\Desktop\The Main Folder\The Main Folder\Week 4\Day 1\train_and_test2.csv"
titanic = pd.read_csv(file)
titanic = titanic.rename(
    columns={"2urvived": "Survived"}
)

features = [
    "Age",
    "Fare",
    "Sex",
    "sibsp",
    "Parch",
    "Pclass",
    "Embarked"
]

X = titanic[features].copy()
y = titanic["Survived"].copy()

print("X shape:", X.shape)
print("y shape:", y.shape)


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
print("Total rows:", len(X))
print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

total = len(X)
print("Train percentage:",round(len(X_train) / total * 100, 2),"%")
print("Validation percentage:",round(len(X_val) / total * 100, 2),"%")
print("Test percentage:",round(len(X_test) / total * 100, 2),"%")


split_names = ["Training", "Validation", "Test"]
split_sizes = [len(X_train),len(X_val),len(X_test)]
plt.figure(figsize=(8, 5))
plt.bar(split_names,split_sizes)
plt.title("Train / Validation / Test Split")
plt.xlabel("Dataset Split")
plt.ylabel("Number of Rows")

for i, size in enumerate(split_sizes):
    percentage = size / len(X) * 100

    plt.text(i,size + 10,f"{size} rows\n({percentage:.1f}%)",ha="center")

plt.tight_layout()
plt.show()



#Max Depth
candidate_depths = [3, 5, 10, None]
validation_results = []

for depth in candidate_depths:

    model = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=42) # n_estimators=100 : Means random forests will build:100 Decision Trees
    model.fit(X_train,y_train)
    val_predictions = model.predict(X_val)
    val_f1 = f1_score(y_val,val_predictions)
    validation_results.append(
        {
            "max_depth": depth,
            "Validation F1": round(val_f1, 3)
        }
    )

validation_table = pd.DataFrame(validation_results)
print("\nValidation Results:")
print(validation_table)

#Select the best Mac Depth
best_row = validation_table.sort_values("Validation F1", ascending=False).iloc[0]
best_depth = best_row["max_depth"]
if pd.notna(best_depth):
    best_depth = int(best_depth)
else:
    best_depth = None

print("\nBest max_depth chosen using validation:",best_depth)

#Final test evaluation
final_model = RandomForestClassifier(max_depth=best_depth, n_estimators=100, random_state=42)
final_model.fit(X_train, y_train)
test_predictions = final_model.predict(X_test)
test_f1 = f1_score(y_test, test_predictions)
print(f"Final, one-time Test F1: {test_f1:.3f}")