# This File For Practice
#EDA
"""
1. Load the dataset
2. Inspect shape and columns
3. Check data types
4. Check missing values
5. Examine numeric distributions
6. Examine categorical frequencies
7. Detect outliers
8. Document findings
 
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data = {"age": [20, 22, 21, 23, 24, 25, 22, 120], "study_hours": [4, 6, 5, 7, 8, 6, 5, 9], "score": [70, 80, 75, 90, 95, 85, 78, 88], "track": ["AI", "Web", "AI", "Data", "AI", "Web", "Data", "AI"]}
df = pd.DataFrame(data)
print(df)
print("Show The First Rows : ",df.head())
print("Shape : ",df.shape)
print("Columns Name : ",df.columns)
print("Type Of Data : ",df.dtypes)
print("All Information : ",df.info())
print("Describe The Data : ",df.describe())

print(df.isna().sum()) #To Check The missing values if all results = zero , that maen there is no missing values

numeric_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(exclude="number").columns
print("Numeric columns:")
print(numeric_columns.tolist())
print("\nCategorical columns:")
print(categorical_columns.tolist())



#seaborn
sns.set_theme(style="darkgrid")
data1 = {
    "study_hours": [2, 4, 5, 7, 8],
    "score": [60, 70, 75, 88, 95],
    "track": ["AI", "Web", "AI", "Data", "AI"]
}

df = pd.DataFrame(data1)

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="study_hours",
    y="score",
    hue="track"
)

plt.title("Study Hours vs Exam Score")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.tight_layout()
plt.show()



#


data3 = {
    "age": [20, 22, 21, 23, 24, 25],
    "study_hours": [4, 6, 5, 7, 8, 6],
    "score": [70, 80, 75, 90, 95, 85],
    "track": ["AI", "Web", "AI", "Data", "AI", "Web"]
}

df = pd.DataFrame(data3)

print("=" * 50)
print("DATASET")
print("=" * 50)
print(df)

numeric_columns = df.select_dtypes(
    include="number"
).columns

categorical_columns = df.select_dtypes(
    exclude="number"
).columns

print("\n" + "=" * 50)
print("NUMERICAL COLUMNS")
print("=" * 50)
print(numeric_columns.tolist())

print("\n" + "=" * 50)
print("CATEGORICAL COLUMNS")
print("=" * 50)
print(categorical_columns.tolist())

print("\n" + "=" * 50)
print("SCORE SUMMARY")
print("=" * 50)
print(df["score"].describe())

print("\n" + "=" * 50)
print("TRACK COUNTS")
print("=" * 50)
print(df["track"].value_counts())


#Histogram
sns.set_theme(style="whitegrid")
data4 = {
    "age": [20, 22, 21, 23, 24, 25, 22, 120],
    "study_hours": [4, 6, 5, 7, 8, 6, 5, 9],
    "score": [70, 80, 75, 90, 95, 85, 78, 88],
    "track": [
        "AI",
        "Web",
        "AI",
        "Data",
        "AI",
        "Web",
        "Data",
        "AI"
    ]
}

df = pd.DataFrame(data4)

numeric_columns = df.select_dtypes(
    include="number"
).columns

for column in numeric_columns:
    print("=" * 50)
    print(f"Analyzing: {column}")
    print("=" * 50)
    print(df[column].describe())

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=column,
        bins=8,
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    

#Box Plot
sns.set_theme(style="whitegrid")
data5 = {
    "age": [
        20, 21, 22, 22, 23,
        24, 25, 26, 27, 120
    ]
}

df = pd.DataFrame(data5)


print("=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)
print(df["age"].describe())


Q1 = df["age"].quantile(0.25)
median = df["age"].median()
Q3 = df["age"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print("\n" + "=" * 50)
print("IQR CALCULATIONS")
print("=" * 50)
print("Q1:", Q1)
print("Median:", median)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)


outliers = df[
    (df["age"] < lower_bound)
    | (df["age"] > upper_bound)
]
print("\n" + "=" * 50)
print("POTENTIAL OUTLIERS")
print("=" * 50)
print(outliers)


plt.figure(figsize=(8, 4))

sns.boxplot(
    data=df,
    x="age"
)
plt.title("Age Box Plot")
plt.xlabel("Age")
plt.tight_layout()
plt.show()


#Count Plot

# Apply a clear Seaborn theme
sns.set_theme(style="whitegrid")

# Create the dataset
data6 = {
    "track": [
        "AI",
        "Web",
        "AI",
        "Data",
        "AI",
        "Web",
        "Cybersecurity",
        "Data",
        "AI",
        "Web"
    ]
}

df = pd.DataFrame(data6)

# Display the dataset
print("=" * 50)
print("DATASET")
print("=" * 50)
print(df)

# Count each category
track_counts = df["track"].value_counts()
print("\n" + "=" * 50)
print("TRACK COUNTS")
print("=" * 50)
print(track_counts)

# Calculate category percentages
track_percentages = (
    df["track"]
    .value_counts(normalize=True)
    .mul(100)
)
print("\n" + "=" * 50)
print("TRACK PERCENTAGES")
print("=" * 50)
print(track_percentages.round(1))

# Create the count plot
plt.figure(figsize=(8, 5))
ax = sns.countplot(
    data=df,
    x="track",
    order=track_counts.index
)

# Add a value label above each bar
for container in ax.containers:
    ax.bar_label(container)

# Format the plot
plt.title("Number of Students by Track")
plt.xlabel("Track")
plt.ylabel("Student Count")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()


#EDK Plot
# Apply a clear Seaborn theme
sns.set_theme(style="whitegrid")

# Create the dataset
data7 = {
    "score": [
        55, 60, 62, 65, 68,
        70, 72, 74, 75, 76,
        78, 80, 82, 85, 88,
        90, 92, 95
    ]
}

df = pd.DataFrame(data7)

# Display descriptive statistics
print("=" * 50)
print("DESCRIPTIVE STATISTICS")
print("=" * 50)
print(df["score"].describe())

# Create the KDE plot
plt.figure(figsize=(8, 5))

sns.kdeplot(
    data=df,
    x="score",
    fill=True
)

plt.title("Exam Score Density")
plt.xlabel("Exam Score")
plt.ylabel("Density")
plt.tight_layout()
plt.show()




# Outliers
df = pd.DataFrame({
    "score": [65, 70, 72, 75, 78, 80, 82, 85, 90, 150]
})

# Calculate Q1 and Q3
Q1 = df["score"].quantile(0.25)
Q3 = df["score"].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[
    (df["score"] < lower_bound)
    | (df["score"] > upper_bound)
]

# Display calculations
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

print("\nOutliers:")
print(outliers)

# Draw box plot
sns.boxplot(
    data=df,
    x="score"
)

plt.title("Score Outliers")
plt.xlabel("Score")
plt.show()