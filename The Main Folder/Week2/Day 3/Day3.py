import numpy as np
import pandas as pd


# Diffs btw Scalar | Vector | Matrix
# Scalar : only one value this value indicates something such as : age = 25 =>this scalar 
# Vector : More than one value that indicates something ,such as : customer = [25, Palestine, Computer science] , 25 indicates to age , Palestine indicates to Country .... etc
customer = np.array(["Sadeel", 25, "Palestine", "Computer Science"])
print("Customer vector:", customer)
print("Shape:", customer.shape)# (3,) means 1D Array with 3 elements these elements are features not 3 rows 
print("Number of dimensions:", customer.ndim)
print("Number of elements:", customer.size)
print("Data type:", customer.dtype)

print("--------------------------------------------\n")
name =customer[0]
age = customer[1]
country = customer[2]
major = customer[3]
print("Name : ", name)
print("Age : ",age)
print("Country : ",country)
print("Major : ",major)
#Update Vector
customer[0] = "Sadeel Ahmad"
customer[1] = 21
print("--------------------------------------------\n")
print("The Customer info after updated :")
print(customer)
print("--------------------------------------------\n")

# Operation on vector
#First : single vector 

v1 = np.array([3, 6, 7])
print("Vector : ",v1)
print("All Numbers Plused 2 :", v1+2)
print("All Numbers Multiplies by 3 :", v1*3)
print("--------------------------------------------\n")

#Second : Multi Vectors (Condition : the Vectors should have the same shape to avoid the conflicts)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)
print("--------------------------------------------\n")




#Matrices : each row = sample , each column = feature , so the matrix takes (samples , features)

# Read from CSV file
students_df = pd.read_csv(r"C:\Users\sadee\OneDrive\Desktop\The Main Folder\The Main Folder\Week2\Day 3\students_linear_algebra.csv")
print("Original DataFrame:")
print(students_df)


# Inspect DataFrame
print("\nDataFrame shape:", students_df.shape)
print("Columns:", students_df.columns.tolist())
print("\nData types:")
print(students_df.dtypes)


# Select feature columns
feature_columns = ["age", "study_hours", "exam_score"]
features_df = students_df[feature_columns]


# Convert to NumPy matrix
X = features_df.to_numpy()
print("\nFeature Matrix X:")
print(X)
print("\nMatrix shape:", X.shape)
print("Matrix ndim:", X.ndim)
print("Matrix size:", X.size)
print("Matrix dtype:", X.dtype)


# Samples and features
number_of_samples = X.shape[0]
number_of_features = X.shape[1]
print("\nNumber of samples:", number_of_samples)
print("Number of features:", number_of_features)


# Extract first student vector
first_student = X[0]
print("\nFirst Student Vector:")
print(first_student)
print("Vector shape:", first_student.shape)
print("Vector ndim:", first_student.ndim)
print("Vector size:", first_student.size)
print("Vector dtype:", first_student.dtype)


# Read vector components
print("\nFirst student's age:", first_student[0])
print("First student's study hours:", first_student[1])
print("First student's exam score:", first_student[2])


# Extract rows
print("\nSecond student:")
print(X[1])
print("\nFirst two students:")
print(X[:2])


# Extract columns
ages = X[:, 0]
study_hours = X[:, 1]
exam_scores = X[:, 2]
print("\nAges:")
print(ages)
print("\nStudy hours:")
print(study_hours)
print("\nExam scores:")
print(exam_scores)


# Select multiple columns
age_and_score = X[:, [0, 2]]
print("\nAge and Exam Score Matrix:")
print(age_and_score)
print("Selected matrix shape:", age_and_score.shape)


# Access one value
print("\nSecond student's exam score:", X[1, 2])


# Create a safe copy and modify it
X_modified = X.copy()
X_modified[0, 1] = 6
print("\nOriginal X:")
print(X)
print("\nModified X:")
print(X_modified)


# Element-wise operations
matrix_a = np.array([[1, 2], [3, 4]])
matrix_b = np.array([[5, 6], [7, 8]])
print("\nMatrix A + 10:")
print(matrix_a + 10)
print("\nMatrix A * 2:")
print(matrix_a * 2)
print("\nMatrix A + Matrix B:")
print(matrix_a + matrix_b)


# Feature statistics
print("\nMean age:", ages.mean())
print("Mean study hours:", study_hours.mean())
print("Mean exam score:", exam_scores.mean())
print("Minimum exam score:", exam_scores.min())
print("Maximum exam score:", exam_scores.max())
print("--------------------------------------------\n")


#Dot Product
#Note : Features and Weights should be the same lenght لانه كل فيتشرز الها وزن محدد خاص فيها 
student = np.array([20, 5, 80])
weights = np.array([0.1, 2.0, 0.5])

dot_product = np.dot(student, weights) # or dot_product = student @ weights , @ mean -> dot product

print("Student vector:", student)
print("Weights:", weights)
print("Dot product:", dot_product)
print("--------------------------------------------\n")




#another Ex

students_df1 = pd.read_csv(r"C:\Users\sadee\OneDrive\Desktop\The Main Folder\The Main Folder\Week2\Day 3\students_linear_algebra.csv")
print("Original Dataset:")
print(students_df)


feature_columns = ["age", "study_hours", "exam_score"]
X = students_df1[feature_columns].to_numpy()
print("\nFeature Matrix X:")
print(X)
print("\nX shape:", X.shape)

first_student = X[0]
print("\nFirst Student Vector:")
print(first_student)
print("Vector shape:", first_student.shape)


weights = np.array([0.1, 2.0, 0.5])
print("\nWeights:")
print(weights)
print("Weights shape:", weights.shape)

products = first_student * weights
print("\nElement-wise products:")
print(products)

manual_dot_product = products.sum()
print("\nManual dot product:")
print(manual_dot_product)


numpy_dot_product = np.dot(first_student, weights)
print("\nNumPy dot product:")
print(numpy_dot_product)

at_operator_result = first_student @ weights
print("\nResult using @:")
print(at_operator_result)


bias = 3
prediction = first_student @ weights + bias
print("\nBias:", bias)
print("Final prediction:", prediction)
print("--------------------------------------------\n")


# Matrix Multiplication
X = np.array([[20, 5, 80], [22, 7, 90], [21, 4, 75], [23, 6, 85]])
weights = np.array([0.1, 2.0, 0.5])
predictions = X @ weights

print("Predictions:")
print(predictions)
print("Predictions shape:", predictions.shape)
print("--------------------------------------------\n")




#Transpose
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("Original Matrix:")
print(matrix)
print("\nOriginal Shape:")
print(matrix.shape)

transpose_matrix = matrix.T
print("\nTranspose Matrix:")
print(transpose_matrix)
print("\nTranspose Shape:")
print(transpose_matrix.shape)
print("--------------------------------------------\n")
