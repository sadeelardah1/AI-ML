#3.2  Creating Arrays

import numpy as np
print("Numpy version :"+ np.__version__)
#to convert a list to a numpy array(simple array)
a = np.array([1, 2, 3, 4, 5])
print(a)
print(type(a)) # a is not a list it is a numpy array so I can use numpy methods on it
print("--------------------------------")

#buiding a 2D array
DArr =np.array([[1,2,3],[4,5,6]])
print(DArr)
print("--------------------------------")

#2D array with zeros & ones , NOTE : the . in array mean that the array is a float array
zearr = np.zeros((4,6)) # array of 4 rows and 6 columns , all values inside the array will be 0
print(zearr)
onesarr = np.ones((5,2)) # array of 5 rows and 2 columns , all values inside the array will be 1
print(onesarr)

onedarr = np.zeros(10) # one dimensional array of 10 elements , all values inside the array will be 0
print(onedarr)  # () mean one dimensional array , ((,)) mean two dimensional array
print("--------------------------------")


# arange function to create a range of numbers
arr0 =np.arange(10) # 0 to 9 , the default step is 1 , the end exclusive
print(arr0)
arr1=np.arange(50, 61)
print(arr1)
arr2 = np.arange(1, 10, 2) # (start, stop, step) , the end exclusive 
print(arr2)
print("--------------------------------")

# linspace function to create a range of numbers
arr3 = np.linspace(0, 1, 5) # (start, stop, num) , the end inclusive , num is the number of elements in the array
print("linspace function")
print(arr3)
arr4 = np.linspace(5,100,3)
print(arr4) # the output will be [  5.  52.5 100. ] , the step is calculated automatically
print("--------------------------------")
# random function to create a range of numbers
# there is two types of methode using with random function , the first one is using np.random.rand(x) :give random numbers between 0 and 1 ,x # of elements 
# and the second one is using np.random.seed(x) :give random numbers , x is the beginning number of the random numbers , and the output will be the same every time you run the code

np.random.seed(20) #if i run the code again the output will be the same because i set the seed to 20 , if i remove the seed the output will be different every time i run the code
randarr = np.random.rand(3)
print(randarr)
print("--------------------------------")



#3.3  Array Attributes, Indexing & Slicing
print("Attributes")
atarr = np.array([[1,2,3],[4,5,6],[7,8,9]])
print("Shape of the array : ", atarr.shape)
print("Number of dimensions : ", atarr.ndim)
print("Size of the array : ", atarr.size)  
print("Data type of the array : ", atarr.dtype)
print("--------------------------------")
t = np.array([1, 2, 3, 4, 5])
print("Shape of the array : ", t.shape)# it takes the length of the array and put it in a tuple and the length put first cause it is a one dimensional array , if the array is 2D it will take the number of rows and columns and put them in a tuple
print("Number of dimensions : ", t.ndim)
print("Size of the array : ", t.size)  
print("Data type of the array : ", t.dtype)
print("--------------------------------")



#Indexing 
print("Indexing")
indarr = np.array([[1,2,3],[4,5,6]]) # the indexing count start from 0
print(indarr[0, 1])
print(indarr[1, 2])
print("--------------------------------")


#ٍSlicing
print("Slicing")
sliarr=np.array([["Banan","Watermelon","Cherry","Mango"],["Apple","Orange","Grapes","Strawberry"],["Pineapple","Kiwi","Peach","Pear"]])
print(sliarr)
print("All rows ,column 1 : ",sliarr[:,1])
print("All columns , row 2 : ",sliarr[2,:])
print("All rows , first 2 columns",sliarr[:, 0:2])
print("--------------------------------")


#vectorization
print("Vectorization")
scores = np.array([70,80,90,60,66,75])
new_scores = scores+5
print(new_scores)

print("Original array:", scores)
print("Add 10:", scores + 10)
print("Subtract 1:", scores - 1)
print("Multiply by 2:", scores * 2)
print("Divide by 2:", scores / 2)
print("Power of 2:", scores ** 2)

#Operations in 1D
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print("Addition:", a + b)
print("Subtraction:", b - a)
print("Multiplication:", a * b)
print("Division:", b / a)

#Operation in 2D
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

print("Original matrix:")
print(matrix)

print("\nMultiply by 10:")
print(matrix * 10)

print("\nAdd 5:")
print(matrix + 5)

print("\nSquare:")
print(matrix ** 2)

#Aggregation
numbers = np.array([10, 20, 30, 40, 50])

print("Sum:", numbers.sum())
print("Mean:", numbers.mean())
print("Minimum:", numbers.min())
print("Maximum:", numbers.max())
print("Standard deviation:", numbers.std())

#axis 
matrix = np.array([
    [10, 20, 30],
    [40, 50, 60]
])
print(matrix.shape)
print(matrix.sum())
print(matrix.sum(axis=0))
print(matrix.sum(axis=1))




temperatures = np.array([18, 22, 25, 31, 15, 28, 35])

print("Original temperatures:")
print(temperatures)

temperatures_plus_2 = temperatures + 2
print("\nTemperatures after adding 2:")
print(temperatures_plus_2)

temperatures_squared = temperatures ** 2
print("\nSquared temperatures:")
print(temperatures_squared)


average_temperature = temperatures.mean()
print("\nAverage temperature:")
print(average_temperature)


above_average = temperatures[temperatures > average_temperature]
print("\nTemperatures above average:")
print(above_average)


between_20_and_30 = temperatures[
    (temperatures >= 20) & (temperatures <= 30)
]
print("\nTemperatures between 20 and 30:")
print(between_20_and_30)


greater_than_25_count = (temperatures > 25).sum()
print("\nNumber of temperatures greater than 25:")
print(greater_than_25_count)

#Broadcasting
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

row = np.array([10, 20, 30])

result = matrix + row

print("Matrix:")
print(matrix)

print("\nRow:")
print(row)

print("\nMatrix shape:")
print(matrix.shape)

print("\nRow shape:")
print(row.shape)

print("\nResult:")
print(result)

#Broadcasting on rows and columns
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

# قيم تضاف إلى كل عمود
row_values = np.array([10, 20, 30])

# قيم تضاف إلى كل صف
column_values = np.array([
    [100],
    [200]
])

# إضافة قيم إلى الأعمدة
result_by_columns = matrix + row_values

# إضافة قيم إلى الصفوف
result_by_rows = matrix + column_values

print("Original matrix:")
print(matrix)

print("\nMatrix shape:")
print(matrix.shape)

print("\nRow values:")
print(row_values)

print("\nRow values shape:")
print(row_values.shape)

print("\nAdd values to each column:")
print(result_by_columns)

print("\nColumn values:")
print(column_values)

print("\nColumn values shape:")
print(column_values.shape)

print("\nAdd values to each row:")
print(result_by_rows)