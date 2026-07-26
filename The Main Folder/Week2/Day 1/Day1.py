import numpy as np
import pandas as pd
import math
grades = np.array([70, 75, 80, 85, 90])
print(grades)
print("The Number of observations : ",len(grades))
print("The minimun grade : ",grades.min())
print("The maximum grade : ",grades.max())

grades_series = pd.Series([70, 75, 80, 85, 90], name="Grade")
print(grades_series.describe())

#Mean
study_hours = [2, 4, 3, 5, 6, 4, 4]
total = sum(study_hours)
count = len(study_hours)
mean = total / count

print("Total:", total)
print("Count:", count)
print("Mean:", mean)

# Deviation 
sweets =pd.Series([3, 7, 12], name="Sweets")
print(sweets)
print("Mean of sweet : ", sweets.mean())
print("Deviation : ", sweets-(sweets.mean()))

# Median يتم حسابه على العناصر المرتبة فقط 
#Median لما يكون عدد العناصر فردي
numbers = np.array([7, 2, 9, 5, 4])
sorted_numbers = np.sort(numbers)
print(sorted_numbers)
print("Median : ", np.median(sorted_numbers))

#Median لما يكون عدد العناصر زوجي 
numbers1 = np.array([7, 2, 9, 5, 4, 3 ,1, 6]) 
sorted_numbers1=np.sort(numbers1)
print(sorted_numbers1)
median = np.median(sorted_numbers1)
print("Median : ", median)


#Mode تظهر العناصر التي تكررت اكثر من مرة ، اذا ظهرت العناصر مرة واحدة فقط اذاً لا يوجد mode 
Colors = pd.Series(["Red", "White", "Black", "White", "Green", "Green", "Blue", "Red"], name="Color")# Series.mode() ترجع كل العناصر التي تشترك في أعلى تكرار، وليس كل العناصر التي لها نفس التكرار عمومًا.
print(Colors)
#Mode = Colors.mode()

print("Mode : ", Colors.mode())

#1.3 Measures of Spread
#1.3.1Range 
grades1 = pd.Series([70, 75, 80, 85, 90])

data_range = grades1.max() - grades1.min()
print("Range:", data_range)

"""1.3.2 Variance :
1. نحسب Mean
2. نطرح Mean من كل قيمة
3. نربّع كل فرق
4. نجمع مربعات الفروق
5. نقسم
"""
data = [2, 4, 6]
mean = sum(data) / len(data)
squared_deviations = []

for value in data:
    deviation = value - mean
    squared_deviation = deviation ** 2
    squared_deviations.append(squared_deviation)

population_variance = sum(squared_deviations) / len(data)

print("Mean:", mean)
print("Squared deviations:", squared_deviations)
print("Population variance:", population_variance)

#OR
"""
data = pd.Series([2, 4, 6])
variance = data.var()
print("Variance:", variance)
"""

#1.3.3 Standard deviation


data1 = [2, 4, 6]
mean1 = sum(data1) / len(data1)
squared_deviations1 = []

for value in data1:
    deviation1 = value - mean1
    squared_deviation1 = deviation1 ** 2
    squared_deviations1.append(squared_deviation1)

population_variance1 = sum(squared_deviations1) / len(data1)
population_std1 = math.sqrt(population_variance1)

print("Data:", data1)
print("Mean:", mean1)
print("Squared deviations:", squared_deviations1)
print("Population variance:", population_variance1)
print("Population standard deviation:", population_std1)