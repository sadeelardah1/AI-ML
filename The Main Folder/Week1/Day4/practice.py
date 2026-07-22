
# Series and DataFrame
print("\n")
print("Series and DataFrame")
import numpy as np
import pandas as pd

employees = pd.DataFrame({
    "Name":["Sadeel", "Ali", "Mohamad", "Assel"],
    "Departments":["Sales", "Technology", "Accounting", "Sales"],
    "Salary":[4500, 7500, 6230, 2800]
})
print(employees)
print("Salary column:\n",employees.iloc[: ,2]) # it is like employees["Salary"] 
print("Salary column:\n",employees["Salary"])
print("Shape :",employees.shape)
print("index :",employees.index)
print("-------------------------------------------------------")
#Loading and Inspecting Data
print("\n")
print("Loading and Inspecting Data")
rf = pd.read_csv(r"C:\Users\sadee\OneDrive\Desktop\The Main Folder\The Main Folder\Week1\Day4\dirty_cafe_sales.csv")
print(rf.shape)
print(rf.head(10))
print("Information :",rf.info())
print("Describe :",rf.describe())
print("-------------------------------------------------------")

#Selecting and Filtering
#to print only series
print("\n")
print("series")
print("Series :",rf["Item"])
#Boolean Filtering
print("\n")
print("Boolean Filtering")
print(rf["Quantity"].unique())#to find the data that make errors , to fix this errors convert the dataType of Quantity column into numaric values by using to_numeric()
rf["Quantity"]=pd.to_numeric(rf["Quantity"],errors="coerce") #The dtype changed from str to float64 - actually means real numbers, regular numbers ('2', '4'...) Converted to actual numbers (2.0, 4.0...) 'Error' and 'UNKNOWN' - since they are not numbers at all, pandas as much as converts them, check them NaN (thanks to error="coerce" — means "if at some value you estimate its transformation, lower it NaN instead of giving error and stop")
print("Quantity >3 :",rf[rf["Quantity"] > 3])
print("Items With Quantity >3 :",rf.loc[rf["Quantity"] > 3,"Item"])

#cleaning Data
print("\n")
print("Cleaning Data")
cd =pd.read_csv(r"C:\Users\sadee\OneDrive\Desktop\The Main Folder\The Main Folder\Week1\Day4\train_and_test2.csv")
print(cd.shape)
print(cd.columns)
print(cd.info())
print(cd.isnull().sum())
cd["Embarked"] = cd["Embarked"].fillna(cd["Embarked"].mode()[0])
print(cd.isnull().sum())
print(cd.duplicated().sum())
#print(cd["zero"].unique())
#print(cd.nunique()) #By translating the number of unique values in each column. If a column has only one repeated value, the result is 1.
print(cd.nunique()[cd.nunique() == 1])
cols_to_drop = cd.nunique()[cd.nunique() == 1].index
cd = cd.drop(columns=cols_to_drop)
print(cd.shape)
print(cd.columns)

print("\n")
print("Grouping By :")
print(cd.groupby("Pclass")["Fare"].mean())
cd = cd.rename(columns={"2urvived": "Survived"})
print(cd.groupby("Survived")["Age"].mean())
print(cd.groupby(["Survived", "Sex"])["Age"].mean())
print(cd["Sex"].unique())
print(cd.groupby("Sex")["Survived"].mean())