import numpy as np
import pandas as dp
import matplotlib.pyplot as plt
#Prepare -> Create Figure -> Set Size -> Plot -> Label -> Display

days =[1, 2, 3, 4, 5, 6, 7]
sales =[100, 120, 115, 160, 190, 230, 210]
plt.figure(figsize=(10,5))   #plt.figure(figsize=(width, height)) 
plt.plot(days,sales, marker=".") #plot takes (x ,y)   x = horizontal_values, y = vertical_values
plt.title("Sales Over Seven Days")
plt.xlabel("Days")
plt.ylabel("Sales Values")
plt.grid(True)
plt.show()



# Example using numpy + matplotlib
day = ["Sunday", "Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]
study_hour = np.array([2, 2.5, 4, 6, 4.5, 8, 6.5])
plt.figure(figsize=(10,5))
plt.plot(day,study_hour, marker="o")
plt.title("Study Hours During the Week")
plt.xlabel("Day")
plt.ylabel("Study Hours")
plt.grid(True)
plt.show()




dayss = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"
]

study_hours = np.array([2, 2.5, 4, 4.5, 6, 6.5, 8])
practice_hours = np.array([1, 1.5, 2, 3, 3.5, 4, 5])

plt.figure(figsize=(10, 5))

plt.plot(dayss, study_hours, marker="o", markersize=7, label="Study Hours" , linestyle="-", linewidth=2 )
plt.plot(dayss, practice_hours, marker="s" ,label="Study Hours",linestyle="--")

plt.title("Study and Practice Hours During the Week")
plt.xlabel("Day")
plt.ylabel("Hours")
plt.legend() #To Show the Label name on lines
plt.grid(True)

plt.show()


#Scatter polt
study_hours = np.array([1, 2, 3, 4, 5, 6, 7, 8])
exam_scores = np.array([50, 55, 65, 68, 75, 80, 88, 94]) 

plt.figure(figsize=(9, 5))

plt.scatter(study_hours, exam_scores, s=100,  alpha=0.7 ,marker="s") #positive relationship because x increases, y increases , s means the size of points in chart , alpha for Transparency of the points

plt.title("Study Hours vs Exam Scores")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)
plt.show()



#another example
group_a_hours = np.array([1, 2, 3, 4, 5])
group_a_scores = np.array([52, 60, 67, 76, 82])

group_b_hours = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
group_b_scores = np.array([48, 58, 70, 73, 88])

plt.figure(figsize=(9, 5))

plt.scatter(group_a_hours, group_a_scores, marker="o", s=90, label="Group A")

plt.scatter(group_b_hours, group_b_scores, marker="s", s=90, label="Group B")

plt.title("Study Hours vs Exam Scores by Group")
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.legend()
plt.grid(True)
plt.show()


#bar polt
#edgecolor to put color for bar border
departments = ["Sales", "HR", "IT", "Marketing"]
employees = [25, 10, 18, 15]

plt.figure(figsize=(9, 5))

bars = plt.bar(departments, employees, width=0.6, edgecolor="black")

plt.title("Number of Employees by Department")
plt.xlabel("Department")
plt.ylabel("Number of Employees")
plt.grid(axis="y", alpha=0.4)
plt.bar_label(bars)
plt.show()

courses = [
    "Python Fundamentals",
    "Data Analysis with Pandas",
    "Machine Learning Basics",
    "Database Management"
]

students = [40, 32, 25, 28]

plt.figure(figsize=(10, 5))

plt.barh(courses, students, edgecolor="black")
plt.title("Number of Students by Course")
plt.xlabel("Number of Students")
plt.ylabel("Course")
plt.grid(axis="x", alpha=0.4)
plt.show()

#Histogram
exam_scores = np.array([55, 60, 61, 63, 65, 67, 70, 72, 73, 75, 78, 80, 82, 85, 88, 90, 92, 95])
score_bins = [50, 60, 70, 80, 90, 100]

plt.figure(figsize=(9, 5))
plt.hist(exam_scores, bins=score_bins, edgecolor="black", rwidth=0.9, alpha=0.7)
plt.title("Distribution of Exam Scores")
plt.xlabel("Exam Score Range")
plt.ylabel("Number of Students")
plt.xticks(score_bins)
plt.grid(axis="y", alpha=0.4)
plt.show()


#subplots
"""days = np.arange(1, 8)

study_hours = np.array([2, 2.5, 4, 4.5, 6, 6.5, 8])
sleep_hours = np.array([7, 6.5, 7.5, 8, 7, 8.5, 9])

fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # 1, 2 means 1 row and 2 column

axes[0].plot(days, study_hours, marker="o")
axes[0].set_title("Study Hours")
axes[0].set_xlabel("Day")
axes[0].set_ylabel("Hours")
axes[0].grid(True)

axes[1].plot(days, sleep_hours, marker="s")
axes[1].set_title("Sleep Hours")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Hours")
axes[1].grid(True)

plt.tight_layout()
plt.show()"""

days = np.arange(1, 8)

study_hours = np.array([2, 2.5, 4, 4.5, 6, 6.5, 8])
phone_hours = np.array([5, 4.5, 4, 3.5, 4, 3, 2.5])

fig, axes = plt.subplots(2, 1, figsize=(10, 8)) #2, 1 means 2 rows and 1 column

axes[0].plot(days, study_hours, marker="o")
axes[0].set_title("Study Hours")
axes[0].set_xlabel("Day")
axes[0].set_ylabel("Hours")
axes[0].grid(True)

axes[1].plot(days, phone_hours, marker="s")
axes[1].set_title("Phone Usage")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Hours")
axes[1].grid(True)

plt.tight_layout()
plt.show()