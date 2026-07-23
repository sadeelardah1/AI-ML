import numpy as np
import matplotlib.pyplot as plt

days = np.arange(1, 8)
study_hours = np.array([2, 2.5, 4, 4.5, 6, 6.5, 8])

exam_scores = np.array([55, 60, 65, 70, 75, 85, 92])

languages = ["Python", "Java", "C++", "C#"]
students = np.array([45, 25, 15, 18])

score_distribution = np.array([
    55, 60, 61, 63, 65,
    67, 70, 72, 73, 75,
    78, 80, 82, 85, 88,
    90, 92, 95
])

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

fig.suptitle("Student Data Visualization Dashboard", fontsize=16)

# Line Plot
axes[0, 0].plot(
    days,
    study_hours,
    marker="o"
)

axes[0, 0].set_title("Study Hours During the Week")
axes[0, 0].set_xlabel("Day")
axes[0, 0].set_ylabel("Study Hours")
axes[0, 0].grid(True)

# Scatter Plot
axes[0, 1].scatter(
    study_hours,
    exam_scores,
    s=90,
    alpha=0.7
)

axes[0, 1].set_title("Study Hours vs Exam Scores")
axes[0, 1].set_xlabel("Study Hours")
axes[0, 1].set_ylabel("Exam Score")
axes[0, 1].grid(True)

# Bar Plot
bars = axes[1, 0].bar(
    languages,
    students,
    edgecolor="black"
)

axes[1, 0].set_title("Students by Programming Language")
axes[1, 0].set_xlabel("Programming Language")
axes[1, 0].set_ylabel("Number of Students")
axes[1, 0].grid(axis="y", alpha=0.4)

axes[1, 0].bar_label(bars, padding=3)

# Histogram
axes[1, 1].hist(
    score_distribution,
    bins=[50, 60, 70, 80, 90, 100],
    edgecolor="black",
    alpha=0.7
)

axes[1, 1].set_title("Distribution of Exam Scores")
axes[1, 1].set_xlabel("Exam Score")
axes[1, 1].set_ylabel("Frequency")
axes[1, 1].grid(axis="y", alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()