import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
total_outcomes = [1, 2, 3, 4, 5, 6]
favorable_outcomes = [
    value
    for value in total_outcomes 
    if value % 2 == 0
]

probability = len(favorable_outcomes) / len(total_outcomes )

print("Sample space:", total_outcomes )
print("Even outcomes:", favorable_outcomes)
print("Probability of even number:", probability)
print("Probability of odd number:", 1-probability)


#Additon Rule 
sample_space = {1, 2, 3, 4, 5, 6}

event_a = {2, 4, 6}
event_b = {4, 5, 6}

intersection = event_a & event_b
union = event_a | event_b

probability_a = len(event_a) / len(sample_space)
probability_b = len(event_b) / len(sample_space)
probability_intersection = len(intersection) / len(sample_space)

probability_union = (
    probability_a
    + probability_b
    - probability_intersection
)

print("A:", event_a)
print("B:", event_b)
print("A ∩ B:", intersection)
print("A ∪ B:", union)
print("P(A or B):", probability_union)

#Multiplicstion
probability_heads = 1 / 2
probability_six = 1 / 6

probability_heads_and_six = (
    probability_heads * probability_six
)

print("P(Heads):", probability_heads)
print("P(6):", probability_six)
print("P(Heads and 6):", probability_heads_and_six)
print(f"Percentage: {probability_heads_and_six:.2%}")



#Conditional Probability
# Load the CSV file
students = pd.read_csv(r"C:\Users\sadee\OneDrive\Desktop\The Main Folder\The Main Folder\Week2\Day 2\students_probability_example.csv")

# Display the complete dataset
print("Students Dataset:")
print(students)

# Select only students in the AI track
ai_students = students[students["track"] == "AI"]

# Select AI students who passed
ai_students_passed = ai_students[ai_students["passed"] == "Yes"]

# Count the required observations
number_of_ai_students = len(ai_students)
number_of_ai_students_passed = len(ai_students_passed)

# Calculate conditional probability
probability_passed_given_ai = (
    number_of_ai_students_passed / number_of_ai_students
)

print("\nAI Students:")
print(ai_students)

print("\nAI Students Who Passed:")
print(ai_students_passed)

print("\nNumber of AI students:", number_of_ai_students)
print("Number of AI students who passed:", number_of_ai_students_passed)

print(
    "P(Passed | AI):",
    probability_passed_given_ai
)

print(
    f"P(Passed | AI) as percentage: "
    f"{probability_passed_given_ai:.2%}"
)







#Bayes Theotem
# Load the dataset
people = pd.read_csv(r"C:\Users\sadee\OneDrive\Desktop\The Main Folder\The Main Folder\Week2\Day 2\bayes_medical_test_100_people.csv")
print("Dataset:")
print(people)

total_people = len(people)
diseased_people = people[
    people["disease_status"] == "Disease"
]

healthy_people = people[
    people["disease_status"] == "No Disease"
]

print("\nTotal people:", total_people)
print("People with disease:", len(diseased_people))
print("People without disease:", len(healthy_people))

positive_results = people[
    people["test_result"] == "Positive"
]

print("\nPeople with positive results:")
print(positive_results)
print("\nTotal positive results:", len(positive_results))


diseased_and_positive = people[
    (people["disease_status"] == "Disease")
    & (people["test_result"] == "Positive")
]

print("\nDiseased people with positive results:")
print(diseased_and_positive)

print(
    "\nDiseased and positive:",
    len(diseased_and_positive)
)


probability_disease_given_positive = (
    len(diseased_and_positive)
    / len(positive_results)
)

print(
    "\nP(Disease | Positive):",
    probability_disease_given_positive
)

print(
    f"P(Disease | Positive): "
    f"{probability_disease_given_positive:.2%}"
)


#Common Distributions
#Uniform Distribution
np.random.seed(42)
number_of_rolls = 10000

dice_rolls = np.random.randint(
    low=1,
    high=7,
    size=number_of_rolls
)

print("First 20 dice rolls:")
print(dice_rolls[:20])

dice_series = pd.Series(
    dice_rolls,
    name="dice_result"
)


frequency = (
    dice_series
    .value_counts()
    .sort_index()
)

print("\nFrequency of each outcome:")
print(frequency)


experimental_probability = (
    dice_series
    .value_counts(normalize=True)
    .sort_index()
)

print("\nExperimental probabilities:")
print(experimental_probability)

theoretical_probability = 1 / 6

print(
    "\nTheoretical probability for each outcome:",
    theoretical_probability
)

comparison = pd.DataFrame({
    "Frequency": frequency,
    "Experimental Probability": experimental_probability,
    "Theoretical Probability": theoretical_probability
})

comparison["Difference"] = (
    comparison["Experimental Probability"]
    - comparison["Theoretical Probability"]
)

print("\nComparison table:")
print(comparison)

experimental_probability.plot(
    kind="bar"
)

plt.axhline(
    y=theoretical_probability,
    linestyle="--",
    label="Theoretical Probability"
)

plt.title("Uniform Distribution: Dice Roll Simulation")
plt.xlabel("Dice Outcome")
plt.ylabel("Probability")
plt.legend()
plt.tight_layout()
plt.show()


#Binomial Distribution
number_of_flips = 10
probability_of_heads = 0.5
number_of_experiments = 10000

heads_counts = np.random.binomial(
    n=number_of_flips,
    p=probability_of_heads,
    size=number_of_experiments
)


heads_series = pd.Series(
    heads_counts,
    name="number_of_heads"
)

frequency = (
    heads_series
    .value_counts()
    .sort_index()
)

experimental_probabilities = (
    heads_series
    .value_counts(normalize=True)
    .sort_index()
)

print("First 20 experiments:")
print(heads_counts[:20])

print("\nFrequency:")
print(frequency)

print("\nExperimental probabilities:")
print(experimental_probabilities)

print("\nExpected number of Heads:")
print(number_of_flips * probability_of_heads)

experimental_probabilities.plot(
    kind="bar"
)

plt.title("Binomial Distribution: Heads in 10 Coin Flips")
plt.xlabel("Number of Heads")
plt.ylabel("Probability")
plt.tight_layout()
plt.show()


#Normal Distribution

population_mean = 70
population_std = 10
sample_size = 10000


scores = np.random.normal(
    loc=population_mean,
    scale=population_std,
    size=sample_size
)

scores_series = pd.Series(
    scores,
    name="score"
)


sample_mean = scores_series.mean()
sample_median = scores_series.median()
sample_std = scores_series.std(ddof=0)

print("Target mean:", population_mean)
print("Sample mean:", round(sample_mean, 2))
print("Sample median:", round(sample_median, 2))

print("\nTarget standard deviation:", population_std)
print(
    "Sample standard deviation:",
    round(sample_std, 2)
)

print("\nSummary statistics:")
print(scores_series.describe())

plt.figure(figsize=(9, 5))

plt.hist(
    scores_series,
    bins=40,
    edgecolor="black"
)

plt.axvline(
    sample_mean,
    linestyle="--",
    label=f"Mean = {sample_mean:.2f}"
)

plt.axvline(
    sample_median,
    linestyle=":",
    label=f"Median = {sample_median:.2f}"
)

plt.title("Simulated Normal Distribution of Scores")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()