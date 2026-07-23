# Python Fundamentals — Day 2

This repository contains the completed **Day 2 Python Fundamentals** work for the **BinX Tech AI & Machine Learning Internship Program**.

The notebooks cover the core Python concepts required for data science, including data types, control flow, functions, list comprehensions, and object-oriented programming basics.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Learning Objectives](#learning-objectives)
- [Repository Structure](#repository-structure)
- [Notebook Details](#notebook-details)
- [Requirements](#requirements)
- [Installation and Setup](#installation-and-setup)
- [How to Run the Notebooks](#how-to-run-the-notebooks)
- [Topics Covered](#topics-covered)
- [Hands-On Tasks Completed](#hands-on-tasks-completed)
- [Code Quality and Documentation](#code-quality-and-documentation)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)
- [Git Workflow](#git-workflow)
- [Author](#author)

---

## Project Overview

Day 2 focuses on building a strong foundation in Python before moving to NumPy, Pandas, data visualization, and machine learning.

The project is organized into four Jupyter notebooks. Each notebook explains the concepts using Markdown cells and demonstrates them using executable Python examples.

The notebooks are designed to be read from top to bottom like a short technical report:

1. A Markdown cell explains the concept.
2. A code cell demonstrates the concept.
3. The output confirms the result.

---

## Learning Objectives

After completing these notebooks, the learner should be able to:

- Understand and use Python's core data types.
- Store and organize data using lists, tuples, dictionaries, and sets.
- Write conditional logic using `if`, `elif`, and `else`.
- Repeat operations using `for` and `while` loops.
- Control loop execution using `break` and `continue`.
- Create reusable functions with parameters and return values.
- Write clear function documentation using docstrings.
- Use default arguments and keyword arguments.
- Create lists efficiently using list comprehensions.
- Define a basic Python class.
- Work with class attributes, objects, and methods.
- Write clean and readable Python code inside Jupyter Notebook.

---

## Repository Structure

```text
Day-2-Python-Fundamentals/
│
├── DataTypes_Modified.ipynb
├── ControlFlow_Modified.ipynb
├── Function_Modified.ipynb
├── ListComprehension&OOP_Modified.ipynb
├── README.md
└── requirements.txt
```

> The `requirements.txt` file may be generated from the active environment using `pip freeze > requirements.txt`.

---

## Notebook Details

### 1. `DataTypes.ipynb`

This notebook introduces Python's main built-in data types.

Topics include:

- Integers: `int`
- Decimal numbers: `float`
- Text values: `str`
- Boolean values: `bool`
- Lists: `list`
- Tuples: `tuple`
- Dictionaries: `dict`
- Sets: `set`
- Type inspection using `type()`
- Basic operations on each data type
- Mutable and immutable data structures

This notebook provides the foundation needed to store and manipulate data in later data science tasks.

---

### 2. `ControlFlow.ipynb`

This notebook demonstrates how Python controls the execution of a program.

Topics include:

- `if` statements
- `if` and `else`
- `if`, `elif`, and `else`
- Comparison operators
- Logical operators
- Nested conditions
- `for` loops
- `while` loops
- The `range()` function
- `break`
- `continue`
- Loop-based data processing

The notebook includes a non-interactive password example so that all cells can run without stopping for user input.

---

### 3. `Function.ipynb`

This notebook explains how to organize reusable logic using functions.

Topics include:

- Defining a function with `def`
- Calling functions
- Function parameters
- Return values
- Multiple parameters
- Default arguments
- Keyword arguments
- Local variables
- Docstrings
- Returning structured results using dictionaries

A key exercise implements a function that receives a list of numbers and returns the mean, minimum, and maximum values in a dictionary.

Example:

```python
def get_stats(numbers):
    """
    Return the mean, minimum, and maximum of a list of numbers.
    """
    return {
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers)
    }
```

---

### 4. `ListComprehension&OOP.ipynb`

This notebook covers two important Python concepts: list comprehensions and object-oriented programming.

#### List Comprehensions

Topics include:

- Creating lists in one readable line
- Applying an expression to every value
- Filtering values using a condition
- Replacing simple loops with list comprehensions
- Creating squares and filtered lists

Example:

```python
even_numbers = [number for number in range(20) if number % 2 == 0]
```

#### Object-Oriented Programming

Topics include:

- Classes and objects
- The `__init__` constructor
- Instance attributes
- Instance methods
- Creating class instances
- Calling methods
- Representing a simple data record as an object

Example:

```python
class DataRecord:
    """Represent a simple data record."""

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def describe(self):
        return f"{self.name}: {self.value}"
```

---

## Requirements

Recommended software:

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- Git
- Visual Studio Code with the Python and Jupyter extensions, optional

The Day 2 notebooks mainly use Python's standard library, so no external data science package is required for the examples.

A minimal `requirements.txt` may contain:

```text
jupyter
notebook
ipykernel
```

To record the exact packages installed in the current environment, run:

```bash
pip freeze > requirements.txt
```

---

## Installation and Setup

### Option 1: Using `venv`

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate it on Windows Command Prompt:

```cmd
.venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install jupyter notebook ipykernel
```

Generate the dependency file:

```bash
pip freeze > requirements.txt
```

---

### Option 2: Using Conda

Create a Conda environment:

```bash
conda create --name day2-python python=3.10
```

Activate it:

```bash
conda activate day2-python
```

Install Jupyter:

```bash
conda install jupyter notebook ipykernel
```

Export the environment if needed:

```bash
conda env export > environment.yml
```

---

## How to Run the Notebooks

### Using Jupyter Notebook

Run:

```bash
jupyter notebook
```

Then:

1. Open the project folder.
2. Select one of the `.ipynb` files.
3. Choose **Kernel → Restart & Run All**.
4. Confirm that all cells execute without errors.
5. Save the notebook after execution.

### Using JupyterLab

Run:

```bash
jupyter lab
```

### Using Visual Studio Code

1. Open the repository folder in VS Code.
2. Install the Python and Jupyter extensions.
3. Open a notebook.
4. Select the correct Python interpreter or kernel.
5. Click **Run All**.

---

## Topics Covered

| Category | Concepts |
|---|---|
| Data Types | `int`, `float`, `str`, `bool`, `list`, `tuple`, `dict`, `set` |
| Conditions | `if`, `elif`, `else`, comparison operators, logical operators |
| Loops | `for`, `while`, `range`, `break`, `continue` |
| Functions | Parameters, arguments, return values, docstrings, default arguments |
| List Comprehensions | Transformation, filtering, concise list creation |
| OOP Basics | Classes, objects, constructors, attributes, methods |
| Documentation | Markdown explanations, comments, docstrings |
| Workflow | Jupyter Notebook execution and Git version control |

---

## Hands-On Tasks Completed

The following Day 2 practical requirements are included:

- A function that accepts a list of numbers.
- Calculation of the mean, minimum, and maximum.
- Returning the statistics in a dictionary.
- Rewriting an even-number filtering loop as a list comprehension.
- Defining a small class with at least two attributes.
- Adding at least one method to the class.
- Adding Markdown documentation before the important code sections.
- Executing notebook cells and displaying their outputs.

---

## Code Quality and Documentation

The notebooks follow these practices:

- Clear and descriptive variable names
- Small examples focused on one concept at a time
- Markdown explanations before code sections
- Function docstrings
- Class docstrings where appropriate
- Consistent formatting
- Visible outputs
- No duplicated notebook sections
- No required interactive input during **Run All**

---

## Expected Results

When all notebooks are executed successfully:

- Every code cell should run without errors.
- Outputs should appear directly below their code cells.
- The statistics function should return a dictionary.
- The list comprehension examples should produce the expected filtered or transformed lists.
- Class objects should be created successfully.
- Class methods should return or print meaningful information.

---

## Troubleshooting

### Jupyter is not recognized

Install it using:

```bash
pip install jupyter notebook
```

Then try:

```bash
python -m notebook
```

### The wrong Python kernel is selected

Install an IPython kernel:

```bash
python -m ipykernel install --user --name day2-python --display-name "Python (Day 2)"
```

Then select **Python (Day 2)** from the notebook kernel menu.

### PowerShell blocks environment activation

Run PowerShell as administrator and execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.venv\Scripts\Activate.ps1
```

### Notebook cells run in the wrong order

Use:

1. **Kernel → Restart Kernel**
2. **Run → Run All Cells**

This ensures the notebook is reproducible from top to bottom.

---

## Git Workflow

Initialize the repository:

```bash
git init
```

Add the project files:

```bash
git add .
```

Create a clear commit:

```bash
git commit -m "Complete Day 2 Python fundamentals notebooks"
```

Connect the local repository to GitHub:

```bash
git remote add origin YOUR_REPOSITORY_URL
```

Push the project:

```bash
git branch -M main
git push -u origin main
```

Suggested future commit messages:

```text
Add Python data types examples
Complete control flow exercises
Add reusable functions and statistics task
Complete list comprehensions and OOP basics
Improve notebook Markdown documentation
Update Day 2 README
```

---

## Completion Status

- [x] Core Python data types
- [x] Conditional statements
- [x] `for` loops
- [x] `while` loops
- [x] Functions with parameters
- [x] Return values
- [x] Function docstrings
- [x] Statistics dictionary exercise
- [x] List comprehensions
- [x] Basic class implementation
- [x] Attributes and methods
- [x] Markdown documentation
- [x] Clean notebook structure
- [x] Reproducible notebook execution

---

## Acknowledgment

This project was completed as part of the BinX Tech AI & Machine Learning Internship Program. It demonstrates the Python fundamentals required for upcoming work with NumPy, Pandas, Matplotlib, and machine learning libraries.