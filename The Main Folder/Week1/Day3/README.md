# NumPy

A practical Jupyter Notebook covering the fundamental concepts of **NumPy** and numerical computing in Python.

---

## Table of Contents

- [Overview](#overview)
- [Topics Covered](#topics-covered)
- [Hands-On Lab](#hands-on-lab)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Key Takeaways](#key-takeaways)

---

## Overview

This notebook introduces NumPy arrays and demonstrates how to manipulate numerical data efficiently using indexing, slicing, vectorized operations, Boolean masking, and broadcasting.

---

## Topics Covered

- Introduction to NumPy
- Array creation
- Array attributes
  - `shape`
  - `ndim`
  - `size`
  - `dtype`
- Indexing and slicing
- Vectorized operations
- Boolean masking
- Broadcasting

---

## Hands-On Lab

The notebook includes a practical exercise that covers:

- Creating a `4 × 4` array containing values from `1` to `16`
- Printing the array shape and data type
- Extracting the second column
- Extracting the last row
- Selecting values greater than the array mean
- Adding a 1D row array to every row using broadcasting
- Verifying the broadcasting result manually

Example:

```python
import numpy as np

array_2d = np.arange(1, 17).reshape(4, 4)

second_column = array_2d[:, 1]
last_row = array_2d[-1, :]

values_above_mean = array_2d[array_2d > array_2d.mean()]

row_array = np.array([10, 20, 30, 40])
broadcasting_result = array_2d + row_array
```

---

## Technologies Used

- Python
- NumPy
- Jupyter Notebook
- Git
- GitHub

---

## How to Run

Clone the repository:

```bash
git clone <repository-url>
```

Navigate to the project folder:

```bash
cd NumPy
```

Install the required libraries:

```bash
pip install numpy jupyter
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Then open:

```text
Day3_NumPy_Deep_Dive.ipynb
```

---

## Key Takeaways

- NumPy arrays provide efficient numerical computation.
- Vectorized operations reduce the need for Python loops.
- Boolean masking makes data filtering simple.
- Broadcasting allows operations between arrays with compatible shapes.
- Understanding array shape is essential when working with multidimensional data.
