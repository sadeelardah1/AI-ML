# Week 1 — Python & Data Science Foundations

**BinX Tech · AI & Machine Learning Internship Program · Phase 1: Foundations**
**Week 1 of 10 · 5 Days · 40 Hours**

---

## Overview

This repository documents **Week 1** of my AI & Machine Learning internship journey — the
**Foundations** phase. Over 5 days, it covers everything needed to go from zero to a working,
professional Python + Data Science toolkit: environment setup, core Python, NumPy, Pandas, and
Matplotlib — closing with an integrated **load → process → visualize** mini-project that ties
every topic together.

Each day lives in its own folder with a dedicated notebook and its own `README.md` for a deeper,
day-by-day breakdown.

---

## Table of Contents

- [Overview](#-overview)
- [Repository Structure](#-repository-structure)
- [Week 1 Roadmap](#-week-1-roadmap)
  - [Day 1 — Environment Setup & Jupyter Workflow](#day-1--environment-setup--jupyter-workflow)
  - [Day 2 — Python Fundamentals](#day-2--python-fundamentals)
  - [Day 3 — NumPy](#day-3--numpy)
  - [Day 4 — Pandas](#day-4--pandas)
  - [Day 5 — Matplotlib & Week 1 Mini-Notebook](#day-5--matplotlib--week-1-mini-notebook)
- [Tools & Technologies](#-tools--technologies)
- [How to Run This Repository](#️-how-to-run-this-repository)
- [Deliverables Checklist](#-deliverables-checklist)
- [Key Takeaways](#-key-takeaways)
- [Where This Leads Next](#-where-this-leads-next)

---

## Repository Structure

```
WEEK1/
├── Day1/
│   ├── Day1_EnviernmentSetup.ipynb
│   ├── Day1_Summery.ipynb
│   ├── requiremnents.txt
│   └── README.md
├── Day2/
│   ├── DataTypes.ipynb
│   ├── ControlFlow.ipynb
│   ├── Function.ipynb
│   ├── ListComprehension&OOP.ipynb
│   └── README.md
├── Day3/
│   ├── Day3.ipynb
│   ├── Day3.py
│   └── Task.py
├── Day4/
│   ├── Day4_Pandas.ipynb
│   ├── Task.ipynb
│   ├── practice.py
│   ├── dirty_cafe_sales.csv
│   ├── train_and_test2.csv
│   └── README.md
├── Day5/
│   ├── Day5.ipynb
│   ├── Day5 practice.py
│   ├── FullEX.py
│   └── README.md
├── Mini NoteBook/
│   └── Mini NoteBook.ipynb
└── README.md   ← you are here
```

---

## Week 1 Roadmap

### Day 1 — Environment Setup & Jupyter Workflow
[`Day1/`](./Day1)

Set up a clean, reproducible Python environment and learned the Jupyter Notebook workflow.

- Verified Python & pip installation
- Created and activated a virtual environment (`.venv`)
- Installed core libraries: `numpy`, `pandas`, `matplotlib`, `jupyter`
- Saved dependencies to `requirements.txt`
- Initialized Git & committed the first project to GitHub

### Day 2 — Python Fundamentals
[`Day2/`](./Day2)

Core Python building blocks needed before touching any data library.

| Notebook | Topic |
|---|---|
| `DataTypes.ipynb` | Numeric, text, sequence, mapping, set, boolean & binary data types |
| `ControlFlow.ipynb` | `if`/`elif`/`else`, `for` and `while` loops |
| `Function.ipynb` | Defining and calling functions, parameters, return values |
| `ListComprehension&OOP.ipynb` | List comprehensions & Object-Oriented Programming basics (classes, objects) |

### Day 3 — NumPy
[`Day3/`](./Day3)

Introduction to **NumPy**, the foundation of numerical computing in Python.

- Why NumPy exists — fast, vectorized array operations
- Creating arrays, indexing & slicing
- Boolean masking and broadcasting
- `Day3.py` / `Task.py` — standalone practice scripts applying the concepts

### Day 4 — Pandas 🐼
[`Day4/`](./Day4)

Deep dive into **Pandas**, the core library for tabular data analysis.

- Series vs. DataFrame
- Loading & inspecting data (`read_csv`, `.head()`, `.info()`, `.describe()`)
- Selecting & filtering (`.loc`, `.iloc`, boolean filtering)
- Cleaning data (missing values, duplicates)
- Grouping & aggregation with `.groupby()`
- Mini project applying the full **load → inspect → clean → filter → group** pipeline
- Real datasets included: `dirty_cafe_sales.csv`, `train_and_test2.csv`

### Day 5 — Matplotlib & Week 1 Mini-Notebook
[`Day5/`](./Day5)

Closed the week with **Matplotlib** and an integrated mini-project.

- Why visualization matters (exploration vs. communication)
- The four core plot types: line, scatter, bar, histogram
- Styling plots (colors, markers, legends, grids) & subplots
- Saving figures with `plt.savefig()`
- **Hands-On Lab:** an integrated `NumPy + Pandas + Matplotlib` mini-notebook following
  **load → process → visualize**, combining everything learned across the whole week

Also see [`Mini NoteBook/`](./Mini%20NoteBook) — an extra consolidated notebook wrapping up
Week 1 concepts in one place.

---

## Tools & Technologies

- **Language:** Python 3.13
- **Libraries:** NumPy, Pandas, Matplotlib
- **Environment:** Jupyter Notebook, virtual environments (`.venv`)
- **Version Control:** Git & GitHub

---

## How to Run This Repository

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd WEEK1
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```
3. Install the dependencies:
   ```bash
   pip install -r Day1/requiremnents.txt
   ```
4. Launch Jupyter and open any day's notebook:
   ```bash
   jupyter notebook
   ```
5. For a deeper explanation of any single day, check that day's own `README.md`.

---

## Deliverables Checklist

- [x] Reproducible Python environment with a committed `requirements.txt`
- [x] Python fundamentals notebooks (data types, control flow, functions, list comprehensions & OOP)
- [x] NumPy notebook — array creation, slicing, boolean masking, broadcasting
- [x] Pandas notebook — loading, cleaning, filtering, and aggregating real datasets
- [x] Matplotlib notebook with at least 3 labeled plots
- [x] Week 1 integrated mini-notebook (NumPy + Pandas + Matplotlib)
- [x] All notebooks committed to GitHub with clear structure and documentation

---

## Key Takeaways

- A reproducible environment (`venv` + `requirements.txt`) is the foundation of any real project
- Python's core building blocks (data types, control flow, functions, OOP) are prerequisites for
  any data science work
- NumPy provides fast, vectorized array operations that Pandas builds on top of
- Pandas turns messy, real-world data into clean, analyzable tables
- Matplotlib turns tables of numbers into insights people can actually see
- Every real data project follows the same shape: **load → clean/process → analyze → visualize**

---

## Where This Leads Next

Week 1 closes **Phase 1: Foundations**. The **load → process → visualize** pipeline built this
week is the same shape used throughout the rest of the 10-week program — starting with
**Exploratory Data Analysis (EDA) in Week 2**, and building all the way up to the program's
final **capstone project**.
