# Week 1 — Hands-On Lab (Integrated Mini-Notebook)

**BinX Tech · AI & Machine Learning Internship Program · Phase 1: Foundations**
**Week 1 of 10 · All 5 Days Combined**


---

## Overview

`Mini_NoteBook.ipynb` is a **single, self-contained notebook** that consolidates all 5 days of
Week 1 — Environment Setup, Python Fundamentals, NumPy, Pandas, and Matplotlib — into one
continuous, end-to-end learning log. It is meant as a **quick, all-in-one reference**: instead
of jumping between five separate day folders, everything from the whole week can be reviewed,
run, and revisited in a single file, in the exact order it was learned.

The notebook closes with an **integrated Hands-On Lab** that combines NumPy, Pandas, and
Matplotlib together in one pipeline — the same **load → clean → compute → visualize** shape
used throughout the rest of the internship program.

---

## Table of Contents

- [Overview](#-overview)
- [Notebook Structure](#-notebook-structure)
  - [Day 1 — Environment Setup & Jupyter Workflow](#day-1--environment-setup--jupyter-workflow)
  - [Day 2 — Python for Data Science](#day-2--python-for-data-science)
  - [Day 3 — NumPy](#day-3--numpy)
  - [Day 4 — Pandas](#day-4--pandas)
  - [Day 5 — Matplotlib & Week 1 Integrated Lab](#day-5--matplotlib--week-1-integrated-lab)
- [Tools & Technologies](#-tools--technologies)
- [How to Run](#️-how-to-run)
- [Key Skills Gained](#-key-skills-gained)
- [Best Practices & Reproducibility](#-best-practices--reproducibility)
- [Where This Leads Next](#-where-this-leads-next)

---

## Notebook Structure

The notebook is organized into 5 major sections, one per day, each with its own internal table
of contents and section anchors.

### Day 1 — Environment Setup & Jupyter Workflow
- Checking Python & pip installation
- Creating a dedicated project folder
- Creating and activating a virtual environment (`venv`)
- Installing core libraries: `numpy`, `pandas`, `matplotlib`, `jupyter`
- Saving dependencies with `requirements.txt`
- Opening and navigating Jupyter Notebook
- Code cells vs. Markdown cells
- Git basics (init, add, commit)

### Day 2 — Python for Data Science
Four core topics that form the Python foundation for everything after:

| Topic | Covers |
|---|---|
| **Data Types** | `int`, `float`, `str`, `bool`, `list`, `tuple`, `dict`, `set` — with methods & examples for each |
| **Control Flow** | `if`/`elif`/`else`, `for` loops, `while` loops, `break` & `continue` |
| **Functions** | `def`, parameters vs. arguments, default values, `return`, docstrings, multiple return values, keyword arguments |
| **List Comprehensions & OOP** | One-line list/dict/set comprehensions, classes, `__init__`, `self`, methods, multiple objects |

### Day 3 — NumPy
- Why NumPy exists (speed & vectorization vs. plain Python lists)
- Creating arrays: `np.array()`, `np.zeros()`, `np.ones()`, `np.arange()`, `np.linspace()`, `np.random.rand()`
- Array attributes: `.shape`, `.ndim`, `.dtype`, `.size`
- Indexing, slicing & boolean masking
- Vectorized operations vs. loops
- Broadcasting rules and common shape-mismatch errors
- Quick-reference cheat sheet
- Hands-On Lab: array manipulation exercise (reshape, slicing, masking, broadcasting)

### Day 4 — Pandas
- Series vs. DataFrame
- Loading & inspecting data: `.head()`, `.shape`, `.info()`, `.describe()`
- Selecting & filtering: `[]`, `.loc[]`, `.iloc[]`, boolean filtering
- Cleaning data: `.isnull().sum()`, `.fillna()`, `.drop_duplicates()`
- Grouping & aggregation with `.groupby()` — "split → apply → combine"
- Mini project: full **load → inspect → clean → filter → group** pipeline with interpretation

### Day 5 — Matplotlib & Week 1 Integrated Lab
- Why visualization matters (exploration vs. communication)
- The four core plot types: line, scatter, bar, histogram
- Styling: colors, markers, legends, grids, figure size
- Subplots for multi-chart figures
- Saving figures with `plt.savefig()`
- Common mistakes to avoid & a quick-reference cheat sheet
- **Hands-On Lab — Week 1 Integrated Mini-Notebook:**
  1. Load & clean a sample "interns" dataset with Pandas
  2. Compute a derived feature (age z-score) with NumPy
  3. Produce 3+ labeled plots (histogram, bar chart, scatter) plus a combined subplot figure
  4. Interpret each chart in Markdown
  5. Commit the finished work to GitHub

---

## Tools & Technologies

- **Language:** Python 3.13
- **Libraries:** NumPy, Pandas, Matplotlib
- **Environment:** Jupyter Notebook
- **Version Control:** Git & GitHub

---

## How to Run

1. Make sure the required libraries are installed:
   ```bash
   pip install numpy pandas matplotlib jupyter
   ```
2. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Mini_NoteBook.ipynb
   ```
3. Run cells top-to-bottom (`Shift + Enter`) — the notebook is fully self-contained and generates
   its own sample data internally, so no external CSV files are required.
4. Use the in-notebook table of contents at the start of each day's section to jump directly to
   any topic.

---

## Key Skills Gained

- Setting up a reproducible Python environment from scratch
- Core Python: data types, control flow, functions, list comprehensions, and basic OOP
- Fast, vectorized numerical computing with NumPy (indexing, slicing, masking, broadcasting)
- Real-world data handling with Pandas (loading, cleaning, filtering, grouping)
- Building clear, labeled, professional visualizations with Matplotlib
- Assembling all of the above into one integrated **load → clean → compute → visualize** pipeline

---

## Best Practices & Reproducibility

- A random seed is fixed wherever randomness is used (`np.random.seed(42)`), so results are
  reproducible for anyone re-running the notebook.
- Every chart includes an x-axis label, a y-axis label, and a title.
- The plot type is chosen to match the kind of data (trend → line, relationship → scatter,
  categories → bar, distribution → histogram).
- Missing values are filled rather than dropped, to avoid losing real data.
- Short Markdown notes accompany each chart and each cleaning step, explaining *why*, not just
  *how*.

---

## Where This Leads Next

This notebook wraps up **Week 1: Python & Data Science Foundations**. The
**load → clean → compute → visualize** pipeline practiced here is the same shape used throughout
the rest of the 10-week program — starting with **Exploratory Data Analysis (EDA) in Week 2**,
and building up to the program's final **capstone project**.
