# Day 5 — Matplotlib & the Week 1 Mini-Notebook

**BinX Tech · AI & Machine Learning Internship Program · Phase 1: Foundations**
**Week 1 of 10 · Day 5 of 5 · 8 hours**

---

## Table of Contents

1. [Overview](#overview)
2. [Learning Objectives](#learning-objectives)
3. [Key Topics](#key-topics)
4. [Files in This Folder](#files-in-this-folder)
5. [How to Use `Day5.ipynb`](#how-to-use-day5ipynb)
6. [Lesson Summary](#lesson-summary)
   - [1. Why Visualization Matters](#1-why-visualization-matters)
   - [2. The Matplotlib Basics](#2-the-matplotlib-basics)
   - [3. The Four Core Plot Types](#3-the-four-core-plot-types)
   - [4. Styling a Plot](#4-styling-a-plot)
   - [5. Subplots](#5-subplots)
   - [6. Saving a Figure](#6-saving-a-figure)
7. [Hands-On Lab — Week 1 Integrated Mini-Notebook](#hands-on-lab--week-1-integrated-mini-notebook)
8. [Deliverables Checklist](#deliverables-checklist)
9. [Tools Used](#tools-used)
10. [Best Practices & Reproducibility](#best-practices--reproducibility)
11. [Where This Leads Next](#where-this-leads-next)

---

## Overview

Day 5 is the final day of **Week 1: Python & Data Science Foundations**. It covers **Matplotlib**,
the core Python plotting library, and closes the week by combining everything learned across
Days 1–5 — **Python fundamentals, NumPy, and Pandas** — into a single, integrated
**load → process → visualize** mini-notebook. This exact pipeline shape is the same one every
future project in this 400-hour program will follow.

## Learning Objectives

By the end of Day 5, you should be able to:

- Create labeled **line**, **scatter**, **bar**, and **histogram** plots.
- Choose the *correct* plot type for a given kind of data.
- Style a chart (colors, markers, legend, grid) so it looks professional, not like a rough draft.
- Combine multiple plots into one figure using **subplots**.
- Save a finished chart to an image file.
- Assemble a complete mini-notebook using **NumPy + Pandas + Matplotlib** together.

## Key Topics

- Why visualization matters for exploration *and* communication
- Matplotlib basics: `plot`, `xlabel`, `ylabel`, `title`, `show`
- The four core plot types: line, scatter, bar, histogram
- Styling: colors, markers, legends, grids, figure size
- Subplots for side-by-side comparison
- Saving figures with `plt.savefig()`
- Assembling a load → process → visualize notebook

## Files in This Folder

| File | Description |
|---|---|
| `Day5.ipynb` | The full, detailed Day 5 lesson notebook — explanations, examples, a cheat sheet, and the Hands-On Lab, fully executed with outputs. |
| `README.md` | This file — a plain-language overview of the whole day, for anyone browsing the repository. |
| `age_distribution.png` | Example image exported by the notebook using `plt.savefig()` (generated when you run Section 9). |

## How to Use `Day5.ipynb`

1. Open the notebook in **Jupyter Notebook**, **VS Code**, or **Google Colab**.
2. Read top-to-bottom — every section starts with a short **Goal** or **Note** box explaining
   *why* the topic matters before showing *how* to code it.
3. Run each code cell in order (`Shift + Enter`). All cells are already executed once, so you can
   also just read the saved outputs without re-running anything.
4. Use the **Table of Contents** at the top of the notebook to jump straight to any section.
5. The **Cheat Sheet** (Section 11) is a fast reference you can come back to later in the program.

## Lesson Summary

### 1. Why Visualization Matters
Numbers in a table hide patterns — trends, clusters, outliers, relationships — that a chart
reveals instantly. Visualization has two jobs: **exploration** (charts for yourself, while
investigating data) and **communication** (charts for others, which must always be clearly
labeled).

### 2. The Matplotlib Basics
Every chart follows the same four-step recipe:
```python
plt.plot(x, y)        # 1. draw the chart
plt.xlabel("...")     # 2. label the x-axis
plt.ylabel("...")     # 3. label the y-axis
plt.title("...")      # 4. title the chart
plt.show()             # 5. display it
```

### 3. The Four Core Plot Types

| Plot | Function | Use It To Show |
|---|---|---|
| Line | `plt.plot()` | A trend or change over a continuous axis (e.g. time) |
| Scatter | `plt.scatter()` | The relationship between two numeric variables |
| Bar | `plt.bar()` | Comparison of a value across categories |
| Histogram | `plt.hist()` | The distribution (shape/spread) of a single variable |

### 4. Styling a Plot
Small touches — `color`, `marker`, `linewidth`, `plt.legend()`, `plt.grid(True, alpha=0.3)`,
`plt.figure(figsize=(w, h))` — turn a rough chart into a professional, report-ready one.

### 5. Subplots
`plt.subplots(rows, cols)` creates a grid of chart areas (`axes`) so several charts can sit
side by side in one figure. Inside a subplot, use `axes[i].set_title()` /
`axes[i].set_xlabel()` instead of the plain `plt.title()` style.

### 6. Saving a Figure
```python
plt.savefig("chart.png", dpi=150, bbox_inches="tight")   # call BEFORE plt.show()
plt.show()
```

## Hands-On Lab — Week 1 Integrated Mini-Notebook

The lab builds a small sample "interns" dataset (with missing values on purpose) and runs the
full pipeline:

1. **Load & clean with Pandas** — build/load the dataset, count and fill missing values.
2. **Compute with NumPy** — calculate a derived feature (an age z-score) using vectorized math.
3. **Visualize with Matplotlib** — produce at least three labeled plots:
   - A **histogram** of intern ages (distribution)
   - A **bar chart** of interns per track (category comparison, via `groupby`)
   - A **scatter plot** of age vs. weekly hours (relationship between two variables)
   - A **bonus subplot figure** combining all three side by side
4. **Interpret the results** in Markdown — what each chart actually reveals about the data.
5. **Commit to GitHub** — `pip freeze > requirements.txt`, then `git add`, `git commit`,
   `git push`.

## Deliverables Checklist

By the end of Week 1, submit the following to your mentor and GitHub repository:

- [ ] A reproducible Python environment with a committed `requirements.txt`
- [ ] A Python fundamentals notebook (functions, list comprehensions, a small class) with Markdown documentation
- [ ] A NumPy notebook demonstrating array creation, slicing, boolean masking, and broadcasting
- [ ] A Pandas notebook loading, cleaning, filtering, and aggregating a real dataset
- [ ] **The Week 1 integrated mini-notebook (NumPy + Pandas + Matplotlib) with at least three labeled plots** ← this is `Day5.ipynb`
- [ ] All Week 1 notebooks committed to the intern's GitHub repository with clear commit messages

## Tools Used

Matplotlib • Pandas • NumPy • Jupyter Notebook • Git & GitHub

## Best Practices & Reproducibility

- Always fix a random seed wherever randomness is used: `np.random.seed(42)`.
- Every chart should always have an **x-axis label**, a **y-axis label**, and a **title** — an
  unlabeled chart is not a finding, it's a puzzle.
- Choose the plot type that matches the *kind* of data (trend → line, relationship → scatter,
  categories → bar, distribution → histogram), not just whichever chart looks nicest.
- Add a short Markdown note near every chart explaining what it shows, in plain words.

## Where This Leads Next

Day 5 closes **Week 1: Python & Data Science Foundations**. The load → process → visualize
pipeline built today is the same shape used throughout the rest of the program — starting with
**Exploratory Data Analysis (EDA) in Week 2**, and continuing all the way to the **Phase 3
capstone project**.



