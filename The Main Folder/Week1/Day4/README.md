# Day 4 — Pandas 🐼

A comprehensive summary of Day 4 in a Python data analysis learning series, focused on the **Pandas** library — explained step by step with hands-on, self-contained examples.

## Table of Contents

- [Overview](#-overview)
- [What Is Pandas?](#0-what-is-pandas)
- [Series vs. DataFrame](#1-series-vs-dataframe)
- [Loading and Inspecting Data](#2-loading-and-inspecting-data)
- [Selecting and Filtering](#3-selecting-and-filtering)
- [Cleaning Data](#4-cleaning-data)
- [Grouping and Aggregation (groupby)](#5-grouping-and-aggregation-groupby)
- [Mini Project](#6-mini-project--putting-it-all-together)
- [Quick Summary](#7-quick-summary)
- [Key Skills Gained](#-key-skills-gained)
- [How to Run](#️-how-to-run)
- [Requirements](#-requirements)

## Overview

This notebook (`Day4_Pandas.ipynb`) covers the fundamentals of Pandas with practical, ready-to-run examples. It runs entirely on its own, generating its own sample dataset internally — no external CSV file required.

## 0. What Is Pandas?

Pandas is the main Python library for working with tabular (spreadsheet-like) data — think of it as Excel, but inside Python and much more powerful. Its two core objects are:
- **Series**: a single labeled column of data
- **DataFrame**: a full table — rows and named columns, like a CSV, Excel sheet, or SQL query result

## 1. Series vs. DataFrame

A Series is one column with labels (an index). A DataFrame is a collection of Series sharing the same index — basically a table.

## 2. Loading and Inspecting Data

The first step with any dataset is always to load it, then inspect it: shape, columns, data types, and a sample of rows.

Key functions:
- `pd.read_csv("file.csv")` — loads a CSV file into a DataFrame
- `.head()` — shows the first 5 rows
- `.shape` — returns (rows, columns)
- `.info()` — column names, non-null counts, and dtypes
- `.describe()` — summary statistics for numeric columns

## 3. Selecting and Filtering

Columns can be selected by name, and rows by condition (boolean filtering — the Pandas equivalent of NumPy's boolean masking).
- `.loc[]` — select by label (column/row name)
- `.iloc[]` — select by integer position

## 4. Cleaning Data

Real data is messy: missing values, duplicates, and wrong types are the norm. Cleaning is often the largest part of any data project.
- `.isnull().sum()` — counts missing values per column
- `.fillna(value)` — fills missing values (e.g., with the mean)
- `.drop_duplicates()` — removes duplicate rows

> **Note:** Missing values were filled instead of dropped (`df.dropna()`), since dropping rows means losing real data. Filling with the mean is a common, simple strategy for numeric columns.

## 5. Grouping and Aggregation (groupby)

`groupby` splits the data into groups based on a column's values, applies an aggregation (mean, sum, count...) to each group, then combines the results — the **"split → apply → combine"** pattern, and one of the most useful Pandas operations for finding patterns in data.

## 6. Mini Project — Putting It All Together

A small, professional example following the same **load → inspect → clean → filter → group** pipeline used with any real dataset:
1. Report shape, columns, and dtypes
2. Filter to a meaningful subset (e.g., above-average earners)
3. Aggregate statistics per category and interpret the results

**Interpretation example:** Nablus has the highest average income among the cities in the sample, while Jenin has the lowest — exactly the kind of pattern `groupby` is used to reveal.

## 7. Quick Summary

- Series = one labeled column, DataFrame = a full table of Series
- Always inspect data first with `.head()`, `.shape`, `.info()`, `.describe()`
- Select columns with `df["col"]`, filter rows with `df[df["col"] > x]`
- `.loc` selects by label, `.iloc` selects by position
- Clean data with `.isnull().sum()`, `.fillna()`, `.drop_duplicates()`
- Use `.groupby()` to summarize and spot patterns across categories

## Key Skills Gained

- Distinguishing between `Series` and `DataFrame` and understanding their structure
- Quickly inspecting any new dataset before working with it
- Selecting and filtering data in multiple ways (`[]`, `.loc`, `.iloc`)
- Cleaning real-world data (missing values, duplicates)
- Using `groupby` to extract patterns and statistics
- Building a simple end-to-end data analysis pipeline

## How to Run

```bash
pip install pandas numpy jupyter
jupyter notebook Day4_Pandas.ipynb
```

The notebook is fully self-contained — it generates its own sample data internally using `numpy` (with a fixed `seed` for reproducible results).

## Requirements

- Python 3.x
- pandas
- numpy
- jupyter / jupyterlab

## Note

This file is part of a "Days" series for learning data analysis with Python (Day 1, Day 2, ...). This is **Day 4**, dedicated to the Pandas library.
