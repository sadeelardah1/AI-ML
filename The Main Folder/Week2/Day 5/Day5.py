#Bivariate analysis تحليل العلاقات بين متغيرين في الوقت نفسه 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
data = {
    "Age":[20, 21, 22, 23, 24, 21, 25, 22, 24, 23, 20, 26],
    "Weekly_hours":[15, 18, 20, 22, 25, 17, 30, 21, 27, 24, 16, 32],
    "Monthly_stipend":[350, 400, 450, 500, 580, 390, 700, 470, 630, 550, 370, 750],
    "Track":["AI & ML", "Web Dev", "AI & ML", "Cybersecurity", "Web Dev", "Data Analysis", "AI & ML", "Web Dev", "AI & ML", "Cybersecurity", "Web Dev", "Data Analysis"]
}

df = pd.DataFrame(data)
print("Dataset :")
print(df)
print("\nDataset Shape :")
print(df.shape)
print("\nDataset types :")
print(df.dtypes)



#Scatter plot 
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="Weekly_hours",
    y="Monthly_stipend",
    s=90,
    alpha=0.8
)

plt.title("Weekly Hours VS Monthly Stipend")
plt.xlabel("Weekly Hours")
plt.ylabel("Montly Stipend ($)")
plt.tight_layout()
plt.show()



#using Hue
plt.figure(figsize=(8,5))
sns.scatterplot(
    data=df,
    x="Weekly_hours",
    y="Monthly_stipend",
    hue="Track",
    s=100,
    alpha=0.8
)
plt.title("Weekly Hours VS Monthly Stipend")
plt.xlabel("Weekly Hours")
plt.ylabel("Montly Stipend ($)")
plt.legend(title="Track")
plt.tight_layout()
plt.show()



#Trend line
plt.figure(figsize=(8,5))
sns.regplot(
    data=df,
    x="Weekly_hours",
    y="Monthly_stipend",
    scatter_kws={
       "s": 80,
    "alpha": 0.8 
    },
    line_kws={
        "linestyle": "--"
    }
    
)
plt.title("Weekly Hours VS Monthly Stipend")
plt.xlabel("Weekly Hours")
plt.ylabel("Montly Stipend ($)")
plt.tight_layout()
plt.show()


#Grouped Plot
plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="Track",
    y="Monthly_stipend"
)

plt.title("Monthly Stipend by Track")
plt.xlabel("Track")
plt.ylabel("Monthly Stipend ($)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


#Numerical summary by category

track_summary = (
    df.groupby("Track")["Monthly_stipend"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        minimum="min",
        maximum="max"
    )
    .round(2)
    .sort_values(
        by="median",
        ascending=False
    )
)
print("\n" + "=" * 60)
print("MONTHLY STIPEND SUMMARY BY TRACK")
print("=" * 60)
print(track_summary)



#Correlation
hours_stipend_corr = df["Weekly_hours"].corr(
    df["Monthly_stipend"]
)

print("Correlation:", hours_stipend_corr)
print("Correlation between weekly hours and stipend:",round(hours_stipend_corr, 3))# تقريب النتيجة 

#Interpret the result
if hours_stipend_corr >= 0.7:
    interpretation = "Strong positive correlation"

elif hours_stipend_corr >= 0.3:
    interpretation = "Moderate positive correlation"

elif hours_stipend_corr > -0.3:
    interpretation = "Weak or no linear correlation"

elif hours_stipend_corr > -0.7:
    interpretation = "Moderate negative correlation"

else:
    interpretation = "Strong negative correlation"

print("Interpretation:", interpretation)

#Calculate the full correlation matrix
correlation_matrix = df.corr(
    numeric_only=True
)

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)
print(correlation_matrix.round(2))

#Create the heatmap
plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()



#Pairplot 
data1 = {
    "age": [
        20, 21, 22, 23, 24,
        25, 26, 27, 28, 29,
        30, 31
    ],
    "weekly_hours": [
        10, 15, 16, 17, 18,
        20, 21, 24, 25, 27,
        30, 32
    ],
    "monthly_stipend": [
        580, 350, 750, 390, 400,
        450, 740, 550, 500, 630,
        700, 370
    ],
    "completed_tasks": [
        4, 6, 7, 6, 8,
        9, 10, 11, 12, 13,
        14, 15
    ],
    "track": [
        "AI", "Web", "AI", "Data",
        "Web", "AI", "Data", "AI",
        "Web", "Data", "AI", "Web"
    ]
}

df = pd.DataFrame(data1)

selected_columns = [
    "age",
    "weekly_hours",
    "monthly_stipend",
    "completed_tasks"
]

print("=" * 60)
print("SELECTED NUMERIC DATA")
print("=" * 60)
print(df[selected_columns].describe())

pair_grid = sns.pairplot(
    data=df,
    vars=selected_columns,
    hue="track",
    diag_kind="hist",
    corner=True,
    plot_kws={
        "s": 70,
        "alpha": 0.8
    }
)

pair_grid.fig.suptitle(
    "Pairwise Relationships Between Internship Variables",
    y=1.02
)

plt.show()




#Data Storytelling : EX : Online store
"""
Context
   ↓
Finding
   ↓
Evidence
   ↓
Interpretation
   ↓
Recommendation
 
"""
data2 = {
    "month": [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ],
    "ad_spend": [
        1200, 1500, 1800, 1700,
        2200, 2500, 2700, 3000,
        3200, 3500, 4000, 4500
    ],
    "website_visits": [
        8000, 9200, 10500, 10000,
        12500, 14000, 15000, 16800,
        17500, 19000, 22000, 24500
    ],
    "orders": [
        180, 210, 245, 230,
        290, 330, 350, 390,
        410, 450, 520, 580
    ],
    "revenue": [
        13500, 15800, 18200, 17100,
        22000, 24800, 26500, 29800,
        31500, 34800, 41000, 46500
    ],
    "channel": [
        "Social Media", "Search", "Search", "Social Media",
        "Email", "Search", "Social Media", "Search",
        "Email", "Social Media", "Search", "Email"
    ]
}

df = pd.DataFrame(data2)

print(df)
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print("Shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isna().sum())
print("\nDescriptive statistics:")
print(df.describe())


plt.figure(figsize=(8, 5))
sns.regplot(
    data=df,
    x="ad_spend",
    y="revenue",
    scatter_kws={
        "s": 90,
        "alpha": 0.8
    },
    line_kws={
        "linestyle": "--"
    }
)
plt.title("Advertising Spend vs Revenue")
plt.xlabel("Advertising Spend ($)")
plt.ylabel("Revenue ($)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="website_visits",
    y="orders",
    hue="channel",
    s=100,
    alpha=0.8
)
plt.title("Website Visits vs Orders")
plt.xlabel("Website Visits")
plt.ylabel("Number of Orders")
plt.legend(title="Marketing Channel")
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="website_visits",
    y="orders",
    hue="channel",
    s=100,
    alpha=0.8
)
plt.title("Website Visits vs Orders")
plt.xlabel("Website Visits")
plt.ylabel("Number of Orders")
plt.legend(title="Marketing Channel")
plt.tight_layout()
plt.show()


channel_summary = (
    df.groupby("channel")
    .agg(
        number_of_months=("month", "count"),
        average_ad_spend=("ad_spend", "mean"),
        average_visits=("website_visits", "mean"),
        average_orders=("orders", "mean"),
        average_revenue=("revenue", "mean"),
        total_revenue=("revenue", "sum")
    )
    .round(2)
    .sort_values(
        by="average_revenue",
        ascending=False
    )
)
print(channel_summary)

numeric_columns = [
    "ad_spend",
    "website_visits",
    "orders",
    "revenue"
]

correlation_matrix = df[
    numeric_columns
].corr()
print("Correlation Matrix:")
print(correlation_matrix.round(2))


numeric_columns = [
    "ad_spend",
    "website_visits",
    "orders",
    "revenue"
]

correlation_matrix = df[
    numeric_columns
].corr()
print("Correlation Matrix:")
print(correlation_matrix.round(2))


plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("E-commerce Correlation Heatmap")
plt.tight_layout()
plt.show()


pair_grid = sns.pairplot(
    data=df,
    vars=numeric_columns,
    hue="channel",
    diag_kind="hist",
    corner=True,
    plot_kws={
        "s": 70,
        "alpha": 0.8
    }
)

pair_grid.fig.suptitle(
    "E-commerce Variable Relationships",
    y=1.02
)

plt.show()