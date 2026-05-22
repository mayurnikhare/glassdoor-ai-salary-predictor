import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# =========================================
# CREATE GRAPHS FOLDER
# =========================================

os.makedirs("graphs", exist_ok=True)

# =========================================
# LOAD DATASET
# =========================================

df = pd.read_csv("data/glassdoor_jobs.csv")

print("Dataset Loaded Successfully")

# =========================================
# REMOVE INVALID SALARY
# =========================================

df = df[df['Salary Estimate'] != '-1']

# =========================================
# CLEAN SALARY COLUMN
# =========================================

salary = df['Salary Estimate']

salary = salary.apply(
    lambda x: str(x)
    .replace('$', '')
    .replace('K', '')
    .replace('Employer Provided Salary:', '')
    .replace('Employer Provided Salary', '')
    .replace('Per Hour', '')
)

salary = salary.apply(
    lambda x: x.split('(')[0]
)

# =========================================
# CREATE MIN/MAX/AVG SALARY
# =========================================

df['min_salary'] = salary.apply(
    lambda x: int(x.split('-')[0].strip())
)

df['max_salary'] = salary.apply(
    lambda x: int(x.split('-')[1].strip())
)

df['avg_salary'] = (
    df['min_salary'] + df['max_salary']
) / 2

print("Salary Cleaning Completed")

# =========================================
# GRAPH STYLE
# =========================================

sns.set_style("darkgrid")

# =========================================
# 1. SALARY DISTRIBUTION
# =========================================

plt.figure(figsize=(10,6))

sns.histplot(
    df['avg_salary'],
    bins=30,
    kde=True,
    color='blue'
)

plt.title("Salary Distribution", fontsize=16)

plt.xlabel("Average Salary")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("graphs/salary_distribution.png")

plt.show()

# =========================================
# 2. COMPANY SIZE VS SALARY
# =========================================

company_df = df[df['Size'] != '-1']

plt.figure(figsize=(14,6))

sns.barplot(
    x='Size',
    y='avg_salary',
    data=company_df
)

plt.xticks(rotation=90)

plt.title(
    "Company Size vs Average Salary",
    fontsize=16
)

plt.xlabel("Company Size")

plt.ylabel("Average Salary")

plt.tight_layout()

plt.savefig("graphs/company_size_salary.png")

plt.show()

# =========================================
# 3. LOCATION VS SALARY
# =========================================

location_df = df.groupby('Location')['avg_salary'] \
                .mean() \
                .sort_values(ascending=False) \
                .head(10)

plt.figure(figsize=(12,6))

sns.barplot(
    x=location_df.index,
    y=location_df.values
)

plt.xticks(rotation=45)

plt.title(
    "Top 10 Locations by Average Salary",
    fontsize=16
)

plt.xlabel("Location")

plt.ylabel("Average Salary")

plt.tight_layout()

plt.savefig("graphs/location_salary.png")

plt.show()

# =========================================
# 4. CORRELATION HEATMAP
# =========================================

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    cmap='coolwarm',
    annot=True
)

plt.title("Correlation Heatmap", fontsize=16)

plt.tight_layout()

plt.savefig("graphs/correlation_heatmap.png")

plt.show()

# =========================================
# DONE
# =========================================

print("EDA COMPLETED SUCCESSFULLY")