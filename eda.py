import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# CREATE graphs folder
os.makedirs("graphs", exist_ok=True)

# LOAD DATASET
df = pd.read_csv("data/glassdoor_jobs.csv")

print("Dataset Loaded Successfully")

# REMOVE INVALID SALARY
df = df[df['Salary Estimate'] != '-1']

# =====================================
# CLEAN SALARY COLUMN
# =====================================

salary = df['Salary Estimate']

salary = salary.apply(
    lambda x: str(x)
    .replace('$', '')
    .replace('K', '')
    .replace('Employer Provided Salary:', '')
    .replace('Per Hour', '')
    .replace('Employer Provided Salary', '')
)

# REMOVE EXTRA TEXT
salary = salary.apply(
    lambda x: x.split('(')[0]
)

# =====================================
# CREATE MIN/MAX SALARY
# =====================================

df['min_salary'] = salary.apply(
    lambda x: int(x.split('-')[0].strip())
)

df['max_salary'] = salary.apply(
    lambda x: int(x.split('-')[1].strip())
)

# AVERAGE SALARY
df['avg_salary'] = (
    df['min_salary'] + df['max_salary']
) / 2

print("Salary Cleaning Completed")

# =====================================
# 1. SALARY DISTRIBUTION
# =====================================

plt.figure(figsize=(10,6))

sns.histplot(
    df['avg_salary'],
    bins=30,
    kde=True
)

plt.title("Salary Distribution")

plt.xlabel("Average Salary")

plt.ylabel("Count")

plt.tight_layout()

plt.savefig("graphs/salary_distribution.png")

plt.close()

print("Salary Distribution Graph Saved")

# =====================================
# 2. COMPANY SIZE VS SALARY
# =====================================

# REMOVE MISSING SIZE
company_df = df[df['Size'] != '-1']

plt.figure(figsize=(14,6))

sns.barplot(
    x='Size',
    y='avg_salary',
    data=company_df
)

plt.xticks(rotation=90)

plt.title("Company Size vs Average Salary")

plt.xlabel("Company Size")

plt.ylabel("Average Salary")

plt.tight_layout()

plt.savefig("graphs/company_size_salary.png")

plt.close()

print("Company Size Graph Saved")

# =====================================
# 3. LOCATION VS SALARY
# =====================================

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

plt.title("Top 10 Locations by Average Salary")

plt.xlabel("Location")

plt.ylabel("Average Salary")

plt.tight_layout()

plt.savefig("graphs/location_salary.png")

plt.close()

print("Location Graph Saved")

# =====================================
# 4. CORRELATION HEATMAP
# =====================================

numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig("graphs/correlation_heatmap.png")

plt.close()

print("Heatmap Saved")

print("EDA COMPLETED SUCCESSFULLY")