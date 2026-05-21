import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("data/glassdoor_jobs.csv")

print("Dataset Loaded Successfully")

# =========================
# REMOVE INVALID SALARY
# =========================

df = df[df['Salary Estimate'] != '-1']

# =========================
# CLEAN SALARY COLUMN
# =========================

salary = df['Salary Estimate'].apply(
    lambda x: x.replace('$', '')
              .replace('K', '')
              .replace('Employer Provided Salary:', '')
              .replace('Per Hour', '')
)

# =========================
# CREATE SALARY COLUMNS
# =========================

df['min_salary'] = salary.apply(
    lambda x: int(x.split('-')[0])
)

df['max_salary'] = salary.apply(
    lambda x: int(x.split('-')[1].split('(')[0])
)

df['avg_salary'] = (
    df['min_salary'] + df['max_salary']
) / 2

print("Salary Cleaning Completed")

# =========================
# FEATURE ENGINEERING
# =========================

skills = {
    'python_skill': 'python',
    'sql_skill': 'sql',
    'aws_skill': 'aws',
    'excel_skill': 'excel',
    'tableau_skill': 'tableau',
    'powerbi_skill': 'power bi',
    'ml_skill': 'machine learning',
    'dl_skill': 'deep learning',
    'tensorflow_skill': 'tensorflow',
    'spark_skill': 'spark',
    'hadoop_skill': 'hadoop',
    'statistics_skill': 'statistics'
}

for column, keyword in skills.items():

    df[column] = df['Job Description'].apply(
        lambda x: 1 if keyword in str(x).lower() else 0
    )

print("Feature Engineering Completed")

# =========================
# FEATURES
# =========================

features = list(skills.keys())

X = df[features]

y = df['avg_salary']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# MODEL
# =========================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed")

# =========================
# EVALUATION
# =========================

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

print("MAE:", mae)

print("R2 Score:", r2)

# =========================
# SAVE MODEL
# =========================

pickle.dump(model, open('models/model.pkl', 'wb'))

print("Model Saved Successfully")