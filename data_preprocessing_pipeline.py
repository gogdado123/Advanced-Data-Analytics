"""
Data Preprocessing & Transformation Pipeline
-------------------------------------------
This script combines data cleaning, structured transformations, character splitting,
missing value handling, categorical encoding, and feature scaling techniques
originally developed across multiple lab workflows (Lab 03, Lab 04, Lab 06).

Targeted for deployment ready / GitHub portfolio showcase.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler

# ==========================================
# 1. DATA TRANSFORMATION & FEATURE ENGINEERING (Lab 03)
# ==========================================
print("[INFO] Phase 1: Initiating Data Transformation & Feature Engineering...")

# Simulating high-quality data generation resembling scooter.csv for modular testing
np.random.seed(42)
sample_size = 100
scooter_data = pd.DataFrame({
    'started_at': pd.date_range(start='2026-01-01 08:00', periods=sample_size, freq='h'),
    'ended_at': pd.date_range(start='2026-01-01 08:30', periods=sample_size, freq='h'),
    'start_location_name': [f"Street {i}, Bangkok 10{i:02d}" for i in range(sample_size)],
    'end_location_name': [f"Street {i+1}, Bangkok 10{i+1:02d}" if i % 10 != 0 else np.nan for i in range(sample_size)],
    'region_id': np.random.randint(1, 5, size=sample_size),
    'trip_id_161335': np.random.randint(100000, 999999, size=sample_size)
})
# Introduce dummy missing values for verification
scooter_data.loc[5:10, 'start_location_name'] = np.nan

# Copying to work on the dataset
df_scooter = scooter_data.copy()

# Ensure dates are strings for demonstration of conversion flow
df_scooter['ended_at'] = df_scooter['ended_at'].astype(str)

# Split strings into distinct date and time components
time_split = df_scooter['ended_at'].str.split(expand=True)
df_scooter['ended_date'] = time_split[0]
df_scooter['ended_time'] = time_split[1]

# Convert date columns back to datetime objects to perform arithmetic calculations
df_scooter['started_at'] = pd.to_datetime(df_scooter['started_at'])
df_scooter['ended_at'] = pd.to_datetime(df_scooter['ended_at'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

# Calculate duration metrics
df_scooter['DURATION'] = df_scooter['ended_at'] - df_scooter['started_at']

# Text Processing: Extract location names and clean strings
loc_split = df_scooter['start_location_name'].str.split(pat=',', n=1, expand=True)
df_scooter['start_location_street'] = loc_split[0]
df_scooter['end_location_street'] = df_scooter['end_location_name'].str.split(pat=',', n=1, expand=True)[0]

# Impute missing textual fields with 'Unknown' instead of dropping rows wholesale
cols_loc = ['start_location_street', 'end_location_street']
df_scooter[cols_loc] = df_scooter[cols_loc].fillna('Unknown')

# Regex extraction: Isolate numerical zip codes from free text fields
df_scooter['start_zip_code'] = df_scooter['start_location_name'].str.extract(r'(\d{5})')
df_scooter['end_zip_code'] = df_scooter['end_location_name'].str.extract(r'(\d{5})')

# Drop redundant, unneeded, or highly sparse identifier columns to reduce complexity
cols_to_drop = ['region_id', 'start_location_name', 'end_location_name', 'trip_id_161335']
df_transformed = df_scooter.drop(columns=cols_to_drop)

print(f"[SUCCESS] Phase 1 Completed. Formatted Shape: {df_transformed.shape}")


# ==========================================
# 2. CATEGORICAL ENCODING & FEATURE SCALING (Lab 04)
# ==========================================
print("\n[INFO] Phase 2: Running Categorical Encoding & Feature Scaling...")

# Simulating adult dataset
adult_df = pd.DataFrame({
    'workclass': np.random.choice(['Private', 'Local-gov', 'Self-emp-not-inc'], size=sample_size),
    'education': np.random.choice(['Bachelors', 'Masters', 'HS-grad'], size=sample_size),
    'hours.per.week': np.random.randint(10, 60, size=sample_size),
    'capital.gain': np.random.exponential(scale=2000, size=sample_size),
    'capital.loss': np.random.exponential(scale=200, size=sample_size)
})

# Drop rows where critical values are missing if needed (demonstrating lab procedure)
adult_df.dropna(subset=["workclass", "education"], inplace=True)
adult_df.reset_index(inplace=True, drop=True)

# 2.1 One-Hot Encoding via Scikit-Learn OneHotEncoder (Aligned with Resume Profile)
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded_workclass = ohe.fit_transform(adult_df[['workclass']])
df_workclass_ohe = pd.DataFrame(encoded_workclass, columns=ohe.get_feature_names_out(['workclass']))

# 2.2 Feature Scaling: Min-Max Normalization using MinMaxScaler
scaler_minmax = MinMaxScaler()
adult_df['normalized_hours.per.week'] = scaler_minmax.fit_transform(adult_df[['hours.per.week']])

# 2.3 Feature Scaling: Standardization (Z-score Scaling) using StandardScaler
scaler_std = StandardScaler()
adult_df['standardized_capital.gain'] = scaler_std.fit_transform(adult_df[['capital.gain']])
adult_df['standardized_capital.loss'] = scaler_std.fit_transform(adult_df[['capital.loss']])

# Combine processed numerical properties and encoded categories
adult_encoded = pd.concat([adult_df, df_workclass_ohe], axis=1)
adult_encoded.drop(columns=["capital.loss", "capital.gain", "hours.per.week", "workclass"], inplace=True)

print(f"[SUCCESS] Phase 2 Completed. Encoded Shape: {adult_encoded.shape}")


# ==========================================
# 3. MASS MISSING DATA IMPUTATION PIPELINE (Lab 06)
# ==========================================
print("\n[INFO] Phase 3: Executing Automated Missing Data Imputation...")

# Simulating high-dimensional dataset with various missing values (like NFL dataset)
nfl_mock = pd.DataFrame({
    'Metric_A': [1.0, 2.5, np.nan, 4.2, 5.1] * 20,
    'Metric_B': [np.nan, np.nan, 12.0, 15.5, 19.1] * 20,
    'PlayType': ['Pass', 'Run', np.nan, 'Pass', 'Pass'] * 20,
    'Team': [np.nan, 'NE', 'BUF', 'NYJ', np.nan] * 20
})

# Isolate columns by semantic data type automatically
numeric_cols = nfl_mock.select_dtypes(include=['number']).columns
categorical_cols = nfl_mock.select_dtypes(include=['object']).columns

# Standardized Imputation Loop for Continuous Numeric Columns
for col in numeric_cols:
    median_val = nfl_mock[col].median()
    if pd.isna(median_val): 
        nfl_mock[col] = nfl_mock[col].fillna(0)
    else:
        nfl_mock[col] = nfl_mock[col].fillna(median_val)

# Standardized Imputation Loop for Categorical String Columns
for col in categorical_cols:
    mode_val = nfl_mock[col].mode()
    if not mode_val.empty:
        nfl_mock[col] = nfl_mock[col].fillna(mode_val[0])
    else:
        nfl_mock[col] = nfl_mock[col].fillna('NA')

# Verify total remaining null references across the entire dataframe
remaining_nulls = nfl_mock.isnull().sum().sum()
print(f"[SUCCESS] Phase 3 Completed. Remaining Missing Values: {remaining_nulls}")
print("\n[COMPLETE] Preprocessing pipeline executed successfully without any errors.")
