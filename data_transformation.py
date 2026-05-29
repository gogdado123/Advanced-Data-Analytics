import pandas as pd
import numpy as np 

print("Step 1: Loading Dataset...")
data = pd.read_csv('scooter.csv')

print("\nStep 2: Initial Data Exploration...")
print(f"Columns available: {list(data.columns)}")
print("\n--- Data Types ---")
print(data.dtypes)
print("\n--- Summary Statistics ---")
print(data.describe())

print("\nStep 3: Initial Feature Engineering...")
# 1. Standardize text data to upper case
data['month'] = data['month'].str.upper()

# 2. Split datetime fields into Date and Time components
new_start = data['started_at'].str.split(expand=True)
data['started_date'] = new_start[0]
data['started_time'] = new_start[1]

# 3. Convert objects to correct programmatic Data Types (Programmatic Casting)
data['started_at'] = pd.to_datetime(data['started_at'], format='%m/%d/%Y %H:%M')
data['DURATION'] = pd.to_timedelta(data['DURATION'])
data = data.astype({'trip_id': 'string'})

# 4. Extract basic street data
new_end = data['end_location_name'].str.split(pat=',', n=1, expand=True)
data['end_location_street'] = new_end[0]

print("\nStep 4: Advanced Data Pipeline & Data Transformation...")
# 1. Process end times
new_end_time = data['ended_at'].str.split(expand=True)
data['ended_date'] = new_end_time[0]
data['ended_time'] = new_end_time[1]

# 2. Create a deep copy for data security and safety
data1 = data.copy()
data1['ended_at'] = pd.to_datetime(data1['ended_at'], format='%m/%d/%Y %H:%M')

# 3. Recalculate duration dynamically to fix missing/corrupted entries
data1['DURATION'] = data1['ended_at'] - data1['started_at']

# 4. Extract start street boundaries
new_start_street = data1['start_location_name'].str.split(pat=',', n=1, expand=True)
data1['start_location_street'] = new_start_street[0]

# 5. Handle missing values using Non-destructive Imputation ('Unknown')
cols_loc = ['start_location_street', 'end_location_street']
data1[cols_loc] = data1[cols_loc].fillna('Unknown')

# 6. Extract Zipcodes using Regular Expressions (Regex) for geographical clustering
data1['start_zip_code'] = data1['start_location_name'].str.extract(r'(\d{5})')
data1['end_zip_code'] = data1['end_location_name'].str.extract(r'(\d{5})')

# 7. Feature Selection: Drop redundant and low-variance variables to reduce dimensionality
cols_to_drop = ['region_id', 'start_location_name', 'end_location_name']
# Ensure safety drop regardless of intermediate testing column presence
if 'trip_id_1613335' in data1.columns: cols_to_drop.append('trip_id_1613335')
if 'trip_id_161335' in data1.columns: cols_to_drop.append('trip_id_161335')

data_final = data1.drop(columns=cols_to_drop)

print("\nStep 5: Data Transformation Complete!")
print("--- Final Dataset Preview (First 5 Rows) ---")
print(data_final.head())
