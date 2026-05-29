"""
Machine Learning Modeling & Outlier Detection Portfolio
-------------------------------------------------------
This script incorporates classical supervised and unsupervised learning algorithms
including kNN Regression, optimization loop for Hyperparameter Tuning, DBSCAN, 
and Isolation Forest from multiple research exercises (s1-exa2, s1-exb1, Lab 07).

Targeted for deployment ready / GitHub portfolio showcase.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. SUPERVISED LEARNING & HYPERPARAMETER TUNING (kNN Regression)
# ==========================================
print("[INFO] Phase 1: Initiating Supervised Learning (kNN Regression)...")

# Simulating clean experimental dataset (similar to Advertising.csv)
np.random.seed(42)
n_samples = 200
tv_budget = np.random.uniform(10, 300, n_samples)
sales = 5 + 0.05 * tv_budget + np.random.normal(0, 2, n_samples)

df_adv = pd.DataFrame({'TV': tv_budget, 'sales': sales})

# Select response variable and predictor variable
x = df_adv[['TV']]
y = df_adv['sales']

# Partition variables into controlled training and testing validation splits
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Generate a wide series of discrete 'k' candidates for neighbor count evaluation
k_list = np.linspace(1, 70, 30, dtype=int)
knn_dict = {}

# Hyperparameter search loop to minimize generalization error
for k_value in k_list:
    model = KNeighborsRegressor(n_neighbors=int(k_value))
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    # Store Validation Mean Squared Error (MSE) metrics per candidate parameter
    mse = mean_squared_error(y_test, y_pred)
    knn_dict[k_value] = mse

# Extract optimal configuration parameter via lowest empirical loss
min_mse = min(knn_dict.values())
best_k = [k for k, mse in knn_dict.items() if mse == min_mse][0]
print(f"[SUCCESS] Hyperparameter Tuning Done. Best k: {best_k} with Test MSE: {min_mse:.4f}")

# Train the best final candidate model instance
best_model = KNeighborsRegressor(n_neighbors=best_k)
best_model.fit(x_train, y_train)
y_pred_best = best_model.predict(x_test)

# Calculate goodness of fit score metric
r2 = r2_score(y_test, y_pred_best)
print(f"[METRIC] Final Optimized Model R2 Score: {r2:.4f}")


# ==========================================
# 2. UNSUPERVISED LEARNING: OUTLIER DETECTION (Lab 07)
# ==========================================
print("\n[INFO] Phase 2: Running Unsupervised Outlier Detection Algorithms...")

# Simulating human biometrics dataset (similar to weight-height.csv)
biometrics_df = pd.DataFrame({
    'Gender': np.random.choice(['Male', 'Female'], size=200),
    'Height': np.random.normal(66, 3, 200),
    'Weight': np.random.normal(160, 30, 200)
})

# Inject manual anomalies to test outlier detection capabilities
biometrics_df.loc[10, 'Weight'] = 350.0
biometrics_df.loc[50, 'Height'] = 40.0

# 2.1 Classical Statistical Outlier Detection: Z-Score Thresholding
threshold_z = 3
biometrics_df['weight_zscore'] = (biometrics_df['Weight'] - biometrics_df['Weight'].mean()) / biometrics_df['Weight'].std()
statistical_outliers = biometrics_df[np.abs(biometrics_df['weight_zscore']) > threshold_z]
print(f"[OUTLIERS] Z-Score Method detected {len(statistical_outliers)} structural height/weight anomalies.")

# 2.2 Density-Based Clustering Outlier Detection: DBSCAN
# Fit model to structural numerical elements
dbscan_model = DBSCAN(eps=5, min_samples=5)
cluster_assignments = dbscan_model.fit_predict(biometrics_df[['Weight']])

# Instances tagged with an index of -1 are flagged as isolation density noise
dbscan_outliers = biometrics_df[cluster_assignments == -1]
print(f"[OUTLIERS] DBSCAN (eps=5) flagged {len(dbscan_outliers)} density-isolated objects as noise.")

# 2.3 Non-Parametric Ensemble Anomaly Detection: Isolation Forest
iforest = IsolationForest(n_estimators=100, random_state=42)
iforest.fit(biometrics_df[['Height']])

# Score anomalous profiles via tree depth paths across feature space partitions
anomaly_scores = iforest.score_samples(biometrics_df[['Height']])

# Set analytical score drop threshold
score_threshold = -0.65
iforest_outliers = biometrics_df[anomaly_scores < score_threshold]
print(f"[OUTLIERS] Isolation Forest flagged {len(iforest_outliers)} spatial partitions below score threshold {score_threshold}.")

print("\n[COMPLETE] Machine Learning Modeling script completed successfully.")
