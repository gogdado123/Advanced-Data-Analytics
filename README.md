# Advanced Data Analytics & Predictive Modeling Pipeline

A comprehensive data science repository showcasing end-to-end data preprocessing, programmatic feature engineering, and classical machine learning implementation (Supervised & Unsupervised Learning).

## 🚀 Key Frameworks & Libraries
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & Analytics:** Scikit-Learn (sklearn)
* **Visualizations:** Matplotlib

## 📂 Repository Architecture & Contents

### 1. `data_preprocessing_pipeline.py` (Data Engineering Phase)
An automated multi-stage pipeline designed to ingest raw messy records and output programmatic analytical feature matrices.
* **Feature Engineering:** Implemented modular text parsing, automated date-time character splitting, and programmatic data type casting.
* **Geographical Parsing:** Utilized Regular Expressions (Regex) to extract structural postal zip codes out of free-form textual location fields.
* **Missing Value Imputation:** Handled missing data via data-preserving categorical constants (`'Unknown'`) and multi-type median/mode imputation loop strategies.
* **Categorical Encoding & Scaling:** Configured Scikit-Learn's `OneHotEncoder` for nominal fields, custom stratified `OrdinalEncoder` for sequential features, and `MinMaxScaler` / `StandardScaler` for numeric bound normalization.

### 2. `machine_learning_modeling.py` (AI & Predictive Analytics Phase)
Implementation of structural statistical modeling, target variable forecasting, and multi-dimensional anomaly isolation.
* **Hyperparameter Optimization Loop:** Built a discrete parameter testing grid for K-Nearest Neighbors (kNN) Regression to locate optimal $K$-neighbors, effectively minimizing empirical Mean Squared Error (MSE).
* **Statistical Fitness Evaluation:** Tracked performance metrics via Coefficient of Determination ($R^2$ Score) to secure predictive validity.
* **Density-Based Outlier Detection:** Applied `DBSCAN` clustering logic to flag isolated, low-density feature objects as structural noise instances (-1 mapping).
* **Ensemble Anomaly Isolation:** Deployed non-parametric `Isolation Forest` models to identify deep spatial partition anomalies using structural tree-depth scoring mechanisms.

## 🛠️ Execution & Testing
Every script in this repository includes independent **Mock Data Generation engines**. This means all algorithms run entirely out-of-the-box (Self-contained unit testing) without needing external `.csv` dependencies.

To execute the scripts locally, install dependencies and run:
```bash
pip install pandas numpy scikit-learn matplotlib
python data_preprocessing_pipeline.py
python machine_learning_modeling.py
```
