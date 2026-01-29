import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr

# 1. Load Dataset
# Use relative paths for GitHub to protect local directory privacy
file_path = 'your path'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: {file_path} not found. Please ensure the data file is in the correct directory.")
    exit()

# 2. Variable Preparation
# Define features and target based on domain analysis
X_cols = ['LUMO / eV', 'CATS3D_02_AP', 'Mor04m', 'E1p', 'P_VSA_MR_5']
y_col = 'ie_ze41'

# Split Dataset: Training set (first 60) and Test set (remaining 15)
X_train = df.loc[:59, X_cols]
y_train = df.loc[:59, y_col]
X_test = df.loc[60:74, X_cols]
y_test = df.loc[60:74, y_col]

# 3. XGBoost Hyperparameter Optimization (Grid Search)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9],
    'gamma': [0, 0.1, 0.2]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Perform Grid Search with 5-fold Cross-Validation
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                           cv=5, scoring='r2', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

# Retrieve the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# 4. Model Prediction
y_pred = best_model.predict(X_test)

# 5. Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
corr, _ = pearsonr(y_test, y_pred)

# 6. Output Results to Console
print("\n" + "="*40)
print("XGBoost Best Hyperparameters:")
for param, value in best_params.items():
    print(f"{param:18}: {value}")

print("\n" + "="*40)
print("Performance Metrics (Test Set):")
print(f"R² Score             : {r2:.4f}")
print(f"MAE                  : {mae:.4f}")
print(f"RMSE                 : {rmse:.4f}")
print(f"Pearson Correlation  : {corr:.4f}")
print("="*40)

# 7. Comparison Table (Direct Display)
print("\nActual vs. Predicted Values:")
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.to_string(index=False))