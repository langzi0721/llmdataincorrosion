import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm

# 1. Load Dataset
# Note: Ensure the dataset is in the same directory or update the relative path
file_path = 'Old_Descriptors.csv'
model_name = "seyonec/ChemBERTa-zinc-base-v1"
df = pd.read_csv(file_path)

# 2. Extract 768-dim ChemBERTa Embeddings
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> Initializing Model (Device: {device})...")

# Set local_files_only=True if you are running in an offline environment
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

def get_embedding(smiles):
    model.eval()
    inputs = tokenizer(smiles, return_tensors="pt", padding=True,
                       truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract the CLS token (index 0) as the molecular representation
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

print(">>> Extracting molecular semantic vectors (ChemBERTa)...")
tqdm.pandas()
embeddings = np.stack(df['isomeric_smiles'].progress_apply(get_embedding).values)

# 3. Feature Fusion: 5 (Numerical) + 768 (Semantic) = 773 Dimensions
X_raw_cols = ['LUMO / eV', 'CATS3D_02_AP', 'Mor04m', 'E1p', 'P_VSA_MR_5']
X_raw = df[X_raw_cols].values
X_combined = np.hstack([X_raw, embeddings])
y = df['ie_ze41'].values

# 4. Dataset Splitting: Training (60 samples), Testing (15 samples)
X_train, X_test = X_combined[:60], X_combined[60:75]
y_train, y_test = y[:60], y[60:75]

# 5. XGBoost Hyperparameter Tuning
param_grid = {
    'colsample_bytree': [0.5, 0.8],
    'gamma': [0, 0.1, 0.2],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'n_estimators': [50, 100, 200],
    'subsample': [0.7, 0.9]
}

print(f"\nTotal Feature Dimensions: {X_combined.shape[1]}")
print(">>> Starting XGBoost Grid Search...")

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 6. Prediction and Evaluation
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
corr, _ = pearsonr(y_test, y_pred)

# 7. Formatted Results Output
bp = grid_search.best_params_

print("\n" + "="*45)
print("XGBoost Best Hyperparameters:")
for param, value in bp.items():
    print(f"{param:18}: {value}")
print("-" * 45)
print(f"Hybrid Model Performance (Test Set - 15 Samples):")
print(f"R² Score             : {r2:.4f}")
print(f"MAE                  : {mae:.4f}")
print(f"RMSE                 : {rmse:.4f}")
print(f"Pearson Correlation  : {corr:.4f}")
print("="*45)

# 8. Comparison Preview
print("\nActual vs. Predicted Values (Test Set):")
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df.to_string(index=False))