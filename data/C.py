import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold
from scipy.stats import pearsonr
from tqdm import tqdm
import os
import random

# 1. Global Reproducibility Setup
def seed_everything(seed=42):
    """
    Locks all random seeds to ensure the results are reproducible
    across different runs.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# 2. Data and Model Loading
# IMPORTANT: Update 'data_path' to your local CSV file location.
# Example: 'C:/Users/Name/Project/data.csv' or just 'data.csv' if in the same folder.
data_path = 'Old_Descriptors.csv'
model_name = "seyonec/ChemBERTa-zinc-base-v1"

if not os.path.exists(data_path):
    print(f"Error: Dataset not found at {data_path}. Please check the file path.")
    exit()

df = pd.read_csv(data_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f">>> Initializing {model_name} on {device}...")

# local_files_only: Set to True if you have already downloaded the model
# to a local directory and want to avoid checking for online updates.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)

def get_embedding(smiles):
    model.eval()
    inputs = tokenizer(smiles, return_tensors="pt", padding=True,
                       truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # Extract CLS token as the semantic vector
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

print(">>> Extracting molecular embeddings (ChemBERTa)...")
tqdm.pandas()
embeddings = np.stack(df['isomeric_smiles'].progress_apply(get_embedding).values)

# 3. PCA Dimensionality Reduction
# Normalizing data before PCA is crucial for ChemBERTa embeddings.
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)

# Retaining 90% of the cumulative variance to balance info density and model complexity.
pca = PCA(n_components=0.90, random_state=42)
embeddings_reduced = pca.fit_transform(embeddings_scaled)
print(f"PCA Logic: Retained {embeddings_reduced.shape[1]} components for 90% variance.")

# 4. Feature Fusion and Splitting
X_raw_cols = ['LUMO / eV', 'CATS3D_02_AP', 'Mor04m', 'E1p', 'P_VSA_MR_5']
X_raw = df[X_raw_cols].values
X_combined = np.hstack([X_raw, embeddings_reduced])
y = df['ie_ze41'].values

# Strict Split: First 60 samples for Training, Last 15 for Independent Testing
X_train, X_test = X_combined[:60], X_combined[60:75]
y_train, y_test = y[:60], y[60:75]

# 5. Grid Search with 3-Fold Cross-Validation
# reg_lambda (L2 Regularization) is prioritized to handle small-sample variance.
param_grid = {
    'colsample_bytree': [0.4, 0.5],
    'gamma': [0.1, 0.2, 0.3],
    'learning_rate': [0.08, 0.1, 0.12],
    'max_depth': [3, 4, 5],
    'n_estimators': [250, 300, 350],
    'subsample': [0.7, 0.8, 0.9],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1.0, 5.0, 10.0, 20.0]
}

print(f"\nTotal Feature Dimensions: {X_combined.shape[1]}")
print(">>> Starting 3-Fold Cross-Validation...")

cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
    param_grid=param_grid,
    cv=cv_strategy,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# 6. Evaluation
best_model = grid_search.best_estimator_
y_final_pred = best_model.predict(X_test)

print("\n" + "="*45)
print("XGBoost Best Hyperparameters:")
for k, v in grid_search.best_params_.items():
    print(f"{k:18}: {v}")
print(f"Mean CV R² (Train): {grid_search.best_score_:.4f}")
print("-" * 45)
print(f"Independent Test Set Performance (N=15):")
print(f"R² Score             : {r2_score(y_test, y_final_pred):.4f}")
print(f"MAE                  : {mean_absolute_error(y_test, y_final_pred):.4f}")
print(f"RMSE                 : {np.sqrt(mean_squared_error(y_test, y_final_pred)):.4f}")
print(f"Pearson Correlation  : {pearsonr(y_test, y_final_pred)[0]:.4f}")
print("="*45)

# Comparison Preview
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_final_pred})
print("\nActual vs. Predicted Preview:")
print(comparison_df.head(15).to_string(index=False))