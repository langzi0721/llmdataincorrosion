import os
import time
import pandas as pd
import numpy as np
import random
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import xgboost as xgb

# ================= 1. Global Reproducibility =================
def seed_everything(seed=42):
    """Ensure reproducibility across API calls and model training."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42)

# ================= 2. Configuration & API Setup =================
# Replace these with your actual filenames
INPUT_FILE = 'input_dataset.csv'
OUTPUT_EMBEDDING_FILE = 'molecular_embeddings.csv'
PREDICTION_FILE = 'prediction_results.csv'

# SECURITY: Use environment variables to protect your API Key
# In terminal: export M_API_KEY='your-real-key'
API_KEY = os.getenv("M_API_KEY", "your_api_key_placeholder")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_embedding(text, model="text-embedding-v4", dim=1024):
    """
    Real-time API call for High-Dimensional Semantic Embedding.
    Transforms SMILES/Descriptions into 1024-dim vectors.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Clean input text (remove newlines)
            clean_text = str(text).replace("\n", " ")
            resp = client.embeddings.create(model=model, input=[clean_text], dimensions=dim)
            return resp.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2) # Backoff before retry
                continue
            return [0.0] * dim

# ================= 3. Feature Engineering (API Extraction) =================
print(">>> [Step 1] Initializing LLM Feature Engineering (API calls)...")
df = pd.read_csv(INPUT_FILE)

# Column indices (Adjusted for generic use)
# Assuming: index 2 is target, index 8 is SMILES/Text, plus specific descriptors
y = df.iloc[:, 2].values
X_numeric_raw = df[['CATS3D_02_AP', 'Mor04m']].values # Traditional Descriptors
prompt_col = df.iloc[:, 8]

print(">>> Calling LLM API for molecular semantic embedding...")
embeddings = [get_embedding(text) for text in prompt_col]
X_emb_raw = np.array(embeddings)

# Backup raw embeddings for traceability
emb_cols = [f'emb_{i}' for i in range(X_emb_raw.shape[1])]
df_emb_save = pd.DataFrame(X_emb_raw, columns=emb_cols)
pd.concat([df, df_emb_save], axis=1).to_csv(OUTPUT_EMBEDDING_FILE, index=False)

# ================= 4. Dimensionality Reduction & Fusion =================
print("\n>>> [Step 2] Feature Fusion (Hybrid Semantic + Numerical)...")

# Scale and reduce API Embeddings (1024 -> PCA-dim)
scaler_emb = StandardScaler()
X_emb_scaled = scaler_emb.fit_transform(X_emb_raw)
pca = PCA(n_components=0.90, random_state=42)
X_emb_pca = pca.fit_transform(X_emb_scaled)

# Scale Traditional Descriptors
scaler_num = StandardScaler()
X_num_scaled = scaler_num.fit_transform(X_numeric_raw)

# Final Feature Fusion
X_final = np.hstack((X_num_scaled, X_emb_pca))

# Split: Train (first 60), Test (last 15 - Blind Test)
X_train, y_train = X_final[:60], y[:60]
X_test, y_test = X_final[-15:], y[-15:]

# ================= 5. Hyperparameter Optimization =================
print(f"\n>>> [Step 3] Running Randomized Search (Total Features: {X_final.shape[1]})...")

param_distributions = {
    'n_estimators': [100, 300, 500, 800],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=500,
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=0,
    n_jobs=4,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_

# ================= 6. Comprehensive Evaluation =================
y_test_pred = best_model.predict(X_test)
n_pca_dim = X_emb_pca.shape[1]

print("\n" + "="*60)
print("             Model Evaluation & Blind Test Report")
print("="*60)
print(f"API Vector Reduction: 1024 -> {n_pca_dim} (90% Variance)")
print("\n[Blind Test Metrics]")
print(f"R² Score             : {r2_score(y_test, y_test_pred):.4f}")
print(f"MAE                  : {mean_absolute_error(y_test, y_test_pred):.4f}")
print(f"RMSE                 : {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"Pearson Correlation  : {pearsonr(y_test, y_test_pred)[0]:.4f}")

# ================= 7. Full Dataset Prediction & Export =================
print("\n>>> [Step 6] Exporting all predictions...")

y_all_pred = best_model.predict(X_final)
results_df = pd.DataFrame({
    'Row_Index': range(len(df)),
    'Actual_Value': y,
    'Predicted_Value': y_all_pred,
    'Set_Type': ['Train'] * 60 + ['Test'] * (len(df) - 60)
})

results_df['Residual'] = results_df['Actual_Value'] - results_df['Predicted_Value']
results_df.to_csv(PREDICTION_FILE, index=False)

print(f"\nTask Complete! Results saved to: {PREDICTION_FILE}")
print("="*60)