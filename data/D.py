import os
import time
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import xgboost as xgb

# ================= CONFIGURATION =================
# Replace with your actual file names
INPUT_FILE = 'input_data.csv'
OUTPUT_EMBEDDING_FILE = 'features_embedded.csv'

# PRIVACY PROTECTED: Load API credentials from environment variables
# OR use a placeholder. NEVER hardcode your key on GitHub.
API_KEY = os.getenv("MY_API_KEY", "your_api_key_here")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def get_embedding(text, model="text-embedding-v4", dim=1024):
    """
    Inputs text into the LLM Embedding API.
    - text: The molecular string or description (e.g., SMILES).
    - model: The specific embedding model chosen for high-dimensional feature extraction.
    - dim: Output vector size (1024 for text-embedding-v4).
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Pre-processing: remove newlines to maintain clean API input strings
            text = str(text).replace("\n", " ")
            resp = client.embeddings.create(model=model, input=[text], dimensions=dim)
            return resp.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            # Return zero-vector if all retries fail to prevent script crash
            return [0.0] * dim


# ================= 1. DATA LOADING & EMBEDDING =================
print(">>> [Step 1] Loading data and generating embeddings...")
df = pd.read_csv(INPUT_FILE)

# Dynamically select columns to avoid hardcoding local indices
# target_col: The property you want to predict
# prompt_col: The text/SMILES used for embedding
target_col_name = df.columns[2]
prompt_col_name = df.columns[8]

if os.path.exists(OUTPUT_EMBEDDING_FILE):
    print(f">>> Loading existing features from {OUTPUT_EMBEDDING_FILE}...")
    df_emb = pd.read_csv(OUTPUT_EMBEDDING_FILE)
    embedding_cols = [c for c in df_emb.columns if c.startswith('emb_')]
    X_raw = df_emb[embedding_cols].values
    y = df_emb[target_col_name].values
else:
    print(">>> Calling API to generate Embeddings. This may take a while...")
    embeddings = df[prompt_col_name].apply(lambda x: get_embedding(x))
    X_raw = np.array(embeddings.tolist())

    # Save embeddings to avoid redundant API costs/calls
    emb_df = pd.DataFrame(X_raw, columns=[f'emb_{i}' for i in range(X_raw.shape[1])])
    pd.concat([df, emb_df], axis=1).to_csv(OUTPUT_EMBEDDING_FILE, index=False)
    y = df[target_col_name].values

# ================= 2. PCA DIMENSIONALITY REDUCTION =================
print("\n>>> [Step 2] Standardization and PCA Reduction...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Innovation: Reducing 1024-dim API vectors to a space capturing 90% variance
pca = PCA(n_components=0.90, random_state=42)
X_pca = pca.fit_transform(X_scaled)
n_features = X_pca.shape[1]

print("=" * 40)
print(f"★ Original API Dimensions: {X_raw.shape[1]}")
print(f"★ PCA Reduced Dimensions : {n_features}")
print("=" * 40)

# ================= 3. DATASET SPLITTING =================
# Fixed split: 60 for training, 15 for testing
X_train, y_train = X_pca[:60], y[:60]
X_test, y_test = X_pca[-15:], y[-15:]

# ================= 4. XGBOOST RANDOMIZED SEARCH =================
print("\n>>> [Step 4] XGBoost Hyperparameter Tuning (Randomized Search)...")

param_distributions = {
    'n_estimators': [100, 300, 500, 800],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4, 5],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 2, 5]
}

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Randomized search is more efficient for high-dimensional search spaces
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_distributions,
    n_iter=500,
    scoring='neg_root_mean_squared_error',
    cv=3,
    verbose=1,
    n_jobs=4,
    random_state=42
)

random_search.fit(X_train, y_train)
best_model = random_search.best_estimator_
best_params = random_search.best_params_

# ================= 5. EVALUATION =================
print("\n>>> [Step 5] Final Evaluation...")
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
pearson_corr, _ = pearsonr(y_test, y_pred)

# ================= 6. OUTPUT RESULTS =================
print("\n" + "=" * 45)
print("             Final Performance Report")
print("=" * 45)
print(f"PCA Components       : {n_features}")
print("\n[Optimal Parameters]")
for k, v in best_params.items():
    print(f"{k:20}: {v}")

print("\n[Model Metrics]")
print(f"R² Score             : {r2:.4f}")
print(f"MAE                  : {mae:.4f}")
print(f"RMSE                 : {rmse:.4f}")
print(f"Pearson Correlation  : {pearson_corr:.4f}")

print("\n[Prediction Preview (First 5 Samples)]")
res_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(res_df.head(5).to_string(index=False))