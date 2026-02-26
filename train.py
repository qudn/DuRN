import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import NearestNeighbors
from scipy.stats import rankdata
import os
import glob
import warnings
import sys

warnings.filterwarnings("ignore")

# ==================== Configuration Area ====================
DATA_DIRS = [
    'datasets' # Update this to the actual dataset directory path
]

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1024
MAX_EPOCHS = 30
ENSEMBLE_SIZE = 5
KNN_K = 5
OUTPUT_FILE = "final_benchmark_dual_view.csv" # Keep filename for compatibility with downstream scripts

# ==================== ResRPR Model ====================
class ResRPR(nn.Module):
    def __init__(self, input_dim, proj_dim, seed):
        super().__init__()
        torch.manual_seed(seed)
        # Random weight matrix W for target projection
        self.W = nn.Parameter(torch.randn(input_dim, proj_dim), requires_grad=False)
        nn.init.orthogonal_(self.W)
        
        hidden_dim = min(256, input_dim * 4 + 64)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LayerNorm(hidden_dim // 2), nn.SiLU()
        )
        self.head = nn.Linear(hidden_dim // 2, proj_dim)

    def forward(self, x):
        feat = self.encoder(x)
        pred = self.head(feat)
        return pred

    def get_target(self, x):
        # Negative selection / Identity mapping avoidance via random projection
        return x @ self.W

# ==================== Core Logic (Adaptive Hard Gating) ====================
def run_method(X_train, X_test, y_test):
    dim = X_train.shape[1]
    
    # --- View I: Raw Space (Topological Safety Net) ---
    # Sample if training data is too large to maintain efficiency
    knn_ref = X_train if len(X_train) < 10000 else X_train[np.random.choice(len(X_train), 10000, replace=False)]
    knn = NearestNeighbors(n_neighbors=KNN_K, n_jobs=-1).fit(knn_ref)
    dist_raw, _ = knn.kneighbors(X_test)
    s_raw = dist_raw.mean(axis=1)

    # --- View II: Deep Space & Uncertainty (Semantic Booster) ---
    X_tr_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    ensemble_errors = [] 
    
    for k in range(ENSEMBLE_SIZE):
        try:
            proj_dim = max(4, min(16, dim // 2))
            model = ResRPR(dim, proj_dim, seed=k).to(DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            
            dataset = torch.utils.data.TensorDataset(X_tr_t)
            loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            model.train()
            for _ in range(MAX_EPOCHS):
                for (bx,) in loader:
                    target = model.get_target(bx)
                    # Add small noise to enforce robustness and prevent manifold folding
                    noise = torch.randn_like(bx) * 0.01
                    pred = model(bx + noise)
                    loss = loss_fn(pred, target)
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            model.eval()
            errors_k = []
            with torch.no_grad():
                for i in range(0, len(X_test), BATCH_SIZE):
                    bx_te = torch.tensor(X_test[i:i+BATCH_SIZE], dtype=torch.float32).to(DEVICE)
                    target = model.get_target(bx_te)
                    pred = model(bx_te)
                    err = torch.sum((pred - target)**2, dim=1)
                    errors_k.append(err.cpu().numpy())
            
            ensemble_errors.append(np.concatenate(errors_k))
            del model, optimizer; torch.cuda.empty_cache()
        except: continue

    if not ensemble_errors:
        # Fallback if deep view fails completely
        if len(np.unique(y_test)) < 2: return 0, 0
        return roc_auc_score(y_test, s_raw), average_precision_score(y_test, s_raw)

    ensemble_errors = np.array(ensemble_errors)
    s_deep = np.mean(ensemble_errors, axis=0)
    uncertainty = np.var(ensemble_errors, axis=0)
    
    # --- Adaptive Gating Logic (The "Safety Net") ---
    u_mean = np.mean(uncertainty)
    u_std = np.std(uncertainty) + 1e-9
    u_norm = (uncertainty - u_mean) / u_std
    
    w_deep = np.zeros_like(s_deep)
    w_raw = np.ones_like(s_raw)

    # Hard Thresholds (Consistent with ablation settings)
    threshold_high = 0.5   # Unstable/High Uncertainty -> Prioritize Raw View
    threshold_low = -0.5   # Stable/Low Uncertainty -> Prioritize Deep View

    for i in range(len(u_norm)):
        if u_norm[i] > threshold_high:
            w_deep[i] = 0.01; w_raw[i] = 1.0
        elif u_norm[i] < threshold_low:
            w_deep[i] = 1.0; w_raw[i] = 0.3
        else:
            w_deep[i] = 0.5; w_raw[i] = 0.5
            
    # Fusion via Rank Alignment
    s_final = w_raw * rankdata(s_raw) + w_deep * rankdata(s_deep)
    
    if len(np.unique(y_test)) < 2: return 0, 0
    return roc_auc_score(y_test, s_final), average_precision_score(y_test, s_final)

# ==================== Data Processing ====================
def load_data(file_path):
    try:
        data = np.load(file_path, allow_pickle=True)
        if 'X' in data and 'y' in data: X, y = data['X'], data['y']
        else: keys = list(data.keys()); X, y = data[keys[0]], data[keys[1]]
        
        y = y.astype(int)
        if len(np.unique(y)) < 2: return None, None, None
        
        X_normal = X[y == 0]; X_anomalous = X[y == 1]
        y_normal = y[y == 0]; y_anomalous = y[y == 1]
        
        # Split normal data: 50% for training (unsupervised), 50% for testing
        X_train, X_test_normal, y_train, y_test_normal = train_test_split(
            X_normal, y_normal, test_size=0.5, random_state=42, stratify=y_normal
        )
        X_test = np.vstack((X_test_normal, X_anomalous))
        y_test = np.concatenate((y_test_normal, y_anomalous))
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test, y_test
    except: return None, None, None

def main():
    print("=== Running Full Benchmark with Adaptive Gating ===")
    all_files = []
    for d in DATA_DIRS:
        if os.path.exists(d): all_files.extend(glob.glob(os.path.join(d, '*.npz')))
    all_files.sort(key=lambda x: os.path.getsize(x))
    
    results = []
    print(f"{'Dataset':<20} | {'Dual ROC':<10} | {'Dual PR':<10}")
    print("-" * 50)
    
    for f in all_files:
        name = os.path.basename(f).replace('.npz', '')
        X_tr, X_te, y_te = load_data(f)
        if X_tr is None: continue
        
        try:
            roc, pr = run_method(X_tr, X_te, y_te)
            print(f"{name:<20} | {roc:.4f}     | {pr:.4f}")
            
            # Save columns for compatibility with gen_latex.py
            results.append({
                'Dataset': name,
                'Dual': roc,     # ROC Column
                'Ours_PR': pr    # PR Column
            })
        except KeyboardInterrupt: break
        except Exception: continue
        
    # Save Results
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n✅ Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
