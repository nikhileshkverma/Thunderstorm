#!/usr/bin/env python3
"""
06_train_deep_learning.py — VAE, Autoencoder, Transformer training  (v1)
==========================================================================
Models:
  1. Autoencoder     — basic DL baseline (encoder-decoder, binary clf head)
  2. VAE             — Variational Autoencoder (PRIMARY per Waylon PDF)
                       Based on Hamid Kamangir's architecture
  3. Transformer     — attention-based model for tabular features

All models:
  - Use Tesla V100 GPU (CUDA)
  - Train on same train/val/test splits as RF and XGBoost
  - Evaluated with same metrics: PSS, HSS, CSI, POD, FAR, POFD
  - Results appended to metrics_train_val_test.csv for master comparison

Usage:
  bash run_pipeline.sh 6         # train all DL models
  python 06_train_deep_learning.py --model vae
  python 06_train_deep_learning.py --model transformer
  python 06_train_deep_learning.py --model autoencoder
  python 06_train_deep_learning.py --model all
"""

import sys, json, pickle, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from utils.config_loader import load_config
from utils.metrics import compute_all_metrics, print_metrics_table

# ── Try importing PyTorch ──────────────────────────────────────────────────
# ── Try importing PyTorch ──────────────────────────────────────────────────
try:
    import torch
    TORCH_OK = True
except ImportError as e:
    TORCH_OK = False
    print("  ERROR: PyTorch not installed in this environment")
    print("  Details:", e)

# 👉 Only run rest if torch import worked
if TORCH_OK:
    try:
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  PyTorch {torch.__version__}  |  Device: {DEVICE}")

        if DEVICE.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    except Exception as e:
        TORCH_OK = False
        print("  ERROR: Torch submodules failed to import")
        print("  Details:", e)


# =============================================================================
#  DATA LOADING
# =============================================================================
def load_split(name, dataset_dir, cfg, feature_names=None):
    p = dataset_dir / f"{name}.csv"
    if not p.exists():
        print(f"  ERROR: {p} not found. Run steps 1-3 first."); sys.exit(1)
    df = pd.read_csv(p)
    drop = [c for c in cfg["non_feature_cols"]+["label","valid_time","region"]
            if c in df.columns]
    y = df["label"].values.astype(np.float32) if "label" in df.columns else None
    X = df.drop(columns=drop, errors="ignore").select_dtypes(include=[np.number])
    if feature_names is not None:
        for col in feature_names:
            if col not in X.columns: X[col] = 0.0
        X = X[feature_names]
    X = X.fillna(X.median()).astype(np.float32)
    return X.values, y

def make_tensors(X, y, device):
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    yt = torch.tensor(y, dtype=torch.float32).to(device)
    return Xt, yt

def find_threshold(y_val, probs, metric="CSI"):
    best_score = -1; best_t = 0.5
    for t in np.arange(0.10, 0.90, 0.05):
        m = compute_all_metrics(y_val, (probs >= t).astype(int))
        if m[metric] > best_score:
            best_score = m[metric]; best_t = t
    return round(float(best_t), 2)


# =============================================================================
#  MODEL 1: AUTOENCODER WITH CLASSIFICATION HEAD
# =============================================================================
class LightningAutoencoder(nn.Module):
    """
    Encoder-Decoder autoencoder with a binary classification head.
    Trained jointly: reconstruction loss + binary cross-entropy.
    The encoder learns compressed representations of atmospheric state.
    """
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (reverse)
        dec_layers = []
        rev_dims = list(reversed(hidden_dims[:-1])) + [input_dim]
        prev = hidden_dims[-1]
        for h in rev_dims:
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        self.decoder = nn.Sequential(*dec_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        prob = self.classifier(z).squeeze(1)
        return prob, recon


# =============================================================================
#  MODEL 2: VARIATIONAL AUTOENCODER (PRIMARY — Waylon PDF, Hamid's design)
# =============================================================================
class LightningVAE(nn.Module):
    """
    Variational Autoencoder for lightning prediction.
    PRIMARY model per Waylon Collins PDF.
    Based on Hamid Kamangir's architecture adapted for tabular HRRR features.

    Architecture:
      Encoder: input → hidden layers → (mu, log_var) in latent space
      Reparameterisation: z = mu + eps * exp(0.5 * log_var)
      Decoder: z → hidden layers → reconstructed input
      Classifier: z → binary lightning prediction
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout=0.3):
        super().__init__()
        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                           nn.LeakyReLU(0.2), nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu     = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

        # Decoder
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(prev, h), nn.LeakyReLU(0.2),
                           nn.Dropout(dropout)]
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Classification head on latent space z
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference

    def forward(self, x):
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z      = self.reparameterize(mu, logvar)
        recon  = self.decoder(z)
        prob   = self.classifier(z).squeeze(1)
        return prob, recon, mu, logvar


def vae_loss(prob, y, recon, x, mu, logvar, beta=1.0,
             pos_weight=None):
    """
    Combined VAE loss:
      L = BCE(classification) + MSE(reconstruction) + beta * KL-divergence
    beta controls balance between reconstruction quality and latent regularisation.
    """
    # Classification loss (weighted for class imbalance)
    if pos_weight is not None:
        bce = nn.BCELoss(reduction="none")(prob, y)
        weights = torch.where(y == 1, pos_weight, torch.ones_like(y))
        clf_loss = (bce * weights).mean()
    else:
        clf_loss = nn.BCELoss()(prob, y)

    # Reconstruction loss
    recon_loss = nn.MSELoss()(recon, x)

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    return clf_loss + recon_loss + beta * kl_loss


# =============================================================================
#  MODEL 3: TRANSFORMER
# =============================================================================
class LightningTransformer(nn.Module):
    """
    Transformer-based model for tabular lightning prediction.
    Treats each feature as a "token" in a sequence.
    Uses multi-head self-attention to capture feature interactions.
    """
    def __init__(self, input_dim, d_model, nhead, num_layers,
                 dim_feedforward, dropout=0.1):
        super().__init__()
        # Project input features to d_model dimensions
        self.input_proj = nn.Linear(input_dim, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (batch, input_dim) → add sequence dim → (batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.input_proj(x)           # (batch, 1, d_model)
        x = self.transformer(x)          # (batch, 1, d_model)
        x = self.norm(x)
        x = x.squeeze(1)                 # (batch, d_model)
        return self.classifier(x).squeeze(1)


# =============================================================================
#  TRAINING LOOP (shared)
# =============================================================================
def train_model(model, train_loader, val_X, val_y, optimizer, scheduler,
                epochs, early_stopping, model_type, pos_weight, beta=1.0):
    """Universal training loop for all DL models."""
    best_val_csi = -1; patience = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            if model_type == "vae":
                prob, recon, mu, logvar = model(xb)
                loss = vae_loss(prob, yb, recon, xb, mu, logvar,
                                beta=beta, pos_weight=pos_weight)
            elif model_type == "autoencoder":
                prob, recon = model(xb)
                clf = nn.BCELoss()(prob, yb)
                rec = nn.MSELoss()(recon, xb)
                loss = clf + 0.1 * rec
            else:  # transformer
                prob = model(xb)
                w = torch.where(yb==1, pos_weight, torch.ones_like(yb))
                loss = (nn.BCELoss(reduction="none")(prob,yb)*w).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            if model_type == "vae":
                val_prob, _, _, _ = model(val_X)
            elif model_type == "autoencoder":
                val_prob, _ = model(val_X)
            else:
                val_prob = model(val_X)
            val_probs = val_prob.cpu().numpy()

        val_pred = (val_probs >= 0.5).astype(int)
        val_m = compute_all_metrics(val_y, val_pred)
        csi = val_m["CSI"]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:3d}/{epochs}  loss={epoch_loss/len(train_loader):.4f}  "
                  f"val_CSI={csi:.3f}  val_POD={val_m['POD']:.3f}  "
                  f"val_FAR={val_m['FAR']:.3f}")

        if csi > best_val_csi:
            best_val_csi = csi; patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= early_stopping:
                print(f"    Early stopping at epoch {epoch+1} (best val CSI={best_val_csi:.3f})")
                break

        if scheduler: scheduler.step()

    if best_state:
        model.load_state_dict(best_state)
    return model


# =============================================================================
#  MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all",
                        choices=["all","vae","transformer","autoencoder"])
    args = parser.parse_args()

    cfg         = load_config()
    dataset_dir = cfg["paths"]["dataset_dir"]
    models_dir  = cfg["paths"]["models_dir"]
    results_dir = cfg["paths"]["results_dir"]

    if not TORCH_OK:
        print("ERROR: PyTorch required. Run: pip install torch")
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  STEP 6: DEEP LEARNING MODELS  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Training: {args.model}  |  Device: {DEVICE}")
    print(f"{'='*65}\n")

    feat_path = models_dir / "feature_names.json"
    if not feat_path.exists():
        print("  ERROR: Run steps 1-4 first (need feature_names.json)"); sys.exit(1)
    feature_names = json.load(open(feat_path))

    X_tr, y_tr = load_split("train",   dataset_dir, cfg, feature_names)
    X_va, y_va = load_split("val",     dataset_dir, cfg, feature_names)
    X_te, y_te = load_split("test",    dataset_dir, cfg, feature_names)

    input_dim = X_tr.shape[1]
    n_pos = int(y_tr.sum()); n_neg = int((y_tr==0).sum())
    pos_w = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32).to(DEVICE)
    print(f"  Input dim: {input_dim}  | Train: {X_tr.shape}  "
          f"| Lightning rate: {y_tr.mean():.1%}")

    # Normalise features (important for DL models — RF/XGB don't need this)
    mu_  = X_tr.mean(axis=0, keepdims=True)
    sig_ = X_tr.std(axis=0, keepdims=True) + 1e-8
    X_tr = (X_tr - mu_) / sig_
    X_va = (X_va - mu_) / sig_
    X_te = (X_te - mu_) / sig_

    Xtr_t, ytr_t = make_tensors(X_tr, y_tr, DEVICE)
    Xva_t, _     = make_tensors(X_va, y_va, DEVICE)
    Xte_t, _     = make_tensors(X_te, y_te, DEVICE)

    train_ds     = TensorDataset(Xtr_t, ytr_t)
    train_loader = DataLoader(train_ds,
                              batch_size=cfg["models"]["vae"]["batch_size"],
                              shuffle=True)

    all_metrics = {}

    # Try to load existing metrics to append to
    mpath = results_dir / "metrics_train_val_test.csv"
    if mpath.exists():
        existing = pd.read_csv(mpath)
    else:
        existing = pd.DataFrame()

    def run_model(model_name, model, model_type, lr, epochs, early_stop, beta=1.0):
        print(f"\n── {model_name} ────────────────────────────────────────")
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

        model = train_model(model, train_loader, Xva_t, y_va, optimizer,
                            scheduler, epochs, early_stop, model_type,
                            pos_w, beta=beta)

        model.eval()
        results = {}
        for sname, Xt, y_s in [("train",Xtr_t,y_tr),("val",Xva_t,y_va),
                                ("test",Xte_t,y_te)]:
            with torch.no_grad():
                if model_type == "vae":
                    prob, _, _, _ = model(Xt)
                elif model_type == "autoencoder":
                    prob, _ = model(Xt)
                else:
                    prob = model(Xt)
            probs = prob.cpu().numpy()
            thresh = find_threshold(y_va, model.to(DEVICE) and
                                    (lambda p=probs if sname=="val" else
                                     model(Xva_t)[0].cpu().numpy(): p)(),
                                    "CSI") if sname=="val" else 0.5
            # simpler threshold finding
            if sname == "val":
                thresh = find_threshold(y_s, probs, "CSI")
            pred = (probs >= thresh).astype(int)
            m = compute_all_metrics(y_s, pred, probs, threshold=thresh)
            print_metrics_table(m, f"{model_name} {sname.upper()}")
            results[sname] = m

        torch.save(model.state_dict(),
                   models_dir / f"{model_type}_model.pt")
        print(f"  Saved → {models_dir/f'{model_type}_model.pt'}")
        return results

    models_to_run = (["autoencoder","vae","transformer"]
                     if args.model == "all" else [args.model])

    for mname in models_to_run:
        mc = cfg["models"]
        if mname == "autoencoder":
            m = LightningAutoencoder(input_dim,
                                     mc["autoencoder"]["hidden_dims"])
            res = run_model("Autoencoder", m, "autoencoder",
                            mc["autoencoder"]["learning_rate"],
                            mc["autoencoder"]["epochs"],
                            mc["autoencoder"]["early_stopping"])
        elif mname == "vae":
            m = LightningVAE(input_dim, mc["vae"]["hidden_dims"],
                             mc["vae"]["latent_dim"])
            res = run_model("VAE", m, "vae",
                            mc["vae"]["learning_rate"],
                            mc["vae"]["epochs"],
                            mc["vae"]["early_stopping"],
                            beta=mc["vae"]["beta"])
        elif mname == "transformer":
            m = LightningTransformer(input_dim,
                                     mc["transformer"]["d_model"],
                                     mc["transformer"]["nhead"],
                                     mc["transformer"]["num_layers"],
                                     mc["transformer"]["dim_feedforward"],
                                     mc["transformer"]["dropout"])
            res = run_model("Transformer", m, "transformer",
                            mc["transformer"]["learning_rate"],
                            mc["transformer"]["epochs"],
                            mc["transformer"]["early_stopping"])
        else:
            continue

        for split_name, split_m in res.items():
            all_metrics[f"{mname.upper()}_{split_name}"] = split_m

    # Append new results to master metrics CSV
    new_rows = [{"model":k.rsplit("_",1)[0], "split":k.rsplit("_",1)[1], **m}
                for k,m in all_metrics.items()]
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing, new_df], ignore_index=True)
        # Remove old entries for same model+split
        combined = combined.drop_duplicates(subset=["model","split"], keep="last")
        combined.to_csv(mpath, index=False)
        print(f"\n  Results appended to {mpath}")

    print(f"\n{'='*65}")
    print(f"  DEEP LEARNING TRAINING COMPLETE  |  {datetime.now().strftime('%H:%M')}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
