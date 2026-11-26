"""
Train a deployable landslide-recurrence predictor.

Target: `Reccuring=1 New-slides=0`  (1 = recurring slide, 0 = new slide)

Design decisions (see README.md for the reasoning):
  * `Distance to Historical slides` is DROPPED. It is definitional leakage:
    "recurring" essentially means "on top of a past slide", so that column
    almost perfectly separates the classes and teaches the model nothing
    physical.
  * `Data Source` and `Zone` are DROPPED. They are labeling / spatial
    artifacts (e.g. USGS points are 81% recurring), not ground conditions.
  * `Lon`/`Lat` are used ONLY to build spatial cross-validation blocks, not as
    features, so the reported accuracy reflects real predictive skill at
    unseen locations rather than memorized geography.
  * LightGBM is used because it handles the heavy missingness natively
    (Depth-to-mine is 71% missing) and gives interpretable feature importance.
"""
import json
import re
import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
import lightgbm as lgb

CSV = "landslides.csv"
TARGET = "Reccuring=1 New-slides=0"
NA = ["<Null>", ""]

# Columns excluded from the feature set.
LEAKAGE = ["Distance to Historical slides [ft]"]      # definitional
ARTIFACTS = ["Data Source", "Zone"]            # labeling / admin proxies
IDENT = ["ID"]
GEO = ["Lon (Degree)", "Lat (Degree)"]                # used for spatial CV only

N_BLOCKS = 12   # spatial blocks for cross-validation
SEED = 42


def safe_name(col):
    """LightGBM rejects []{}\":, and other JSON-special chars in feature names."""
    return re.sub(r"[^0-9A-Za-z]+", "_", col).strip("_")


def load():
    df = pd.read_csv(CSV, na_values=NA)
    y = df[TARGET].astype(int)
    drop = LEAKAGE + ARTIFACTS + IDENT + GEO + [TARGET]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    coords = df[GEO].to_numpy()
    name_map = {c: safe_name(c) for c in X.columns}   # safe -> original stored in artifact
    X = X.rename(columns=name_map)
    return df, X, y, coords, {v: k for k, v in name_map.items()}


def spatial_blocks(coords, n_blocks=N_BLOCKS):
    """Cluster points into geographic blocks so CV folds are spatially separated."""
    km = KMeans(n_clusters=n_blocks, random_state=SEED, n_init=10)
    return km.fit_predict(coords)


def lgb_params(y):
    pos = (y == 1).sum()
    neg = (y == 0).sum()
    return dict(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        scale_pos_weight=neg / pos,   # counter class imbalance
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def main():
    df, X, y, coords, orig_names = load()
    blocks = spatial_blocks(coords)
    params = lgb_params(y)
    print(f"Rows: {len(X)}  Features: {X.shape[1]}  "
          f"Positives(recurring): {(y==1).sum()}  Negatives(new): {(y==0).sum()}")
    print(f"Features used: {[orig_names[c] for c in X.columns]}\n")

    # ---- Spatially-blocked cross-validation (honest performance estimate) ----
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    aucs, aps = [], []
    for k, (tr, va) in enumerate(cv.split(X, y, groups=blocks)):
        model = lgb.LGBMClassifier(**params)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        auc = roc_auc_score(y.iloc[va], p)
        ap = average_precision_score(y.iloc[va], p)
        aucs.append(auc); aps.append(ap)
        print(f"  fold {k}: ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}  (n_val={len(va)})")

    print("\n=== Spatially-validated performance ===")
    print(f"ROC-AUC : {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")
    print(f"PR-AUC  : {np.mean(aps):.3f} +/- {np.std(aps):.3f}")

    # Pick an operating threshold that maximizes F1 on out-of-fold predictions.
    ths = np.linspace(0.1, 0.9, 81)
    f1s = [f1_score(y, (oof >= t).astype(int)) for t in ths]
    best_t = float(ths[int(np.argmax(f1s))])
    pred = (oof >= best_t).astype(int)
    print(f"\nChosen threshold (max F1): {best_t:.2f}")
    print(f"F1={f1_score(y, pred):.3f}  "
          f"Balanced-acc={balanced_accuracy_score(y, pred):.3f}")
    print("\nConfusion matrix (rows=true 0/1, cols=pred 0/1):")
    print(confusion_matrix(y, pred))
    print("\n" + classification_report(y, pred, target_names=["new", "recurring"]))

    # ---- Fit the final model on ALL data for deployment ----
    final = lgb.LGBMClassifier(**params)
    final.fit(X, y)

    imp = (pd.Series(final.feature_importances_,
                     index=[orig_names[c] for c in X.columns])
           .sort_values(ascending=False))
    print("=== Feature importance (final model) ===")
    print(imp.to_string())

    artifact = {
        "model": final,
        "features": list(X.columns),        # sanitized names, in model order
        "orig_names": orig_names,           # sanitized -> original CSV header
        "threshold": best_t,
        "cv_roc_auc": float(np.mean(aucs)),
        "cv_pr_auc": float(np.mean(aps)),
    }
    joblib.dump(artifact, "landslide_recurrence_model.joblib")
    imp.to_csv("feature_importance.csv", header=["importance"])
    with open("metrics.json", "w") as f:
        json.dump({
            "roc_auc_mean": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs)),
            "pr_auc_mean": float(np.mean(aps)),
            "pr_auc_std": float(np.std(aps)),
            "threshold": best_t,
            "f1": float(f1_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "n_rows": int(len(X)),
            "n_features": int(X.shape[1]),
        }, f, indent=2)
    print("\nSaved: landslide_recurrence_model.joblib, metrics.json, feature_importance.csv")


if __name__ == "__main__":
    main()
