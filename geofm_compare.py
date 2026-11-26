"""
Head-to-head: geospatial foundation model embeddings vs the tabular baseline.

Representations, all evaluated on the SAME points and the SAME spatial folds:
  * Tabular       : the 19 engineered terrain/geologic features (the baseline)
  * Prithvi       : 768-dim Sentinel-2 image embeddings (imagery GeoFM)
  * SatCLIP       : 256-dim location embeddings (location GeoFM)
  * Tabular+Prithvi, Tabular+SatCLIP : fusion

Each is scored with two classifiers so the comparison is fair:
  * Linear probe  : median-impute + standardize + logistic regression
                    (the standard way to evaluate frozen FM embeddings)
  * LightGBM      : gradient-boosted trees (matches the deployable baseline)

The question: do learned satellite/location embeddings add predictive signal for
recurrence beyond the hand-engineered terrain features?
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

from train import load, lgb_params, SEED

warnings.filterwarnings("ignore")
N_BLOCKS = 8


def spatial_folds(lon, lat, y, n_splits=5):
    blocks = KMeans(n_clusters=N_BLOCKS, random_state=SEED, n_init=10).fit_predict(
        np.stack([lon, lat], axis=1))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    return list(cv.split(np.zeros(len(y)), y, groups=blocks))


def eval_repr(X, y, folds, clf):
    """Return mean ROC-AUC and PR-AUC over spatial folds for a given classifier."""
    aucs, aps = [], []
    for tr, va in folds:
        if clf == "logit":
            model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(max_iter=2000, C=1.0))
            model.fit(X[tr], y[tr]); p = model.predict_proba(X[va])[:, 1]
        else:
            model = lgb.LGBMClassifier(**lgb_params(pd.Series(y)))
            model.fit(X[tr], y[tr]); p = model.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p)); aps.append(average_precision_score(y[va], p))
    return np.mean(aucs), np.std(aucs), np.mean(aps)


def main():
    pri = np.load("prithvi_embeddings.npz")
    sat = np.load("satclip_embeddings.npz")
    # align on the common point set (SatCLIP was built from Prithvi's idx, so equal)
    assert np.array_equal(pri["idx"], sat["idx"]), "embedding point sets differ"
    idx, lon, lat, y = pri["idx"], pri["lon"], pri["lat"], pri["y"].astype(int)
    Epri, Esat = pri["emb"], sat["emb"]

    # tabular features for the same rows (idx == original CSV row order)
    _, Xtab_all, y_all, _, _ = load()
    Xtab = Xtab_all.iloc[idx].to_numpy(dtype=float)
    assert np.array_equal(y_all.values[idx], y)

    folds = spatial_folds(lon, lat, y)
    reps = {
        "Tabular (19 feats)": Xtab,
        "Prithvi img emb (768)": Epri,
        "SatCLIP loc emb (256)": Esat,
        "Tabular + Prithvi": np.hstack([Xtab, Epri]),
        "Tabular + SatCLIP": np.hstack([Xtab, Esat]),
    }

    print(f"n={len(y)} points ({int(y.sum())} recurring), spatial 5-fold over "
          f"{N_BLOCKS} blocks\n")
    print(f"{'representation':24s} {'LinearProbe AUC':>17s} {'LightGBM AUC':>15s}")
    print("-" * 58)
    rows = []
    for name, X in reps.items():
        la, ls, lap = eval_repr(X, y, folds, "logit")
        ga, gs, gap = eval_repr(X, y, folds, "lgbm")
        rows.append([name, la, ls, lap, ga, gs, gap])
        print(f"{name:24s} {la:6.3f} +/- {ls:4.3f}   {ga:6.3f} +/- {gs:4.3f}")

    pd.DataFrame(rows, columns=["representation", "logit_auc", "logit_std",
                 "logit_pr", "lgbm_auc", "lgbm_std", "lgbm_pr"]).to_csv(
        "geofm_comparison.csv", index=False)
    print("\nSaved geofm_comparison.csv")


if __name__ == "__main__":
    main()
