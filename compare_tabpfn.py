"""
Head-to-head: TabPFN v2 vs LightGBM on the landslide-recurrence task.

Same 19 physical/geologic features, same spatial cross-validation folds, so the
two models are scored on identical train/validation splits. Leakage columns are
excluded exactly as in train.py (that logic is imported, not duplicated).
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb

from train import load, spatial_blocks, lgb_params, SEED

warnings.filterwarnings("ignore")


def main():
    df, X, y, coords, orig_names = load()
    blocks = spatial_blocks(coords)
    Xv = X.to_numpy(dtype=float)   # TabPFN wants a numeric array; NaNs are OK
    yv = y.to_numpy()

    from tabpfn import TabPFNClassifier

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for k, (tr, va) in enumerate(cv.split(X, y, groups=blocks)):
        # --- LightGBM ---
        gbm = lgb.LGBMClassifier(**lgb_params(y))
        gbm.fit(X.iloc[tr], y.iloc[tr])
        p_gbm = gbm.predict_proba(X.iloc[va])[:, 1]

        # --- TabPFN v2 ---
        tab = TabPFNClassifier(ignore_pretraining_limits=True, random_state=SEED)
        tab.fit(Xv[tr], yv[tr])
        p_tab = tab.predict_proba(Xv[va])[:, 1]

        r = {
            "fold": k, "n_val": len(va),
            "gbm_auc": roc_auc_score(yv[va], p_gbm),
            "tab_auc": roc_auc_score(yv[va], p_tab),
            "gbm_ap": average_precision_score(yv[va], p_gbm),
            "tab_ap": average_precision_score(yv[va], p_tab),
        }
        rows.append(r)
        print(f"fold {k} (n={len(va):4d}): "
              f"AUC  gbm={r['gbm_auc']:.3f} tab={r['tab_auc']:.3f}   "
              f"PR-AUC gbm={r['gbm_ap']:.3f} tab={r['tab_ap']:.3f}")

    res = pd.DataFrame(rows)
    print("\n=== Mean over spatial folds ===")
    print(f"ROC-AUC   LightGBM {res.gbm_auc.mean():.3f} +/- {res.gbm_auc.std():.3f}"
          f"    TabPFN {res.tab_auc.mean():.3f} +/- {res.tab_auc.std():.3f}")
    print(f"PR-AUC    LightGBM {res.gbm_ap.mean():.3f} +/- {res.gbm_ap.std():.3f}"
          f"    TabPFN {res.tab_ap.mean():.3f} +/- {res.tab_ap.std():.3f}")
    res.to_csv("tabpfn_vs_lgbm.csv", index=False)
    print("\nSaved per-fold results to tabpfn_vs_lgbm.csv")


if __name__ == "__main__":
    main()
