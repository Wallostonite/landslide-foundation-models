"""Modeling helpers: spatial blocks, LightGBM config, spatial-CV OOF, calibration."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import lightgbm as lgb
import config as C


def spatial_blocks(coords, n_blocks=C.N_SPATIAL_BLOCKS):
    """Cluster points into geographic blocks so CV holds out whole regions."""
    return KMeans(n_clusters=n_blocks, random_state=C.SEED, n_init=10).fit_predict(coords)


def lgb_params(y):
    pos = int((y == 1).sum()); neg = int((y == 0).sum())
    return dict(objective="binary", n_estimators=600, learning_rate=0.03,
                num_leaves=31, min_child_samples=30, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, scale_pos_weight=neg / pos,
                random_state=C.SEED, n_jobs=-1, verbose=-1)


def spatial_cv_oof(X: pd.DataFrame, y: pd.Series, blocks):
    """Out-of-fold probabilities + per-fold ROC/PR AUC over spatial folds."""
    cv = StratifiedGroupKFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.SEED)
    oof = np.full(len(y), np.nan)
    aucs, aps = [], []
    for tr, va in cv.split(X, y, groups=blocks):
        m = lgb.LGBMClassifier(**lgb_params(y))
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(y.iloc[va], p))
        aps.append(average_precision_score(y.iloc[va], p))
    return oof, np.array(aucs), np.array(aps)


def fit_calibrator(oof, y):
    """Isotonic mapping from OOF scores to calibrated probabilities."""
    return IsotonicRegression(out_of_bounds="clip").fit(oof, y)
