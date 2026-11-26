"""
Train and evaluate the Tabular + SatCLIP landslide-recurrence model.

Pipeline:
  1. Load data, DROP all leakage/coord columns (see config.LEAKAGE_COLS).
  2. Build SatCLIP location embeddings for every point (cached).
  3. Spatial block cross-validation of:
        (a) Tabular only            -- the reference
        (b) Tabular + SatCLIP       -- the deployed model
     so SatCLIP's marginal contribution is measured honestly.
  4. Isotonic-calibrate the deployed model, pick an F1-optimal threshold.
  5. Fit the final model on all data, save artifact + metrics + plots.
"""
import json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix,
                             f1_score, balanced_accuracy_score, brier_score_loss,
                             classification_report)
from sklearn.calibration import calibration_curve

import config as C
from data import load
from satclip_features import get_or_build
from features import build_matrix, satclip_columns
from model import spatial_blocks, lgb_params, spatial_cv_oof, fit_calibrator

warnings.filterwarnings("ignore")
C.PLOTS.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, GREEN, PURPLE = "#2b6cb0", "#c05621", "#2f855a", "#4c51bf"


def best_threshold(y, p):
    ths = np.linspace(0.05, 0.95, 91)
    return float(ths[int(np.argmax([f1_score(y, (p >= t)) for t in ths]))])


def main():
    d = load()
    X_tab, y, coords, orig = d["X"], d["y"], d["coords"], d["orig_names"]
    print(f"Rows: {len(X_tab)}  Tabular features: {X_tab.shape[1]}")
    print(f"Dropped leakage/coord columns: {d['dropped']}")

    emb = get_or_build(coords)
    X_full = build_matrix(X_tab, emb)
    sat_cols = satclip_columns(emb.shape[1])
    print(f"Combined matrix: {X_full.shape[1]} features "
          f"({X_tab.shape[1]} tabular + {len(sat_cols)} SatCLIP)\n")

    blocks = spatial_blocks(coords)

    # ---- (a) tabular-only reference ----
    oof_tab, auc_tab, ap_tab = spatial_cv_oof(X_tab, y, blocks)
    # ---- (b) tabular + SatCLIP (deployed) ----
    oof, aucs, aps = spatial_cv_oof(X_full, y, blocks)

    print("=== Spatial 5-fold cross-validation (ROC-AUC) ===")
    print(f"Tabular only      : {auc_tab.mean():.3f} +/- {auc_tab.std():.3f}")
    print(f"Tabular + SatCLIP : {aucs.mean():.3f} +/- {aucs.std():.3f}   "
          f"(Δ = {aucs.mean()-auc_tab.mean():+.3f})")
    print(f"PR-AUC (Tab+SatCLIP): {aps.mean():.3f}\n")

    # ---- calibration + threshold on the deployed model ----
    cal = fit_calibrator(oof, y)
    cal_oof = cal.predict(oof)
    thr = best_threshold(y, cal_oof)
    pred = (cal_oof >= thr).astype(int)
    brier_raw = brier_score_loss(y, oof)
    brier_cal = brier_score_loss(y, cal_oof)
    print(f"Calibration Brier: raw={brier_raw:.4f} -> calibrated={brier_cal:.4f}")
    print(f"Threshold (F1-opt): {thr:.2f}")
    print(f"F1={f1_score(y,pred):.3f}  Balanced-acc={balanced_accuracy_score(y,pred):.3f}")
    print(confusion_matrix(y, pred))
    print(classification_report(y, pred, target_names=["new", "recurring"]))

    # ---- final model on all data ----
    final = lgb.LGBMClassifier(**lgb_params(y)); final.fit(X_full, y)

    # importance: tabular vs SatCLIP share
    imp = pd.Series(final.feature_importances_, index=X_full.columns)
    sat_share = imp[sat_cols].sum() / imp.sum()
    tab_imp = (imp.drop(sat_cols).rename(index=orig).sort_values(ascending=False))
    print(f"\nSatCLIP share of total importance: {sat_share:.1%}")
    print("Top tabular features:")
    print(tab_imp.head(8).to_string())

    # ---- plots ----
    _plots(y, oof, cal_oof, thr, aucs.mean(), aps.mean(), brier_raw, brier_cal,
           tab_imp, sat_share, auc_tab.mean(), aucs.mean())

    # ---- save ----
    art = dict(model=final, calibrator=cal, threshold=thr,
               tabular_features=list(X_tab.columns), satclip_dims=emb.shape[1],
               orig_names=orig, satclip_repo=C.SATCLIP_REPO, satclip_ckpt=C.SATCLIP_CKPT)
    joblib.dump(art, C.MODEL_FILE)
    metrics = dict(
        n_rows=int(len(y)), n_tabular=int(X_tab.shape[1]), n_satclip=int(emb.shape[1]),
        roc_auc_tabular=float(auc_tab.mean()), roc_auc_tabular_std=float(auc_tab.std()),
        roc_auc_combined=float(aucs.mean()), roc_auc_combined_std=float(aucs.std()),
        pr_auc_combined=float(aps.mean()), delta_auc=float(aucs.mean()-auc_tab.mean()),
        brier_raw=float(brier_raw), brier_calibrated=float(brier_cal),
        threshold=thr, f1=float(f1_score(y, pred)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        satclip_importance_share=float(sat_share))
    json.dump(metrics, open(C.METRICS_FILE, "w"), indent=2)
    tab_imp.to_csv(C.OUT / "feature_importance.csv", header=["importance"])
    print(f"\nSaved model, metrics, feature_importance, and {len(list(C.PLOTS.glob('*.png')))} plots.")


def _plots(y, oof, cal_oof, thr, auc, ap, brier_raw, brier_cal, tab_imp,
           sat_share, auc_tab, auc_comb):
    # ROC + PR
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y, cal_oof)
    ax[0].plot(fpr, tpr, color=BLUE, lw=2, label=f"AUC = {auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="ROC (spatial CV)"); ax[0].legend(loc="lower right")
    prec, rec, _ = precision_recall_curve(y, cal_oof)
    ax[1].plot(rec, prec, color=ORANGE, lw=2, label=f"PR-AUC = {ap:.3f}")
    ax[1].axhline(y.mean(), ls="--", color="grey", lw=1, label=f"baseline={y.mean():.2f}")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax[1].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(C.PLOTS / "roc_pr.png", dpi=140); plt.close(fig)

    # calibration
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    for p, lab, c in [(oof, f"raw (Brier {brier_raw:.3f})", ORANGE),
                      (cal_oof, f"calibrated (Brier {brier_cal:.3f})", GREEN)]:
        fp, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mp, fp, "o-", color=c, label=lab)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction recurring",
           title="Calibration"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(C.PLOTS / "calibration.png", dpi=140); plt.close(fig)

    # confusion
    cm = confusion_matrix(y, (cal_oof >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 4.2)); im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max()/2 else "black", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["new", "recurring"],
           yticklabels=["new", "recurring"], xlabel="Predicted", ylabel="True",
           title=f"Confusion (threshold={thr:.2f})")
    fig.tight_layout(); fig.savefig(C.PLOTS / "confusion_matrix.png", dpi=140); plt.close(fig)

    # top tabular importance
    top = tab_imp.head(15)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top.index, top.values, color=PURPLE)
    ax.set(xlabel="LightGBM importance", title="Top tabular features (final model)")
    fig.tight_layout(); fig.savefig(C.PLOTS / "feature_importance.png", dpi=140); plt.close(fig)

    # contribution bars
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.barh(["Tabular only", "Tabular + SatCLIP"], [auc_tab, auc_comb],
            color=[GREEN, BLUE])
    ax.axvline(0.5, ls=":", color="black", lw=1)
    ax.set(xlabel="ROC-AUC (spatial CV)", xlim=(0.5, max(auc_comb, auc_tab)+0.05),
           title=f"SatCLIP marginal contribution (Δ={auc_comb-auc_tab:+.3f})")
    for i, v in enumerate([auc_tab, auc_comb]):
        ax.text(v+0.002, i, f"{v:.3f}", va="center", fontsize=10)
    fig.tight_layout(); fig.savefig(C.PLOTS / "satclip_contribution.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    main()
