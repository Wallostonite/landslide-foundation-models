"""
Ablation study + diagnostic plots for the landslide-recurrence model.

Ablations (all scored with the SAME spatial cross-validation as train.py):
  * Leave-one-group-out : drop each geologic feature group, measure AUC drop.
  * Single-group-only   : each group alone, to see standalone signal.
  * Leakage demo        : add `Distance to Historical slides` back in, to show
                          how the definitional-leakage column inflates AUC to ~1.0.

Plots written to ./plots/:
  roc_pr_curves.png, confusion_matrix.png, calibration.png,
  feature_importance.png, ablation_leave_one_out.png, ablation_single_group.png
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve,
                             precision_recall_curve, confusion_matrix)
from sklearn.calibration import calibration_curve

from train import (load, spatial_blocks, lgb_params, safe_name, SEED,
                   CSV, NA)

warnings.filterwarnings("ignore")
os.makedirs("plots", exist_ok=True)

# Feature groups defined by ORIGINAL headers (mapped to sanitized names below).
GROUPS = {
    "Terrain":        ["Slope [degrees]", "Aspect [degrees]", "Mean curvature [ft^-1]",
                       "Hillslope position (relative position, 0-1 range)", "Local relief [ft]"],
    "Hydrology":      ["Wetness index [ln(ft^2)]", "Distance to nearest stream [ft]",
                       "Drainage area [ft^2]", "Saturated hydraulic conductivity [micrometer/sec]",
                       "Drainage class [index]"],
    "Mining":         ["Distance to underground mine [ft]", "Depth to underground mine [ft]"],
    "Coal-seam geom": ["Elevation above  Pittsburgh Coal [ft]", "Elevation above Upper Freeport Coal [ft]",
                       "Aspect of Upper Freeport Coal [az. degrees]", "Dip of Upper Freeport Coal [degrees]"],
    "Soil":           ["Clay content [%]", "Erodibility factor [index]"],
    "Infrastructure": ["Distance to nearest road [ft]"],
}


def run_cv(X, y, blocks):
    """Spatial 5-fold CV. Returns (oof_probs, mean_auc, mean_ap, fitted_full_model)."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    aucs, aps = [], []
    for tr, va in cv.split(X, y, groups=blocks):
        m = lgb.LGBMClassifier(**lgb_params(y))
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        aucs.append(roc_auc_score(y.iloc[va], p))
        aps.append(average_precision_score(y.iloc[va], p))
    return oof, float(np.mean(aucs)), float(np.mean(aps)), np.std(aucs)


def main():
    df, X, y, coords, orig_names = load()
    blocks = spatial_blocks(coords)
    groups = {g: [safe_name(c) for c in cols] for g, cols in GROUPS.items()}

    # ---------- Full model: OOF predictions for the diagnostic plots ----------
    oof, full_auc, full_ap, full_sd = run_cv(X, y, blocks)
    print(f"FULL model: ROC-AUC={full_auc:.3f}+/-{full_sd:.3f}  PR-AUC={full_ap:.3f}")

    # ROC + PR curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    fpr, tpr, _ = roc_curve(y, oof)
    ax[0].plot(fpr, tpr, lw=2, color="#2b6cb0", label=f"AUC = {full_auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="ROC curve (spatially cross-validated)")
    ax[0].legend(loc="lower right")
    prec, rec, _ = precision_recall_curve(y, oof)
    base = y.mean()
    ax[1].plot(rec, prec, lw=2, color="#c05621", label=f"PR-AUC = {full_ap:.3f}")
    ax[1].axhline(base, ls="--", color="grey", lw=1, label=f"baseline = {base:.2f}")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="lower left")
    fig.tight_layout(); fig.savefig("plots/roc_pr_curves.png", dpi=140); plt.close(fig)

    # Confusion matrix at the deployed threshold
    import joblib
    thr = joblib.load("landslide_recurrence_model.joblib")["threshold"]
    cm = confusion_matrix(y, (oof >= thr).astype(int))
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v}", ha="center", va="center",
                color="white" if v > cm.max()/2 else "black", fontsize=13)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["new", "recurring"],
           yticklabels=["new", "recurring"], xlabel="Predicted", ylabel="True",
           title=f"Confusion matrix (threshold={thr:.2f})")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig("plots/confusion_matrix.png", dpi=140); plt.close(fig)

    # Calibration curve
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="#2f855a", label="model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction recurring",
           title="Calibration (spatial OOF)")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig("plots/calibration.png", dpi=140); plt.close(fig)

    # Feature importance (final model retrained on all data)
    final = lgb.LGBMClassifier(**lgb_params(y)); final.fit(X, y)
    imp = (pd.Series(final.feature_importances_,
                     index=[orig_names[c] for c in X.columns]).sort_values())
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.barh(imp.index, imp.values, color="#4c51bf")
    ax.set(xlabel="LightGBM gain-split importance", title="Feature importance")
    fig.tight_layout(); fig.savefig("plots/feature_importance.png", dpi=140); plt.close(fig)

    # ---------- Ablations ----------
    all_feats = list(X.columns)
    loo, single = [], []
    for g, cols in groups.items():
        keep = [c for c in all_feats if c not in cols]
        _, a, ap, sd = run_cv(X[keep], y, blocks)
        loo.append({"group": g, "auc": a, "sd": sd, "delta": a - full_auc})
        _, a2, ap2, sd2 = run_cv(X[cols], y, blocks)
        single.append({"group": g, "auc": a2, "sd": sd2})
        print(f"  drop {g:16s}: AUC={a:.3f} (Δ={a-full_auc:+.3f})   "
              f"only {g:16s}: AUC={a2:.3f}")

    loo = pd.DataFrame(loo).sort_values("delta")     # most negative = most important
    single = pd.DataFrame(single).sort_values("auc", ascending=False)

    # Leakage demonstration
    raw = pd.read_csv(CSV, na_values=NA)
    leak = safe_name("Distance to Historical slides [ft]")
    Xleak = X.copy(); Xleak[leak] = pd.to_numeric(raw["Distance to Historical slides [ft]"], errors="coerce")
    _, leak_auc, leak_ap, _ = run_cv(Xleak, y, blocks)
    print(f"\nLEAKAGE demo: adding 'Distance to Historical slides' -> AUC={leak_auc:.3f} "
          f"(vs {full_auc:.3f} without). This column is the label's definition.")

    # Leave-one-group-out plot (AUC drop when a group is removed)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#c53030" if d < 0 else "#2f855a" for d in loo["delta"]]
    ax.barh(loo["group"], loo["delta"], color=colors)
    ax.axvline(0, color="black", lw=1)
    ax.set(xlabel="Change in ROC-AUC when group is REMOVED",
           title=f"Leave-one-group-out ablation (full model AUC={full_auc:.3f})")
    for i, (d, a) in enumerate(zip(loo["delta"], loo["auc"])):
        ax.text(d, i, f" {a:.3f}", va="center",
                ha="left" if d >= 0 else "right", fontsize=9)
    fig.tight_layout(); fig.savefig("plots/ablation_leave_one_out.png", dpi=140); plt.close(fig)

    # Single-group plot (standalone predictive power) with leakage bar for scale
    fig, ax = plt.subplots(figsize=(8, 4.8))
    labels = list(single["group"]) + ["+ Distance-to-historical\n(LEAKAGE)"]
    vals = list(single["auc"]) + [leak_auc]
    bar_colors = ["#4c51bf"] * len(single) + ["#c53030"]
    ax.barh(labels, vals, color=bar_colors)
    ax.axvline(full_auc, ls="--", color="grey", label=f"full model = {full_auc:.3f}")
    ax.axvline(0.5, ls=":", color="black", label="chance = 0.5")
    ax.set(xlabel="ROC-AUC", xlim=(0.45, 1.02),
           title="Standalone group power vs the leakage feature")
    ax.legend(loc="lower right", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig("plots/ablation_single_group.png", dpi=140); plt.close(fig)

    # Save tables
    loo.to_csv("ablation_leave_one_out.csv", index=False)
    single.to_csv("ablation_single_group.csv", index=False)
    print("\nSaved 6 plots to ./plots/ and 2 ablation CSVs.")


if __name__ == "__main__":
    main()
