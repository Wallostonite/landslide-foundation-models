"""
Partial-dependence (PDP) + ICE report: how each driver pushes recurrence risk.

PDP shows the model's average predicted recurrence probability as one feature is
swept across its range, holding the rest at their observed values. ICE lines show
the same per-sample, revealing whether the effect is uniform or interacts.

Outputs:
  plots/pdp_top_drivers.png   -- 6-panel grid of the strongest drivers
  plots/pdp_requested_ice.png -- coal-seam dip, distance to mine, local relief with ICE
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.inspection import PartialDependenceDisplay

from train import load, lgb_params, safe_name

warnings.filterwarnings("ignore")

# Requested drivers first, then other top-importance features.
REQUESTED = ["Dip of Upper Freeport Coal [degrees]",
             "Distance to underground mine [ft]",
             "Local relief [ft]"]
TOP6 = REQUESTED + ["Aspect of Upper Freeport Coal [az. degrees]",
                    "Elevation above Upper Freeport Coal [ft]",
                    "Distance to nearest stream [ft]"]


def main():
    df, X, y, coords, orig_names = load()
    model = lgb.LGBMClassifier(**lgb_params(y)); model.fit(X, y)
    safe2orig = orig_names
    orig2safe = {v: k for k, v in safe2orig.items()}

    # ---- 6-panel PDP grid of the strongest drivers ----
    feats6 = [orig2safe[c] for c in TOP6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    PartialDependenceDisplay.from_estimator(
        model, X, feats6, ax=axes.ravel(), kind="average",
        response_method="predict_proba", grid_resolution=40,
        line_kw={"color": "#2b6cb0", "lw": 2})
    for ax, col in zip(axes.ravel(), TOP6):
        ax.set_title(col, fontsize=9)
        ax.set_ylabel("P(recurring)")
    fig.suptitle("Partial dependence: how each driver moves recurrence probability",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig("plots/pdp_top_drivers.png", dpi=140); plt.close(fig)

    # ---- Requested three with ICE overlay ----
    feats3 = [orig2safe[c] for c in REQUESTED]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    PartialDependenceDisplay.from_estimator(
        model, X, feats3, ax=axes, kind="both",
        response_method="predict_proba", grid_resolution=40, subsample=200,
        ice_lines_kw={"color": "#a0aec0", "alpha": 0.15},
        pd_line_kw={"color": "#c53030", "lw": 3})
    for ax, col in zip(axes, REQUESTED):
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("P(recurring)")
    fig.suptitle("Requested drivers with ICE (grey) and average PDP (red)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig("plots/pdp_requested_ice.png", dpi=140); plt.close(fig)

    # ---- Numeric direction summary: mean P at low vs high of each driver ----
    print("Direction of effect (mean predicted P across the sample):")
    for col in TOP6:
        s = orig2safe[col]
        vals = X[s].dropna()
        lo, hi = vals.quantile(0.1), vals.quantile(0.9)
        Xlo = X.copy(); Xlo[s] = lo
        Xhi = X.copy(); Xhi[s] = hi
        plo = model.predict_proba(Xlo)[:, 1].mean()
        phi = model.predict_proba(Xhi)[:, 1].mean()
        arrow = "UP" if phi > plo else "DOWN"
        print(f"  {col:48s}  low={plo:.3f} -> high={phi:.3f}  risk {arrow} "
              f"({phi-plo:+.3f})")
    print("\nSaved plots/pdp_top_drivers.png and plots/pdp_requested_ice.png")


if __name__ == "__main__":
    main()
