"""
SHAP + partial-dependence view: which tabular features still dominate once the
256 SatCLIP location dimensions are in the model.

Outputs (outputs/plots/):
  shap_tabular_vs_satclip.png  -- mean|SHAP| of each tabular feature next to the
                                  SatCLIP block (summed over its 256 dims)
  shap_beeswarm_tabular.png    -- SHAP beeswarm for the top tabular features
  pdp_tabular_drivers.png      -- partial dependence of the top tabular drivers
And outputs/shap_importance.csv (mean|SHAP| per feature, tabular + SatCLIP block).
"""
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay

import config as C
from data import load
from satclip_features import get_or_build
from features import build_matrix, satclip_columns

warnings.filterwarnings("ignore")
PURPLE, CLAY, BLUE = "#4c51bf", "#c05621", "#2b6cb0"


def main():
    d = load()
    X_tab, orig = d["X"], d["orig_names"]
    emb = get_or_build(d["coords"])
    X = build_matrix(X_tab, emb)
    sat_cols = satclip_columns(emb.shape[1])
    tab_cols = list(X_tab.columns)

    art = joblib.load(C.MODEL_FILE)
    model = art["model"]
    X = X[list(model.feature_name_)]

    print("Computing SHAP values (TreeExplainer)...")
    expl = shap.TreeExplainer(model)
    sv = expl.shap_values(X)
    if isinstance(sv, list):        # some versions return [class0, class1]
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)
    imp = pd.Series(mean_abs, index=X.columns)

    tab_imp = imp[tab_cols].rename(index=orig).sort_values(ascending=False)
    sat_total = imp[sat_cols].sum()
    print(f"SatCLIP block mean|SHAP| (sum of 256 dims): {sat_total:.4f}")
    print(f"Tabular block mean|SHAP| (sum of 19):       {tab_imp.sum():.4f}")
    print("\nTop tabular features by mean|SHAP|:")
    print(tab_imp.head(10).to_string())

    # ---- plot 1: tabular features vs the SatCLIP block ----
    top = tab_imp.head(12)
    labels = list(top.index)[::-1] + ["SatCLIP block (256 dims)"]
    vals = list(top.values)[::-1] + [sat_total]
    colors = [PURPLE] * len(top) + [CLAY]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.barh(labels, vals, color=colors)
    ax.set(xlabel="mean |SHAP| (impact on model output)",
           title="Tabular drivers vs the SatCLIP block, with SatCLIP included")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8.5)
    fig.tight_layout(); fig.savefig(C.PLOTS / "shap_tabular_vs_satclip.png", dpi=140)
    plt.close(fig)

    # ---- plot 2: beeswarm for the top tabular features ----
    topk = tab_imp.head(10).index
    topk_safe = [k for k in tab_cols if orig[k] in set(topk)]
    order = sorted(topk_safe, key=lambda s: -imp[s])
    sv_sub = sv[:, [X.columns.get_loc(s) for s in order]]
    Xsub = X[order].copy(); Xsub.columns = [orig[s] for s in order]
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv_sub, Xsub, show=False, plot_size=None, max_display=10)
    plt.title("SHAP beeswarm: top tabular features (SatCLIP in the model)", fontsize=11)
    plt.tight_layout(); plt.savefig(C.PLOTS / "shap_beeswarm_tabular.png", dpi=140)
    plt.close()

    # ---- plot 3: partial dependence of the top tabular drivers ----
    pdp_feats = [s for s in order[:6]]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    PartialDependenceDisplay.from_estimator(
        model, X, pdp_feats, ax=axes.ravel(), kind="average",
        response_method="predict_proba", grid_resolution=40,
        line_kw={"color": BLUE, "lw": 2})
    for ax, s in zip(axes.ravel(), pdp_feats):
        ax.set_title(orig[s], fontsize=9); ax.set_ylabel("P(recurring)")
    fig.suptitle("Partial dependence of top tabular drivers (SatCLIP held at observed)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(C.PLOTS / "pdp_tabular_drivers.png", dpi=140); plt.close(fig)

    # ---- save table ----
    out = tab_imp.to_frame("mean_abs_shap")
    out.loc["SatCLIP block (256 dims)"] = sat_total
    out.to_csv(C.OUT / "shap_importance.csv")
    print("\nSaved 3 plots + shap_importance.csv")


if __name__ == "__main__":
    main()
