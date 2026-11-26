"""
Probability calibration for the landslide-recurrence model.

The base LightGBM uses `scale_pos_weight` to fight class imbalance, which makes
its raw scores good for RANKING but inflated as PROBABILITIES. Here we learn an
isotonic mapping from spatially-out-of-fold predictions (so the calibration is
honest, not fit on data the model saw) and attach it to the deployed artifact.

Outputs:
  * updates landslide_recurrence_model.joblib with a 'calibrator' and a
    recomputed F1-optimal threshold on calibrated probabilities
  * plots/calibration_before_after.png
  * prints Brier score before/after (lower = better calibrated)
"""
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, f1_score
from sklearn.calibration import calibration_curve

from train import load, spatial_blocks, lgb_params, SEED

warnings.filterwarnings("ignore")


def main():
    df, X, y, coords, orig_names = load()
    blocks = spatial_blocks(coords)

    # Spatial out-of-fold raw probabilities from the weighted model.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y, groups=blocks):
        m = lgb.LGBMClassifier(**lgb_params(y))
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]

    # Isotonic calibrator fit on OOF (monotonic, non-parametric).
    iso = IsotonicRegression(out_of_bounds="clip")
    cal = iso.fit_transform(oof, y)

    b_raw = brier_score_loss(y, oof)
    b_cal = brier_score_loss(y, cal)
    print(f"Brier score  raw={b_raw:.4f}  calibrated={b_cal:.4f}  "
          f"({100*(b_raw-b_cal)/b_raw:+.1f}% change)")

    # Recompute F1-optimal threshold on CALIBRATED probabilities.
    ths = np.linspace(0.05, 0.95, 91)
    best_t = float(ths[int(np.argmax([f1_score(y, (cal >= t)) for t in ths]))])
    print(f"New F1-optimal threshold on calibrated probs: {best_t:.2f}")

    # Before/after reliability diagram.
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    for probs, lab, c in [(oof, f"raw (Brier {b_raw:.3f})", "#c05621"),
                          (cal, f"calibrated (Brier {b_cal:.3f})", "#2f855a")]:
        fp, mp = calibration_curve(y, probs, n_bins=10, strategy="quantile")
        ax.plot(mp, fp, "o-", color=c, label=lab)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed fraction recurring",
           title="Reliability diagram: before vs after calibration")
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig("plots/calibration_before_after.png", dpi=140)
    plt.close(fig)

    # Retrain final model on all data, attach calibrator, update artifact.
    final = lgb.LGBMClassifier(**lgb_params(y)); final.fit(X, y)
    art = joblib.load("landslide_recurrence_model.joblib")
    art.update({"model": final, "calibrator": iso, "threshold": best_t,
                "brier_raw": float(b_raw), "brier_calibrated": float(b_cal)})
    joblib.dump(art, "landslide_recurrence_model.joblib")
    print("Updated landslide_recurrence_model.joblib with calibrator + new threshold.")
    print("Saved plots/calibration_before_after.png")


if __name__ == "__main__":
    main()
