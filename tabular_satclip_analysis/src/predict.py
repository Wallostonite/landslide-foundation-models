"""
Score new locations with the Tabular + SatCLIP model.

    python predict.py INPUT.csv [-o OUTPUT.csv] [-t THRESHOLD]

INPUT.csv must contain the tabular feature headers (same as landslides.csv) plus
Lon/Lat (needed to build SatCLIP embeddings). Output adds calibrated
`recurrence_probability` and `recurrence_pred`.
"""
import argparse, warnings
import numpy as np
import pandas as pd
import joblib

import config as C
from data import safe_name
from features import build_matrix
from satclip_features import compute as satclip_compute

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("-o", "--output", default=str(C.OUT / "predictions.csv"))
    ap.add_argument("-t", "--threshold", type=float, default=None)
    args = ap.parse_args()

    art = joblib.load(C.MODEL_FILE)
    model, cal = art["model"], art["calibrator"]
    thr = args.threshold if args.threshold is not None else art["threshold"]
    tab_cols = art["tabular_features"]           # sanitized names
    orig = art["orig_names"]

    raw = pd.read_csv(args.input, na_values=C.NA_VALUES)

    # tabular block, by original headers -> sanitized
    Xtab = pd.DataFrame(index=raw.index)
    for safe in tab_cols:
        col = orig[safe]
        Xtab[safe] = pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else np.nan

    # SatCLIP block from Lon/Lat
    coords = raw[[C.LON, C.LAT]].to_numpy(dtype=float)
    emb = satclip_compute(coords)
    X = build_matrix(Xtab, emb)[list(model.feature_name_)]

    proba = cal.predict(model.predict_proba(X)[:, 1])
    out = raw.copy()
    out["recurrence_probability"] = np.round(proba, 4)
    out["recurrence_pred"] = (proba >= thr).astype(int)
    out.to_csv(args.output, index=False)
    n, npos = len(out), int(out["recurrence_pred"].sum())
    print(f"Scored {n} rows (threshold={thr:.2f}). Recurring: {npos} ({npos/n:.0%}). "
          f"Wrote {args.output}")


if __name__ == "__main__":
    main()
