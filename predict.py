"""
Score new locations for landslide-recurrence risk.

Usage:
    python predict.py INPUT.csv [-o OUTPUT.csv] [-t THRESHOLD]

INPUT.csv must contain the physical/geologic feature columns used in training
(same headers as landslides.csv). Extra columns (ID, Lon, Lat, Data Source,
etc.) are ignored and passed through to the output. Missing feature columns are
filled with NaN, which LightGBM handles natively.

Output adds two columns:
    recurrence_probability  -- model probability the slide is recurring (0..1)
    recurrence_pred         -- 1 if probability >= threshold else 0
"""
import argparse
import numpy as np
import pandas as pd
import joblib

MODEL_FILE = "landslide_recurrence_model.joblib"
NA = ["<Null>", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="CSV of locations to score")
    ap.add_argument("-o", "--output", default="predictions.csv")
    ap.add_argument("-t", "--threshold", type=float, default=None,
                    help="override the F1-optimal threshold saved with the model")
    args = ap.parse_args()

    art = joblib.load(MODEL_FILE)
    model = art["model"]
    feats = art["features"]                 # sanitized names in model order
    orig_names = art["orig_names"]          # sanitized -> original header
    thr = args.threshold if args.threshold is not None else art["threshold"]

    raw = pd.read_csv(args.input, na_values=NA)

    # Build the feature matrix using the ORIGINAL headers, then rename to the
    # sanitized names the model expects. Warn about anything missing.
    X = pd.DataFrame(index=raw.index)
    missing = []
    for safe in feats:
        col = orig_names[safe]
        if col in raw.columns:
            X[safe] = pd.to_numeric(raw[col], errors="coerce")
        else:
            X[safe] = np.nan
            missing.append(col)
    if missing:
        print(f"WARNING: {len(missing)} feature column(s) missing, filled with NaN:")
        for c in missing:
            print(f"  - {c}")

    proba = model.predict_proba(X[feats])[:, 1]
    # Apply isotonic calibration if the artifact carries one, so the reported
    # probability is a trustworthy risk estimate rather than an inflated score.
    if art.get("calibrator") is not None:
        proba = art["calibrator"].predict(proba)

    out = raw.copy()
    out["recurrence_probability"] = np.round(proba, 4)
    out["recurrence_pred"] = (proba >= thr).astype(int)
    out.to_csv(args.output, index=False)

    n = len(out)
    npos = int(out["recurrence_pred"].sum())
    print(f"Scored {n} rows (threshold={thr:.2f}). "
          f"Predicted recurring: {npos} ({npos/n:.0%}).  Wrote {args.output}")


if __name__ == "__main__":
    main()
