"""Data loading with explicit leakage removal."""
import re
import pandas as pd
import config as C


def safe_name(col: str) -> str:
    """LightGBM rejects []{}\":, etc. in feature names; make a safe token."""
    return re.sub(r"[^0-9A-Za-z]+", "_", col).strip("_")


def load_raw() -> pd.DataFrame:
    return pd.read_csv(C.DATA_CSV, na_values=C.NA_VALUES)


def load() -> dict:
    """Return the modeling matrices AFTER dropping every leakage/coord column.

    Keys:
      X          : tabular feature DataFrame (sanitized column names)
      y          : target Series (1 = recurring, 0 = new)
      coords     : (n,2) array of [lon, lat] for SatCLIP + spatial CV
      orig_names : {sanitized: original header}
      dropped    : list of columns removed as leakage/coords
    """
    df = load_raw()
    y = df[C.TARGET].astype(int)
    coords = df[C.COORD_COLS].to_numpy()

    drop = C.LEAKAGE_COLS + C.COORD_COLS + [C.TARGET]
    dropped = [c for c in drop if c in df.columns]
    X = df.drop(columns=dropped)

    name_map = {c: safe_name(c) for c in X.columns}
    X = X.rename(columns=name_map)
    orig_names = {v: k for k, v in name_map.items()}
    return dict(X=X, y=y, coords=coords, orig_names=orig_names,
                dropped=[c for c in drop if c in df.columns])
