"""Assemble the combined Tabular + SatCLIP feature matrix."""
import numpy as np
import pandas as pd


def satclip_columns(n_dims: int):
    return [f"satclip_{i:03d}" for i in range(n_dims)]


def build_matrix(X_tab: pd.DataFrame, emb: np.ndarray) -> pd.DataFrame:
    """Concatenate tabular features with SatCLIP embedding dims (aligned by row)."""
    assert len(X_tab) == len(emb), "row mismatch between tabular and embeddings"
    E = pd.DataFrame(emb, columns=satclip_columns(emb.shape[1]), index=X_tab.index)
    return pd.concat([X_tab, E], axis=1)
