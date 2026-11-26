"""Compute SatCLIP location embeddings for every point (cached to outputs/)."""
import sys, warnings
import numpy as np
import pandas as pd
import config as C

warnings.filterwarnings("ignore")


def compute(coords: np.ndarray) -> np.ndarray:
    """coords: (n,2) [lon,lat] degrees -> (n,256) SatCLIP location embeddings."""
    import torch
    from huggingface_hub import hf_hub_download
    sys.path.insert(0, str(C.SATCLIP_SRC))
    from load_lightweight import get_satclip_loc_encoder

    ckpt = hf_hub_download(C.SATCLIP_REPO, C.SATCLIP_CKPT)
    enc = get_satclip_loc_encoder(ckpt, "cpu")
    t = torch.tensor(coords, dtype=torch.double)      # SatCLIP expects [lon,lat]
    with torch.no_grad():
        emb = enc(t).float().numpy()
    return emb


def get_or_build(coords: np.ndarray) -> np.ndarray:
    """Load cached embeddings if the row count matches, else compute and cache."""
    if C.SATCLIP_EMB.exists():
        cached = np.load(C.SATCLIP_EMB)
        if cached["emb"].shape[0] == coords.shape[0]:
            return cached["emb"]
    emb = compute(coords)
    C.SATCLIP_EMB.parent.mkdir(parents=True, exist_ok=True)
    np.savez(C.SATCLIP_EMB, emb=emb, coords=coords)
    return emb


if __name__ == "__main__":
    from data import load
    d = load()
    emb = get_or_build(d["coords"])
    print(f"SatCLIP embeddings: {emb.shape}  saved to {C.SATCLIP_EMB}")
