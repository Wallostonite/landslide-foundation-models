"""
SatCLIP location-embedding foundation model for landslide points.

SatCLIP is a geospatial foundation model whose location encoder maps (lon, lat)
to a 256-dim embedding pretrained (contrastively) against Sentinel-2 imagery
worldwide. We embed the SAME points that Prithvi embedded, so the two GeoFMs are
compared head-to-head on identical rows (see geofm_compare.py).

Caveat: SatCLIP is a GLOBAL location encoder. Across this ~100 km study area its
resolution is coarse, so the embeddings mostly encode broad position. We report
its score honestly rather than assuming it helps.
"""
import os, sys, warnings
import numpy as np
import torch
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "satclip_src"))
from load_lightweight import get_satclip_loc_encoder  # noqa: E402

CKPT_REPO = "microsoft/SatCLIP-ViT16-L40"
CKPT_FILE = "satclip-vit16-l40.ckpt"


def main():
    # Use exactly the points Prithvi succeeded on, for a fair head-to-head.
    ref = np.load("prithvi_embeddings.npz")
    idx, lon, lat, y = ref["idx"], ref["lon"], ref["lat"], ref["y"]
    print(f"Embedding {len(idx)} points with SatCLIP...")

    ckpt = hf_hub_download(CKPT_REPO, CKPT_FILE)
    enc = get_satclip_loc_encoder(ckpt, "cpu")

    coords = torch.tensor(np.stack([lon, lat], axis=1), dtype=torch.double)
    with torch.no_grad():
        emb = enc(coords).float().numpy()
    print(f"SatCLIP embeddings: {emb.shape}")

    np.savez("satclip_embeddings.npz", emb=emb, idx=idx, lon=lon, lat=lat, y=y)
    print("Saved satclip_embeddings.npz")


if __name__ == "__main__":
    main()
