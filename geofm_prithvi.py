"""
Geospatial foundation model (Prithvi-EO-1.0-100M) embeddings for landslide points.

For each sampled point we pull a Sentinel-2 L2A image chip (6 HLS-equivalent
bands) from Microsoft Planetary Computer, run it through the pretrained Prithvi
encoder, and mean-pool the patch tokens into a 768-dim embedding. Those
embeddings become features for a LightGBM classifier, benchmarked against the
tabular baseline on the same spatial folds (see geofm_compare.py).

Usage:  python geofm_prithvi.py --n 700
        python geofm_prithvi.py --test        # tiny run to validate the path
"""
import argparse, warnings, time, sys
import numpy as np, pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform as rio_transform
from rasterio.enums import Resampling
from pystac_client import Client
import planetary_computer as pc
import torch, importlib.util
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

REPO = "ibm-nasa-geospatial/Prithvi-EO-1.0-100M"
# Sentinel-2 L2A bands mapped to Prithvi's HLS bands [B02,B03,B04,B05,B06,B07]
S2_BANDS = ["B02", "B03", "B04", "B8A", "B11", "B12"]
MEAN = np.array([775.23, 1080.99, 1228.59, 2497.20, 2204.21, 1610.83], dtype="float32")
STD = np.array([1281.53, 1270.03, 1399.48, 1368.34, 1291.68, 1154.51], dtype="float32")
SIZE = 224
HALF_M = 1120.0          # chip covers 2.24 km, matching 224 px at 30 m HLS scale
DATE = "2021-05-01/2021-09-30"
SEED = 42


def load_points():
    df = pd.read_csv("landslides.csv", na_values=["<Null>", ""])
    return df


def choose_subsample(df, n):
    rng = np.random.default_rng(SEED)
    y = df["Reccuring=1 New-slides=0"].values
    idx = np.arange(len(df))
    # stratified by class to keep the 71/29 balance
    pos = idx[y == 1]; neg = idx[y == 0]
    npos = int(round(n * len(pos) / len(idx))); nneg = n - npos
    take = np.concatenate([rng.choice(pos, min(npos, len(pos)), replace=False),
                           rng.choice(neg, min(nneg, len(neg)), replace=False)])
    take.sort()
    return take


def best_scene_per_point(cat, lons, lats):
    """Search the bbox, pick the least-cloudy scene covering each point."""
    from shapely.geometry import shape, Point
    pad = 0.05
    bbox = [lons.min()-pad, lats.min()-pad, lons.max()+pad, lats.max()+pad]
    search = cat.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=DATE,
                        query={"eo:cloud_cover": {"lt": 15}},
                        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
                        max_items=400)
    items = list(search.items())
    print(f"  {len(items)} candidate scenes in bbox")
    geoms = [(it, shape(it.geometry), it.properties["eo:cloud_cover"]) for it in items]
    assign = [None] * len(lons)
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        p = Point(lon, lat)
        for it, g, cc in geoms:              # geoms already sorted by cloud asc
            if g.contains(p):
                assign[i] = it
                break
    covered = sum(a is not None for a in assign)
    print(f"  {covered}/{len(lons)} points covered by a scene")
    return assign


def read_chip(handles, lon, lat):
    """Read a 224x224x6 chip centered on the point from cached band datasets."""
    chip = np.zeros((6, SIZE, SIZE), dtype="float32")
    for b in range(6):
        src = handles[b]
        xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
        x, y = xs[0], ys[0]
        win = from_bounds(x-HALF_M, y-HALF_M, x+HALF_M, y+HALF_M, src.transform)
        chip[b] = src.read(1, window=win, out_shape=(SIZE, SIZE), boundless=True,
                           fill_value=0, resampling=Resampling.bilinear).astype("float32")
    return chip


def normalize(chip):
    return (chip - MEAN[:, None, None]) / STD[:, None, None]


def load_prithvi():
    mae = hf_hub_download(REPO, "prithvi_mae.py")
    wts = hf_hub_download(REPO, "Prithvi_EO_V1_100M.pt")
    spec = importlib.util.spec_from_file_location("prithvi_mae", mae)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    enc = mod.PrithviViT(img_size=SIZE, patch_size=[1, 16, 16], num_frames=3,
                         in_chans=6, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0)
    sd = torch.load(wts, map_location="cpu", weights_only=False)
    enc.load_state_dict({k[len("encoder."):]: v for k, v in sd.items()
                         if k.startswith("encoder.")}, strict=False)
    enc.eval()
    return enc


@torch.no_grad()
def embed(enc, chips):
    """chips: (N,6,224,224) normalized -> (N,768) mean-pooled patch tokens."""
    embs = []
    for i in range(0, len(chips), 8):
        batch = torch.from_numpy(chips[i:i+8]).float()
        batch = batch.unsqueeze(2).repeat(1, 1, 3, 1, 1)   # add T=3 (replicate date)
        out = enc(batch, mask_ratio=0.0)
        tok = out[0] if isinstance(out, (tuple, list)) else out
        embs.append(tok[:, 1:, :].mean(1).cpu().numpy())   # drop CLS, mean-pool
    return np.concatenate(embs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=700)
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    n = 24 if args.test else args.n

    df = load_points()
    take = choose_subsample(df, n)
    lons = df["Lon (Degree)"].values[take]; lats = df["Lat (Degree)"].values[take]
    y = df["Reccuring=1 New-slides=0"].values[take]
    print(f"Subsample: {len(take)} points ({int(y.sum())} recurring)")

    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                      modifier=pc.sign_inplace)
    print("Assigning scenes...")
    assign = best_scene_per_point(cat, lons, lats)

    # Group points by scene id so each scene's COGs are opened once.
    from collections import defaultdict
    groups = defaultdict(list)
    for i, it in enumerate(assign):
        if it is not None:
            groups[it.id].append(i)

    chips = np.zeros((len(take), 6, SIZE, SIZE), dtype="float32")
    ok = np.zeros(len(take), dtype=bool)
    t0 = time.time()
    done = 0
    for sid, members in groups.items():
        it = assign[members[0]]
        try:
            handles = [rasterio.open(it.assets[b].href) for b in S2_BANDS]
        except Exception as e:
            print(f"  scene {sid} open failed: {e}"); continue
        for i in members:
            try:
                chips[i] = normalize(read_chip(handles, lons[i], lats[i]))
                ok[i] = True
            except Exception:
                pass
            done += 1
            if done % 50 == 0:
                print(f"  chips {done}/{len(take)}  ({time.time()-t0:.0f}s)")
        for h in handles:
            h.close()
    print(f"Harvested {ok.sum()} chips in {time.time()-t0:.0f}s")

    print("Loading Prithvi + embedding...")
    enc = load_prithvi()
    emb = embed(enc, chips[ok])
    print(f"Embeddings: {emb.shape}")

    np.savez("prithvi_embeddings.npz", emb=emb, idx=take[ok], lon=lons[ok],
             lat=lats[ok], y=y[ok])
    print("Saved prithvi_embeddings.npz")


if __name__ == "__main__":
    main()
