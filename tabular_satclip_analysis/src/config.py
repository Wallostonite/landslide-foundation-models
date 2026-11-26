"""Central configuration: paths, columns, and the leakage policy."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "landslides.csv"
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
SATCLIP_SRC = ROOT / "satclip_src"

NA_VALUES = ["<Null>", ""]
TARGET = "Reccuring=1 New-slides=0"
LON, LAT = "Lon (Degree)", "Lat (Degree)"

# ---------------------------------------------------------------------------
# LEAKAGE / ARTIFACT COLUMNS -- dropped before any modeling.
#   Distance to Historical slides : definitional leakage. "Recurring" means "on
#                                   a past slide", so this column almost perfectly
#                                   separates the classes (AUC -> 1.000).
#   Data Source                   : labeling artifact (USGS points are 81% recurring).
#   Zone                   : administrative / spatial proxy (Zone Charlie = 89% recurring).
#   ID                            : row identifier.
# Lon/Lat are NOT tabular features either: they are used to generate SatCLIP
# location embeddings and to build spatial cross-validation blocks.
# ---------------------------------------------------------------------------
LEAKAGE_COLS = [
    "Distance to Historical slides [ft]",
    "Data Source",
    "Zone",
    "ID",
]
COORD_COLS = [LON, LAT]

# Modeling
SEED = 42
N_SPATIAL_BLOCKS = 12
N_SPLITS = 5

# SatCLIP
SATCLIP_REPO = "microsoft/SatCLIP-ViT16-L40"
SATCLIP_CKPT = "satclip-vit16-l40.ckpt"
SATCLIP_EMB = OUT / "satclip_embeddings.npz"

MODEL_FILE = OUT / "tabular_satclip_model.joblib"
METRICS_FILE = OUT / "metrics.json"
