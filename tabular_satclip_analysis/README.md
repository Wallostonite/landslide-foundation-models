# Tabular + SatCLIP: Landslide Recurrence Model

A self-contained analysis that predicts whether a slope location is a **recurring**
slide (reactivated old ground) or a **new** slide, from engineered terrain/geologic
features **fused with SatCLIP geospatial-foundation-model location embeddings**.
Trained on 6,129 mapped points in undermined southwestern Pennsylvania.

## Result (spatial 5-fold cross-validation)

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Tabular only (19 features) | 0.794 ± 0.093 | — |
| **Tabular + SatCLIP (256-d)** | **0.808 ± 0.083** | 0.864 |

SatCLIP adds **+0.014 AUC** and improves calibration (Brier **0.159 → 0.132**).
At the F1-optimal threshold (0.40): recall 0.94 on recurring, balanced accuracy 0.69.

> **Honest note on SatCLIP.** SatCLIP is a *global location encoder*: its embedding
> is a smooth function of (lon, lat). Across this ~100 km area it mostly re-encodes
> spatial position, which the tabular model deliberately excludes (coordinates are
> dropped) to avoid spatial memorization. So the +0.014 gain is real but should be
> read as "residual spatial/environmental context," not a new physical driver.
> Evaluation uses spatial block CV, which limits (but does not fully remove) that
> location signal.

## Leakage columns dropped before any modeling

Removed in `src/config.py` (`LEAKAGE_COLS`), never seen by the model:

| Column | Why |
|---|---|
| `Distance to Historical slides [ft]` | **Definitional leakage** — "recurring" means "on a past slide"; this separates the classes almost perfectly (AUC → 1.000). |
| `Data Source` | Labeling artifact (USGS points are 81% recurring). |
| `Zone` | Administrative / spatial proxy (Zone Charlie is 89% recurring). |
| `ID` | Row identifier. |
| `Lon (Degree)`, `Lat (Degree)` | Not tabular features. Used only to build SatCLIP embeddings and spatial CV blocks. |

## Project structure
```
tabular_satclip_analysis/
├── data/landslides.csv           # input data
├── satclip_src/                  # vendored SatCLIP location-encoder loader
├── src/
│   ├── config.py                 # paths, columns, LEAKAGE policy, hyperparams
│   ├── data.py                   # load + drop leakage cols
│   ├── satclip_features.py       # (lon,lat) -> 256-d SatCLIP embeddings (cached)
│   ├── features.py               # assemble Tabular + SatCLIP matrix
│   ├── model.py                  # spatial blocks, LightGBM, spatial-CV, calibration
│   ├── train.py                  # main: evaluate, calibrate, save model + plots
│   ├── shap_analysis.py          # SHAP + partial-dependence of tabular features
│   └── predict.py                # score new locations
├── outputs/                      # model, metrics, embeddings, plots (generated)
└── requirements.txt
```

## Usage
```bash
cd src
python train.py                       # trains, evaluates, writes outputs/
python shap_analysis.py               # SHAP + partial-dependence plots
python predict.py new_sites.csv -o outputs/predictions.csv
```
See [REPORT.md](REPORT.md) for the full write-up (leakage policy, results,
SHAP interpretation, honest SatCLIP caveat).
`new_sites.csv` needs the tabular headers (same as `landslides.csv`) plus
`Lon (Degree)` / `Lat (Degree)`. Output adds calibrated `recurrence_probability`
and `recurrence_pred` (threshold 0.40).

## Method
- **Model:** LightGBM gradient-boosted trees (handles the 71%-missing mine-depth
  column natively), `scale_pos_weight` for the 71/29 class imbalance.
- **Validation:** stratified group K-fold over 12 KMeans spatial blocks, so whole
  regions are held out. This is the honest estimate of skill at unseen locations.
- **Calibration:** isotonic regression fit on spatial out-of-fold scores, so
  `recurrence_probability` reads as a real risk level.
- **Features:** 19 engineered terrain/geologic covariates + 256 SatCLIP
  (`microsoft/SatCLIP-ViT16-L40`) location embedding dimensions = 275 total.

## Outputs (after `train.py`)
`outputs/tabular_satclip_model.joblib` (model + calibrator + threshold),
`outputs/metrics.json`, `outputs/feature_importance.csv`,
`outputs/satclip_embeddings.npz`, and `outputs/plots/` (roc_pr, calibration,
confusion_matrix, feature_importance, satclip_contribution).

Top tabular drivers remain terrain and coal-seam geometry: local relief, distance
to stream, aspect, Upper Freeport dip, hillslope position, distance to mine.
