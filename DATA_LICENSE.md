# Data License and Source Credits

This document records the licensing terms and the full source attribution for
every dataset, satellite product, pretrained model, and software library used in
this project. It complements the code license in [`LICENSE`](LICENSE) (MIT).

## License at a glance

| Component | Covered by | License |
|---|---|---|
| Original source code (this repository) | `LICENSE` | MIT |
| Compiled landslide inventory (`landslides.csv`) and derived tables | this document | CC BY 4.0 |
| Vendored SatCLIP code (`satclip_src/`) | `satclip_src/LICENSE` | MIT (Microsoft) |
| Pretrained model weights (downloaded at runtime) | upstream model cards | see Section 4 |
| Third-party software libraries | upstream projects | see Section 5 |

> The compiled dataset is released by the repository author for research reuse
> under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Several
> records and covariates are derived from public-sector and third-party surveys
> that retain their own terms (Section 2); when you reuse the data, please credit
> both this repository and the original providers listed below.

---

## 1. The dataset

`landslides.csv` is a compiled inventory of 6,129 mapped slope-failure points in
southwestern Pennsylvania, USA, each labeled as a recurring (reactivated) or new
(first-time) slide and attributed with topographic, hydrological, soil, and
coal-seam / mining covariates. A per-record `Data Source` field records the
contributing survey.

**Note on regional labels.** Administrative engineering-region identifiers have
been relabeled as neutral alphabetic survey zones for publication: the regional
column is published as `Zone` with values `Alpha`, `Bravo`, and `Charlie`, and
the corresponding `Data Source` codes are published as `Alpha`, `Bravo`, and
`Charlie`. No other values were altered. These zone columns are treated as
labeling artifacts and are excluded from modeling.

---

## 2. Landslide inventory: source credits

The inventory aggregates mapped slides from multiple public-sector and
institutional surveys, identified in the `Data Source` field:

| Source label | Contributor (as credited) |
|---|---|
| `USGS` | United States Geological Survey landslide mapping |
| `Pitt` | University of Pittsburgh |
| `Nasa` | National Aeronautics and Space Administration |
| `Alpha`, `Bravo`, `Charlie` | State transportation engineering survey zones (administrative region identifiers relabeled with neutral alphabetic codes for publication) |
| `County`, `City`, `Adams`, `ForestHill` | County, municipal, and township survey records |
| `Ackenheil` | Geotechnical consulting records |
| `311` | Municipal public-reporting (311) records |

Credit for the underlying mapped failures belongs to these originating agencies
and organizations. Users redistributing the inventory should preserve the
`Data Source` field and attribute the originators alongside this repository.

### Derived covariates (provenance)

The terrain, hydrological, soil, and coal-seam / mining covariates are values
derived, in the originating study, from standard public geospatial products.
The products below are the conventional sources for each covariate type; confirm
the exact provenance and vintage with the dataset originator before relying on it
for a specific application.

| Covariate group | Typical public product | Provider | Terms |
|---|---|---|---|
| Terrain (slope, aspect, curvature, hillslope position, local relief) | 3D Elevation Program (3DEP) / National Elevation Dataset digital elevation models | U.S. Geological Survey | Public domain (U.S. Government work) |
| Hydrology (wetness index, distance to stream, drainage area, Ksat, drainage class) | Derived from DEM hydrology and soil surveys | USGS / USDA-NRCS | Public domain |
| Soil (clay content, erodibility factor) | SSURGO / gSSURGO soil survey | USDA Natural Resources Conservation Service | Public domain |
| Coal-seam and mining (distance/depth to mine, elevation above and aspect/dip of coal seams) | Mined-area and coal-geology mapping | Pennsylvania DCNR / USGS coal resources | Public-sector; confirm terms |

U.S. Government works (USGS, USDA-NRCS, NASA) are generally not subject to
domestic copyright and are free to use with attribution. State and third-party
(consulting, municipal) records may carry their own terms.

---

## 3. Satellite imagery

Sentinel-2 Level-2A surface-reflectance imagery was used to produce the imagery
foundation-model embeddings.

- **Product:** Copernicus Sentinel-2 Level-2A.
- **Producer:** European Space Agency (ESA) / European Commission, Copernicus
  Programme.
- **Access:** Microsoft Planetary Computer STAC API and signed asset storage.
- **Terms:** Free, full, and open under the Copernicus data policy. Required
  attribution: *"Contains modified Copernicus Sentinel data [2021]."* Access via
  the Planetary Computer is additionally subject to Microsoft's terms of use.

---

## 4. Pretrained foundation models

Model weights are downloaded at runtime from the Hugging Face Hub and are not
redistributed in this repository. Each remains under its own license.

| Model | Identifier | Owner | License |
|---|---|---|---|
| Prithvi-EO-1.0-100M (imagery GeoFM) | `ibm-nasa-geospatial/Prithvi-EO-1.0-100M` | IBM / NASA | Apache-2.0 |
| SatCLIP (location GeoFM) | `microsoft/SatCLIP-ViT16-L40` | Microsoft | MIT |
| TabPFN v2 (tabular FM) | `tabpfn` (PyPI) | Prior Labs | Prior Labs License (Apache-2.0 with an additional provision) |

Refer to each model's card or repository for the authoritative, current license
text before any downstream or commercial use.

---

## 5. Software libraries

This project uses, among others: LightGBM (MIT), scikit-learn (BSD-3-Clause),
SHAP (MIT), pandas and NumPy (BSD-3-Clause), Matplotlib (Matplotlib/PSF-style
license), PyTorch (BSD-style), rasterio (BSD-3-Clause), pystac-client
(Apache-2.0), planetary-computer (MIT), and huggingface_hub (Apache-2.0). Each
is the property of its respective authors under the stated license.

---

## 6. How to cite

If you use this dataset or code, please cite the repository (see
[`CITATION.cff`](CITATION.cff)) and credit the originating agencies in Section 2.
Please also cite the pretrained models you invoke (Prithvi-EO, SatCLIP, TabPFN)
and the Copernicus Sentinel-2 data per Section 3.

## 7. Disclaimer

This dataset and these models are provided for research and educational purposes,
without warranty of any kind. Outputs are susceptibility estimates, not forecasts,
and must not be used as the sole basis for regulatory, engineering, insurance, or
life-safety decisions. Verify all source terms and data provenance before
operational use.
