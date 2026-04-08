# CONUS Thunderstorm & Lightning Prediction (HRRR + GOES-16 GLM)

## Overview
This repository contains the data pipeline and research workflow for building a **CONUS-scale thunderstorm prediction system** using:
- **NOAA HRRR** numerical weather prediction output (native coordinates)
- **GOES-16 GLM (Geostationary Lightning Mapper)** lightning observations as the target signal

The goal is to learn a mapping from **ambient environmental conditions** (from HRRR) to **thunderstorm occurrence** (from GLM lightning) over the contiguous United States (CONUS).

---

## Problem Statement
We aim to predict whether a thunderstorm will occur within fixed geographic regions across CONUS.

- **Target:** Binary thunderstorm occurrence (0/1)
- **Thunderstorm definition:** A region is labeled **1** if **≥ 1 GLM lightning group** occurs inside the region **within ±1 hour** of the prediction valid time; otherwise **0**.
- **Spatial unit:** **24 km × 24 km** boxes (chosen to capture mesoscale heterogeneity-induced circulations)
- **Lead times:** **12, 18, 24, 36 hours**
- **Forecast cycles:** **00 UTC, 06 UTC**
- **Temporal feature sampling:** **Pt, Pt−3 hr, Pt−6 hr**

---

## Why 24 km × 24 km Boxes?
A key scientific motivation is to capture **heterogeneity-induced circulations** that contribute to convective initiation. These circulations require length scales on the order of **~20 km or more** to survive atmospheric turbulence and produce convergence capable of lifting parcels to the LFC (see Avissar & Liu 1996; Taylor et al. 2011; Dirmeyer et al. 2024).

Using 24 km × 24 km regions is **not reducing resolution**, but increasing the **prediction domain size** to include physically critical mesoscale processes.

---

## Data Sources

### 1) HRRR (Predictors)
- **Model:** High-Resolution Rapid Refresh (HRRR), CONUS, 3-km grid
- **Download type:** **Native hybrid-sigma output (`wrfnat`) only**
- **Explicit exclusions:** No pressure-level or normalized-pressure products
- **Cycles:** 00, 06 UTC
- **Lead times:** 12, 18, 24, 36 hr
- **Years:** 2018–2025 (downloaded year-by-year)
- **Source:** NOAA HRRR archive (AWS)

### 2) GOES-16 GLM L2 LCFA (Targets)
- **Product:** GLM-L2-LCFA
- **Hierarchy:** Events → Groups → Flashes
- **Target unit:** **Groups**
- **Labeling window:** **±1 hour**
- **Source:** Google Cloud public dataset (`gcp-public-data-goes-16/GLM-L2-LCFA`)

---

## HRRR Variables to Download (Native Only)
All variables are sampled at **Pt**, **Pt−3 hr**, and **Pt−6 hr**.

### Atmospheric State (Hybrid-Sigma Levels)
- Air temperature
- Specific humidity
- Relative humidity
- Model-level pressure
- Geopotential height
- Zonal wind (u)
- Meridional wind (v)
- Vertical velocity (w)

### Near-Surface & Boundary Layer
- 2-m air temperature
- 2-m dewpoint temperature
- 10-m zonal wind
- 10-m meridional wind
- Planetary boundary layer height (PBLH)
- Surface sensible heat flux
- Surface latent heat flux

### Moisture & Column Variables
- Precipitable water
- Total column water vapor

### Cloud & Microphysics
- Cloud water mixing ratio
- Cloud ice mixing ratio
- Rain water mixing ratio
- Snow mixing ratio
- Graupel mixing ratio

### Land-Surface Context
- Soil moisture (top layer)
- Soil temperature
- Vegetation fraction
- Terrain height

### Explicit Exclusions
- No pressure-level HRRR data (`wrfprs`)
- No normalized-pressure coordinates
- No pre-computed diagnostics (CAPE, CIN, shear, helicity)
- Diagnostics will be derived later from native variables

---

## Repository Structure (Suggested)
```text
.
├── scripts/
│   ├── download_hrrr/            # HRRR native download scripts
│   ├── download_glm/             # GLM download scripts
│   ├── preprocess_glm/           # GLM parsing + labeling
│   ├── preprocess_hrrr/          # HRRR extraction + aggregation
│   └── utils/                    # shared utilities
├── data/
│   ├── hrrr/                     # HRRR raw (wrfnat)
│   ├── glm/                      # GLM raw (NetCDF)
│   ├── labels/                   # binary target labels
│   └── features/                 # ML-ready feature tables
├── notebooks/                    # exploration, QA, plots
├── docs/                         # project notes, meeting notes
└── README.md
