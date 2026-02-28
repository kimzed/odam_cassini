# oDam — Open Diplomatic Aquatic Modeling

Satellite-driven water management platform for transboundary river basins, built at the Cassini Hackathon.

## Overview

oDam uses Copernicus satellite data and machine learning to predict water quality indicators and provide 2-week drought/flood forecasts for transboundary river basins. The platform addresses hydropolitical tensions by providing data-driven insights for water security and cooperation between upstream and downstream communities.

## Tech Stack

- **ML:** scikit-learn (Random Forest), joblib
- **Satellite Data:** Google Earth Engine API (NASA GDDP-CMIP6, ECMWF ERA5_LAND)
- **Data Processing:** Pandas, GeoPandas, Rasterio
- **Frontend:** Streamlit, Plotly, Altair
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Conda

## Features

- **Quality Indicator Prediction** — Random Forest model trained on historical hydrological data with 30-day lagged precipitation features, GridSearchCV-optimized
- **2-Week Forecasting** — Early warning system for drought and flood risk based on normalized downstream flux predictions
- **Copernicus Data Integration** — Automated retrieval of precipitation, temperature, and evapotranspiration via Google Earth Engine
- **Interactive Dashboard** — Streamlit multi-page app with:
  - Dam Explorer: location selection, early warning display, annual water management plan
  - River Explorer: multi-dam views by river system
  - Analytics: 4-panel time series (precipitation, evapotranspiration, temperature, water flows)

## Project Structure

```
odam_cassini/
├── download_qi_copernicus_data.py   # Google Earth Engine data fetching
├── ml_qi_prediction_training.py     # Random Forest model training
├── ml_qi_inference.py               # Model inference
├── two_weeks_forecasting.py         # Forecast logic
├── utils.py                         # Spatial mean calculation
├── environment.yaml                 # Conda dependencies
└── frontend/
    ├── Dam-Explorer.py              # Main Streamlit dashboard
    ├── generate_analytics.py        # Analytics visualizations
    ├── dams.csv                     # Dam locations metadata
    ├── pages/                       # Multi-page app (River Explorer, About, Team)
    ├── data/                        # Sample datasets
    └── images/                      # UI assets and logos
```

## Running the Dashboard

```bash
conda env create -f environment.yaml
streamlit run frontend/Dam-Explorer.py
```

## Case Study

Demonstrated on the **Blue Nile basin** (Sudan/Ethiopia), using CAMELS-GB hydrometric data. The model is trained on pre-dam construction data (pre-1974) to establish natural flow baselines, then predicts current conditions for comparison.

## Context

Built during the **Cassini Hackathon** by a team of 8. My role was ML engineer — I developed the QI prediction model and data pipeline, and contributed across the full stack. The platform addresses real transboundary water management challenges using Copernicus satellite infrastructure.
