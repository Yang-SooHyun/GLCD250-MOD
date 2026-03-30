# GLCD250-MOD: Global Daily 250 m Chlorophyll-a Dataset for Inland Lakes based on MODIS

This repository provides the source code used to generate **GLCD250-MOD**, a **global daily 250 m chlorophyll-a (Chl-a) dataset for inland lakes based on MODIS**.

The archived dataset is released on Zenodo as: **GLCD250-MOD: Global Lakes Chlorophyll-a Daily 250 m based on MODIS**

## Overview

GLCD250-MOD is a **global daily 250 m chlorophyll-a dataset for inland lakes** derived from **Terra MODIS observations** for the period **2000–2024**. The dataset is produced using **MOD09 surface reflectance (SR)** as model input and **MODIS Terra Level-3 remote-sensing reflectance (Rrs) and chlorophyll-a (Chl-a) products** for model development.

The model is a **transformer-based hierarchical deep learning framework** that predicts Chl-a through an intermediate estimation of Rrs. After quality control and valid-pixel screening, the dataset provides Chl-a estimates for **465,966 inland lakes worldwide**.

## Key Features

- **Spatial resolution**: 250 m
- **Temporal resolution**: Daily
- **Coverage**: Global inland lakes
- **Modeling approach**: Transformer-based hierarchical deep learning

## Data Sources

The GLCD250-MOD framework uses the following MODIS products for model development and dataset generation:

- **MOD09GQ / MOD09GA**: Surface reflectance products (250 m / 500 m)  
  - MOD09GQ: https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD09GQ  
  - MOD09GA: https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD09GA

- **MODIS-Terra Level-3 mapped global products**: Remote sensing reflectance (Rrs) and chlorophyll-a products (4.6 km)  
  - https://oceandata.sci.gsfc.nasa.gov/l3/

## Workflow

1. Download MODIS input products
2. Build application and development datasets
3. Train and evaluate the model
4. Apply the trained model to generate GLCD250-MOD NetCDF outputs

## Authors

- SooHyun Yang, HaeDeun Lee, Taeho Kim, YoonKyung Cha

## Project Structure

```bash
GLCD250-MOD/
├── 01_Data_Download/        # Download, mask, and organize MODIS input products
├── 02_Data_Processing/      # Construct application and development datasets through spatial matching, filtering, and feature engineering
├── 03_Model_Development/    # Define, train, and evaluate models, apply the trained model to application datasets, and generate NetCDF outputs
├── requirements.txt         # Python package requirements
└── README.md
