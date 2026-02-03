# GLCD250-MOD: Global Lakes Chlorophyll-a Daily 250 m using MODIS

## 🌟 Project Overview
This project establishes a robust relationship between **MOD09 Surface Reflectance** and **MODIS Terra L3 Chl-a** products to generate a **daily Chlorophyll-a (Chl-a)** dataset at a **250 m resolution** for **global inland lakes**. By leveraging machine learning, it transforms high-frequency MODIS observations into high-resolution water quality data.

## 🚀 Key Features
* **Resolution**: Provides **250 m** spatial resolution and **daily** temporal resolution.
* **Global Coverage**: Specifically designed and validated for **global lakes**.
* **Data Integration**: Utilizes the synergy between **MOD09 (Reflectance)** and **MODIS Terra L3 (Standard Chl-a product)**.
* **ML-based Estimation**: Employs an optimized **XGBoost (XGB)** model to capture complex relationships between reflectance and water quality parameters.

## 🛠 Tech Stack
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-black?style=for-the-badge&logo=xgboost)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📊 Data Sources
The model integrates the following MODIS data products:
* **MOD09GQ / MOD09GA**: Surface Reflectance (250m / 500m)
* **MODIS Terra L3**: Chl-a (4.6km)

## 📂 Project Structure
* **`01_Data_Download`**: Scripts for automated downloading of MODIS products using NASA Earthdata API.
* **`02_Data_Processing`**: Data cleaning, cloud masking, and spatial resampling (downscaling) to 250m.
* **`03_Model_Development`**: Training and hyperparameter optimization for the **XGBoost (XGB)** model.
* **`04_Figures`**: Code for generating spatial distribution maps, scatter plots, and time-series analysis.
