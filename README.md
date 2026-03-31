# GLCD250-MOD: Global Daily 250 m Chlorophyll-a Dataset for Inland Lakes based on MODIS

This repository provides the source code used to develop **GLCD250-MOD**, a global **daily 250 m chlorophyll-a (Chl-a)** dataset for inland lakes based on **MODIS**.

GLCD250-MOD provides daily Chl-a concentration (mg/m³) for **465,966 lakes** distributed across six continents during **2000–2024**.

The archived dataset is available on Zenodo under the title:  
**[GLCD250-MOD: Global Lakes Chlorophyll-a Daily 250 m based on MODIS](https://doi.org/10.5281/zenodo.18950636)**

---

## 🚀 Overview

Algal blooms affect water quality, lake ecosystems, and human health, highlighting the need for high-resolution monitoring of inland lakes across both space and time. Because **chlorophyll-a (Chl-a)** is widely used as a surrogate indicator of phytoplankton biomass, it provides a useful basis for quantifying algal bloom dynamics.

To support large-scale and long-term monitoring of inland lakes, we developed **GLCD250-MOD**, a new global inland lake Chl-a dataset based on **Terra MODIS** observations. GLCD250-MOD provides **daily Chl-a concentration at 250 m spatial resolution** for inland lakes worldwide from **2000 to 2024**.

By providing Chl-a at finer spatial resolution than existing global products, GLCD250-MOD substantially improves data availability, particularly for **small and medium-sized lakes** that are often underrepresented in global Chl-a datasets. The dataset is intended to support investigations of historical algal bloom patterns and assessments of climate-driven changes in lake systems at regional to global scales.

---

## 📂 Repository Structure

```bash
GLCD250-MOD/
├── 01_Data_Download/        # Download and organize MODIS products
├── 02_Data_Processing/      # Quality control, masking, spatial matching, and feature generation
├── 03_Model_Development/    # Model definition, training, evaluation, and product generation
├── requirements.txt         # Python package requirements
└── README.md
```

---

## 📥 Data sources

GLCD250-MOD was developed using the following MODIS products:

### 🛰️ MODIS surface reflectance products

* [**MOD09GQ**](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD09GQ): Terra Surface Reflectance Daily L2G Global 250 m
* [**MOD09GA**](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD09GA): Terra Surface Reflectance Daily L2G Global 500 m and 1 km

These products were used to construct the input variables for global 250 m Chl-a estimation.

### 🛰️ [Terra MODIS Level-3 global mapped products](https://oceandata.sci.gsfc.nasa.gov/l3/)

* **Remote-sensing reflectance (Rrs)**
* **Chlorophyll-a (Chl-a)**

These products were used as reference data for model training and evaluation.

---

## 🛠️ Method Summary

GLCD250-MOD was generated using a **transformer-based hierarchical deep learning model** developed from MODIS surface reflectance and MODIS Terra Level-3 products. The model was designed to estimate **Rrs as an intermediate variable** and then predict **Chl-a** as the final target variable. This framework was used to produce spatially detailed daily Chl-a estimates for inland lakes worldwide.

---

## ▶️ Workflow

The overall workflow is summarized as follows:

1. Download and organize MODIS products
2. Apply quality control, quality assurance, masking, and feature engineering
3. Construct training and application datasets
4. Train and evaluate the Chl-a estimation model
5. Apply the trained model to generate global 250 m Chl-a outputs
6. Export the final GLCD250-MOD products

---

## 📚 References

Yang, S., Lee, H., Lee, G., Gu, T., Shin, J., Kim, T., & Cha, Y. (2026). GLCD250-MOD: Global Daily 250 m Chlorophyll-a Dataset for Inland Lakes based on MODIS. _Scientific Data_. [In review].

---

## 📬 Contact

* SooHyun Yang — University of Seoul, ghdns95@uos.ac.kr
* HaeDeun Lee — University of Seoul, leehaed@uos.ac.kr
* Taeho Kim — University of Michigan - Ann Arbor, theokim@umich.edu
* YoonKyung Cha — University of Seoul. ykcha@uos.ac.kr
