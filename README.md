# GLCD250-MOD: Global Daily 250 m Chlorophyll-a Dataset for Inland Lakes based on MODIS

This repository provides the source code used to develop **GLCD250-MOD**, a global daily chlorophyll-a (Chl-a) dataset for **465,966 inland lakes** at **250 m spatial resolution** during **2000–2024**.

---

## 🚀 Overview

GLCD250-MOD estimates lake Chl-a concentrations (mg m⁻³) from Terra MODIS observations using a transformer-based hierarchical deep learning model. The dataset supports the analysis of algal bloom patterns and long-term changes in inland lakes. Missing satellite observations are not gap-filled.

The dataset is available on Zenodo:

- [Part 1: Lake IDs 1–50](https://doi.org/10.5281/zenodo.22911946)
- [Part 2: Lake IDs 51–2,000](https://doi.org/10.5281/zenodo.22912781)
- [Part 3: Lake IDs 2,001–1,427,688](https://doi.org/10.5281/zenodo.22913019)

Each NetCDF file contains the Chl-a variable `chlor_a`. File organization and lake information are described in the corresponding Zenodo records.

---

## 📂 Repository Structure

```text
GLCD250-MOD/
├── 01_Data_Download/        # MODIS downloads and lake-pixel extraction
├── 02_Data_Processing/      # Terra L3 recalibration and dataset preparation
├── 03_Model_Development/    # Model training, evaluation, prediction, and NetCDF export
├── workflow_utils.py       # Shared feature and I/O functions
├── requirements.txt        # Python dependencies
└── README.md
```

The trained model (`final_model.pth`) and its configuration (`model_config.json`) are provided in `03_Model_Development/`.

---

## 📥 Data Sources

### MODIS surface reflectance

- [**MOD09GQ**](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD09GQ): Terra daily surface reflectance at 250 m.
- [**MOD09GA**](https://ladsweb.modaps.eosdis.nasa.gov/missions-and-measurements/products/MOD09GA): Terra daily surface reflectance at 500 m and 1 km.

### Terra MODIS Level-3 products

[**Terra MODIS Level-3**](https://oceandata.sci.gsfc.nasa.gov/l3/) remote-sensing reflectance (Rrs) and Chl-a products provide reference data for model development.

### In-situ observations

Public in-situ data sources are listed in `01_Data_Download/in_situ_dataset_urls.txt`. The curated in-situ tables and lake split manifest must be provided separately to run training and evaluation.

---

## 🛠️ Method Summary

### Terra MODIS Level-3 Chl-a recalibration

Terra MODIS Level-3 Chl-a is recalibrated using in-situ observations by combining OCx relationships for low and high Chl-a concentrations. The recalibrated estimates are used as reference targets for model development.

### Chl-a model development

MODIS surface reflectance inputs are prepared as a **250 m Application Dataset (AD)** and an aggregated **4.6 km L3-matched Training Dataset (L3-TD)**. A transformer-based hierarchical deep learning model is trained and evaluated, then applied to the AD to generate the final Chl-a estimates.

---

## ▶️ Workflow

1. Download MODIS products and extract lake pixels.
2. Recalibrate Terra MODIS Level-3 Chl-a using in-situ observations.
3. Prepare the L3-TD and AD.
4. Train and evaluate the model.
5. Apply the trained model to the AD.
6. Export Chl-a estimates as NetCDF files.

Use Python 3.10+ with an appropriate PyTorch CPU/CUDA build, then install the dependencies:

```bash
pip install -r requirements.txt
```

Download scripts also require `aria2c` and a NASA Earthdata token. Local data paths can be configured using `GLCD_DATA_DIR` and `GLCD_LAKE_SHP`.

---

## 📚 Reference

Yang, S., Lee, H., Lee, G., Gu, T., Park, T., Park, J., Shin, J., Kim, T., and Cha, Y. (2026). GLCD250-MOD: Global Daily 250 m Chlorophyll-a Dataset for Inland Lakes based on MODIS. *Earth System Science Data*. [In review].

---

## 📬 Contact

* SooHyun Yang — University of Seoul, ghdns95@uos.ac.kr
* HaeDeun Lee — University of Seoul, leehaed@uos.ac.kr
* Taeho Kim — University of Michigan - Ann Arbor, theokim@umich.edu
* YoonKyung Cha — University of Seoul, ykcha@uos.ac.kr
