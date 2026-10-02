# GLCD250-MOD

## 🚀 Overview

**GLCD250-MOD** is a global daily chlorophyll-a (Chl-a; mg m⁻³) dataset for
**465,966 inland lakes** at a nominal **250 m** resolution during **2000–2024**.
Estimates are produced from MODIS Terra observations using a transformer-based
hierarchical deep-learning model. Missing satellite observations are not gap-filled.

This repository provides the core download, preprocessing, Terra MODIS L3
recalibration, model-training, evaluation, application, and NetCDF-export code used
in the accompanying study. Exploratory notebooks, cluster scripts, and Zenodo
packaging utilities are not included.

## 📂 Repository structure

```text
01_Data_Download/
  download_mod09gq.py              MOD09GQ download and lake-pixel extraction
  download_mod09ga.py              MOD09GA download and lake-pixel extraction
  download_terra_modis_l3.py        Terra MODIS L3 download and lake-pixel extraction
  Terra_MODIS_L3_URLs/             Terra MODIS L3 source URL lists
  in_situ_dataset_urls.txt         in-situ source URL list
02_Data_Processing/
  create_application_dataset.py    Application Dataset (AD), 250 m
  recalibrate_terra_l3_chla.py      Terra MODIS L3 Chl-a recalibration
  create_l3_training_dataset.py    L3-matched Training Dataset (L3-TD), 4.6 km
03_Model_Development/
  train_model.py                   model training
  evaluate_model.py                model evaluation
  predict_application_dataset.py   Chl-a prediction for the AD
  create_nc_file.py                NetCDF export
  models.py / functions.py         model architecture and supporting functions
  model_config.json               released model configuration
  final_model.pth                 released trained checkpoint
  grid_search.json                configurable hyperparameter candidates
workflow_utils.py       shared feature and I/O functions
requirements.txt        Python dependencies
```

The **Application Dataset (AD)** contains the 250 m model inputs, and the
**L3-matched Training Dataset (L3-TD)** contains spatially aggregated inputs matched
with Terra MODIS L3 targets. The script names follow these manuscript terms;
`predict_application_dataset.py` applies the trained model to the AD.

## 📥 Dataset

The released GLCD250-MOD files are distributed across three Zenodo records:

- [Part 1: Lake IDs 1–50](https://doi.org/10.5281/zenodo.22911946)
- [Part 2: Lake IDs 51–2,000](https://doi.org/10.5281/zenodo.22912781)
- [Part 3: Lake IDs 2,001–1,427,688](https://doi.org/10.5281/zenodo.22913019)

Each NetCDF file contains `chlor_a` in mg m⁻³. The archive descriptions on Zenodo
provide the part-specific file organization and lake information.

## 🛠️ Installation and local inputs

Use Python 3.10+ and install an appropriate PyTorch CPU/CUDA build before running:

```bash
pip install -r requirements.txt
```

Download scripts additionally require `aria2c` and a NASA Earthdata token. Paths
under `data/` can be redirected with `GLCD_DATA_DIR`; HydroLAKES can be supplied
through `GLCD_LAKE_SHP`.

The curated in-situ pixel table, lake split manifest, and recalibration in-situ
table are **not distributed in this repository**. Provide them as local files and
pass their paths through the command-line arguments shown below.
The public source datasets are available from the repositories cited in the paper.
The released checkpoint is included as `03_Model_Development/final_model.pth`;
its SHA-256 is recorded in `model_config.json`. Load only trusted checkpoints.

## 🔄 Core workflow

### 1. Terra MODIS L3 recalibration

The matching step accepts either raw in-situ observations or an already matched
table. Raw observations are matched to their containing Terra L3 grid cell,
prioritizing the same day, then the preceding day, then the following day.
Already matched tables retain their supplied match-ups. Recalibration retains
`0 < in-situ Chl-a < 200 mg m⁻³` for recalibration. A small trial search must be
requested explicitly; the complete search requires `--full-search`.

The complete recalibration search evaluates **960 configurations**: 12 shared MBR
definitions × 4 low-range initial coefficient sets × 4 high-range initial
coefficient sets × 5 initial threshold pairs. Each candidate is fitted to all
match-ups with Nelder–Mead (`maxiter=20000`, `xatol=fatol=1e-8`) under
`0.1 < t1 < t2 <= 50 mg m⁻³`. Its objective is the equal-weight sum of
`abs(RMA slope - 1)`, `abs(RMA intercept)`, `1 - R²`, and quantile RMSE, all in
log₁₀ space. R² is the coefficient of determination, and quantile RMSE compares
the sorted observed and retrieved log concentrations at matching empirical ranks.

Full-data curves are checked for numerical stability on 1,000 MBR values from
0.01 to 30 before LOLOV. This check evaluates unbounded curves; it does not impose
monotonicity or a concentration ceiling. LOLOV targets lakes with at least 20
match-ups (eight in the study); smaller lakes remain in every training fold.
The candidate with the lowest mean validation objective across lakes is selected,
and its **full-data fit** supplies the final coefficients and thresholds.
`--max-jobs` and `--max-lakes` produce trial searches. Search results are written
separately; running a new search does not update the released `model_config.json`.

```bash
python -B 02_Data_Processing/recalibrate_terra_l3_chla.py \
  --insitu /local/path/L3_in-situ_matched.parquet \
  --search --max-jobs 10

python -B 02_Data_Processing/recalibrate_terra_l3_chla.py \
  --insitu /local/path/L3_in-situ_matched.parquet \
  --search --full-search

python -B 02_Data_Processing/recalibrate_terra_l3_chla.py \
  --apply --calibration-config 03_Model_Development/model_config.json \
  --terra-dir /local/path/DB_pixels_TERRA_4km \
  --output /local/path/DB_pixels_TERRA_4km_recalibrated
```

The selected OCx ratio definitions, coefficients, and transition thresholds are
stored in `model_config.json` and are used consistently by preprocessing and the
model's OCx baseline.

### 2. Train and evaluate

```bash
python -B 03_Model_Development/train_model.py \
  --l3 /local/path/L3-TD.parquet \
  --insitu /local/path/GLCD_insitu_modeling_pixels.parquet \
  --manifest /local/path/lake_split.parquet \
  --output /local/path/training --device cuda

python -B 03_Model_Development/evaluate_model.py \
  --checkpoint 03_Model_Development/final_model.pth \
  --l3 /local/path/L3-TD.parquet \
  --insitu /local/path/GLCD_insitu_modeling_pixels.parquet \
  --manifest /local/path/lake_split.parquet \
  --output /local/path/evaluation --device cuda
```

The final configuration uses model dimension 512, 8 attention heads, 4 encoder
layers, feed-forward dimension 2048, learning rate 0.0002, dropout 0.1, L3 batch
size 1024, and in-situ batch size 128. Epoch checkpoints and training history are
saved. Optional candidates can be previewed with `--grid ... --list-only`; this
search file describes a configurable search space and does not assert that every
Cartesian combination was executed for the published model.

### 3. Apply the model and export NetCDF

```bash
python -B 03_Model_Development/predict_application_dataset.py \
  --input /local/path/Application_Dataset \
  --manifest /local/path/shards.parquet --shards 1 --plan

python -B 03_Model_Development/predict_application_dataset.py \
  --input /local/path/Application_Dataset \
  --output /local/path/Prediction \
  --manifest /local/path/shards.parquet --shards 1 --shard-index 0 \
  --checkpoint 03_Model_Development/final_model.pth --device cuda

python -B 03_Model_Development/create_nc_file.py \
  --input /local/path/Prediction \
  --mask /local/path/df_masking_250m.parquet \
  --output /local/path/GLCD250-MOD \
  --model-config 03_Model_Development/model_config.json
```

Source Parquet files are preserved. Application outputs contain `pred_chla` and
`pred_log10_chla`; NetCDF files use `chlor_a` and fill value −999.
Use the full 250 m lake mask, not the L3-matched mask (`df_masking_total.parquet`).
If `--model-config` is supplied, its checkpoint SHA-256 must match the prediction
Parquet metadata; use the released configuration only with the released checkpoint.

## 📚 Reference

Yang S., Lee H., Lee G., Gu T., Park T., Park J., Shin J., Kim T., and Cha Y.
(2026). *GLCD250-MOD: Global Daily 250 m Chlorophyll-a Dataset for Inland Lakes
based on MODIS*. Earth System Science Data [in review].

## 📬 Contact

Water Environmental Management Laboratory, University of Seoul  
YoonKyung Cha: ykcha@uos.ac.kr
