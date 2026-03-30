import os
import gc
import pickle
import datetime

import numpy as np
import pandas as pd
import xarray as xr
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from functions import set_seed, set_device
from models import Transformer_OC_MTAN

set_seed()
device = set_device()

# Load masking file
masking_file = pd.read_parquet("/PATH/TO/YOUR/PARQUET/FILE/df_masking_total.parquet")
Hylak_ids = sorted(list(masking_file['Hylak_id'].unique()))

# Load scalers and columns_dict
with open("/PATH/TO/YOUR/PICKLE/FILE/scalers.pkl", "rb") as f:
    _meta = pickle.load(f)
scalers = _meta["scalers"]
columns_dict = _meta.get("columns_dict")

# Load trained model
best_model_path = "/PATH/TO/YOUR/PICKLE/FILE/best_model.pth"
loaded = torch.load(best_model_path, map_location=device, weights_only=False)
hp = loaded['hyperparams']
model = Transformer_OC_MTAN(
    columns_dict=columns_dict,
    d_model=hp['md'],
    nhead=hp['nh'],
    dim_feedforward=hp['md']*4,
    dropout=hp['dr'],
    num_encoder_layers=hp['nl'],
    scalers=scalers
    )
model.load_state_dict(loaded['model_state_dict'])
model.to(device)
model.eval()

application_dir = "/PATH/TO/YOUR/PARQUET/250 m resolution dataset"
output_dir = "/PATH/TO/YOUR/NETCDF/FILE"
os.makedirs(output_dir, exist_ok=True)

for i, Lake_id in enumerate(Hylak_ids):
    
    hylak_id_str = f'{Lake_id:07}'
    
    # Load application dataset
    file_path = os.path.join(application_dir, f"Hylak_id_{hylak_id_str}.parquet")
    df_id = pd.read_parquet(file_path)
    df_id['date'] = pd.to_datetime(df_id['date'])

    set_seed()

    # Build input tensors
    tensor_dict = {}

    SR_arr  = df_id[columns_dict['SR_vars']].values.astype('float32')
    aux_arr = df_id[columns_dict['aux_input']].values

    aux_arr = scalers['aux_input'].transform(aux_arr)

    tensor_dict['SR_vars']   = torch.tensor(SR_arr).unsqueeze(-1)
    tensor_dict['aux_input'] = torch.tensor(aux_arr, dtype=torch.float32)

    tensor_dataset = TensorDataset(tensor_dict['SR_vars'], tensor_dict['aux_input'])
    data_loader    = DataLoader(tensor_dataset, batch_size=10_000, shuffle=False)

    # Generate predictions
    all_outputs_Rrs = []
    all_outputs_Chl = []

    with torch.no_grad():
        for sr_seq, aux_vars in tqdm(data_loader):
            sr_seq, aux_vars = sr_seq.to(device), aux_vars.to(device)
        
            outputs_Rrs, outputs_Chl = model(sr_seq, aux_vars)
            
            all_outputs_Rrs.append(outputs_Rrs.cpu())
            all_outputs_Chl.append(outputs_Chl.cpu())
            
    all_outputs_Rrs = torch.cat(all_outputs_Rrs, dim=0).numpy()
    all_outputs_Chl = torch.cat(all_outputs_Chl, dim=0).numpy()

    all_outputs_Rrs = scalers['mid_vars'].inverse_transform(all_outputs_Rrs)
    all_outputs_Chl = scalers['Chl_a'].inverse_transform(all_outputs_Chl.reshape(-1, 1)).flatten()

    all_outputs_Chl = np.maximum(all_outputs_Chl, 0)

    df_id['pred_Chl-a'] = all_outputs_Chl
    
    # Build 3D array
    start_date = '2000-02-24'
    end_date = '2024-12-31'
    time_coord = pd.date_range(start=start_date, end=end_date, freq='D')

    masking_file_id = masking_file[masking_file['Hylak_id'] == Lake_id].copy()

    sinu_x_names = np.sort(masking_file_id['sinu_x_GQ'].unique())
    sinu_y_names = np.sort(masking_file_id['sinu_y_GQ'].unique())

    lat_names = np.sort(masking_file_id['lat'].unique())
    lon_names = np.sort(masking_file_id['lon'].unique())

    pi_array = np.full((len(time_coord), len(sinu_y_names), len(sinu_x_names)), 
                    np.nan, dtype=np.float32)

    time_map = {val: i for i, val in enumerate(time_coord)}
    x_map = {val: i for i, val in enumerate(sinu_x_names)}
    y_map = {val: i for i, val in enumerate(sinu_y_names)}

    if 'sinu_x_GQ' not in df_id.columns:
        df_id = df_id.merge(masking_file_id[['lon', 'lat', 'sinu_x_GQ', 'sinu_y_GQ']], on=['lon', 'lat'], how='left')
        
    df_id['t_idx'] = df_id['date'].map(time_map)
    df_id['x_idx'] = df_id['sinu_x_GQ'].map(x_map)
    df_id['y_idx'] = df_id['sinu_y_GQ'].map(y_map)

    pi_array[df_id['t_idx'].values, 
            df_id['y_idx'].values, 
            df_id['x_idx'].values] = df_id['pred_Chl-a'].values

    # Create NetCDF file
    now = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    y_dim = 'YDim_MODIS_Grid_2D'
    x_dim = 'XDim_MODIS_Grid_2D'

    ds = xr.Dataset(
        {
            'chlor_a': (['time', y_dim, x_dim], pi_array),
        },
        coords={
            'time': time_coord,
            y_dim: np.array(sinu_y_names),
            x_dim: np.array(sinu_x_names), 
        },
    )

    ds['MODIS_Sinusoidal_Tiling_System'] = xr.DataArray(
        0,
        attrs={
            'grid_mapping_name': 'sinusoidal',
            'longitude_of_projection_origin': 0.0,
            'false_easting': 0.0,
            'false_northing': 0.0,
            'earth_radius': 6371007.181,
            'long_name': 'MODIS_Sinusoidal_Tiling_System'
        }
    )

    ds['chlor_a'].attrs = {
        'standard_name': 'mass_concentration_of_chlorophyll_a_in_sea_water', 
        'long_name': 'Chlorophyll-a Concentration', 
        'units': 'mg m-3', 
        'coverage_content_type': 'physicalMeasurement',
        'grid_mapping': 'MODIS_Sinusoidal_Tiling_System'
    }

    ds['time'].attrs  = {'standard_name': 'time', 'axis': 'T'}
    ds['YDim_MODIS_Grid_2D'].attrs = {'standard_name': 'projection_y_coordinate', 'units': 'm', 'axis': 'Y', 'coverage_content_type': 'coordinate'}
    ds['XDim_MODIS_Grid_2D'].attrs = {'standard_name': 'projection_x_coordinate', 'units': 'm', 'axis': 'X', 'coverage_content_type': 'coordinate'}

    ds.attrs['Conventions']                = 'CF-1.8, ACDD-1.3'
    ds.attrs['title']                      = 'GLCD250-MOD: Global Lakes Chlorophyll-a Daily 250 m based on MODIS'
    ds.attrs['id']                         = f'GLCD250-MOD_{hylak_id_str}'
    ds.attrs['product_version']            = 'v1.0' 
    ds.attrs['summary']                    = "GLCD250-MOD provides daily chlorophyll-a (Chl-a) concentrations (mg m⁻³) for 465,966 inland lakes worldwide at a spatial resolution of 250 m, covering a 25-year period from 2000 to 2024. Chl-a concentrations were estimated using a transformer-based hierarchical deep-learning model trained on the relationships between MODIS Terra surface reflectance (MOD09) and remote sensing reflectance and Chl-a products derived from the MODIS Terra Level-3 dataset. For each lake, data are stored in individual NetCDF files ('GLCD250-MOD_[Hylak_id].nc') containing time, chlorophyll-a concentration ('chlor-a'), projection_x_coordinate, and projection_y_coordinate. Lake metadata, including geographic and morphometric information, are provided in an accompanying shapefile ('Lake_info.shp'). Quality control and quality assurance (QC/QA) for GLCD250-MOD were implemented using MOD09 QC, state QA and spectral indices (MNDWI and NDSI), which were applied to retain valid water pixels while excluding non-water areas, ice-covered pixels, and atmospheric noises."
    ds.attrs['keywords']                   = 'Chlorophyll-a, Algal bloom, Global lakes, Satellite, Deep learning'
    ds.attrs['keywords_vocabulary']        = 'none'

    ds.attrs['cdm_data_type']              = 'Grid' 
    ds.attrs['comment']                    = 'See summary attribute'
    ds.attrs['creator_email']              = 'ykcha@uos.ac.kr' 
    ds.attrs['creator_institution']        = 'Water Environmental Management Laboratory, University of Seoul' 
    ds.attrs['creator_name']               = 'SooHyun Yang, HaeDeun Lee, Taeho Kim, YoonKyung Cha'    
    ds.attrs['creator_type']               = 'person' 
    ds.attrs['creator_url']                = 'http://wemlab.uos.ac.kr' 
    ds.attrs['date_created']               = now
    ds.attrs['institution']                = 'Water Environmental Management Laboratory, University of Seoul' 

    ds.attrs['geospatial_lat_max']         = float(lat_names.max())
    ds.attrs['geospatial_lat_min']         = float(lat_names.min())
    ds.attrs['geospatial_lat_resolution']  = '0.0025 degrees'
    ds.attrs['geospatial_lat_units']       = 'degrees_north'
    ds.attrs['geospatial_lon_max']         = float(lon_names.max())
    ds.attrs['geospatial_lon_min']         = float(lon_names.min())
    ds.attrs['geospatial_lon_resolution']  = '0.0025 degrees'
    ds.attrs['geospatial_lon_units']       = 'degrees_east'

    ds.attrs['time_coverage_start']        = '2000-02-24T00:00:00Z'
    ds.attrs['time_coverage_end']          = '2024-12-31T00:00:00Z'
    ds.attrs['time_coverage_duration']     = 'P25Y'
    ds.attrs['time_coverage_resolution']   = 'P1D'

    ds.attrs['platform']                   = 'Terra'
    ds.attrs['instrument']                 = 'MODIS'
    ds.attrs['source']                     = 'surface reflectance from MODIS-Terra L2 R2018.0, remote sensing reflectance from MODIS Terra L3 R2022.0.2, chlorophyll-a concentration from MODIS Terra L3 R2022.0.2' 
    ds.attrs['processing_level']           = 'Level-2 gridded' 
    ds.attrs['standard_name_vocabulary']   = 'CF Standard Name Table v92' 
    ds.attrs['license']                    = 'CC-BY-4.0' 
    ds.attrs['references']                 = 'https://github.com/Yang-SooHyun/GLCD250-MOD' 
    ds.attrs['naming_authority']           = 'kr.ac.uos.wemlab'
    ds.attrs['history']                    = f"{now}: GLCD250-MOD V1.0 created. Derived from MODIS-Terra 250m daily surface reflectance." 

    ds.attrs['publisher_name']             = 'Water Environmental Management Laboratory, University of Seoul'
    ds.attrs['publisher_email']            = 'ykcha@uos.ac.kr'
    ds.attrs['publisher_url']              = 'http://wemlab.uos.ac.kr'

    ds['time'].encoding['units']    = 'days since 2000-02-24 00:00:00'
    ds['time'].encoding['calendar'] = 'gregorian'

    encoding = {
        'chlor_a': {
            'zlib': True,
            'complevel': 2,
            'dtype': 'float32', 
            '_FillValue': -999.0
        }
    }

    out_nc = os.path.join(output_dir, f"GLCD250-MOD_{hylak_id_str}.nc")
    ds.to_netcdf(out_nc, encoding=encoding)
    ds.close()
    
    del ds, pi_array, df_id, masking_file_id
    gc.collect()