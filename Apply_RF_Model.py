! pip install rasterio joblib tifffile
! apt-get install gdal-bin python3-gdal -qq

import numpy as np
import pandas as pd
import joblib
import rasterio
from rasterio.windows import Window
from google.colab import drive
import os
from tqdm import tqdm
from rasterio.warp import reproject, Resampling

drive.mount('/content/drive')

model_path = #path location storing the pre-trained model

rf_model = joblib.load(model_path)

input_dir = #folder location storing the input data
clear_dir = os.path.join(input_dir, 'TIFF_Daily_Clear_Sky_GSR')
cot_dir = os.path.join(input_dir, 'TIFF_Mean_Daily_COT')
output_dir = #path location storing the outputted GSR_Cloud

os.makedirs(output_dir, exist_ok=True)

def get_tile_windows(width, height, tile_size=2000):
    tiles = []
    for i in range(0, width, tile_size):
        for j in range(0, height, tile_size):
            w = min(tile_size, width - i)
            h = min(tile_size, height - j)
            tiles.append(Window(i, j, w, h))
    return tiles

def process_tile(day_num, tile_idx, window, clear_src, cot_src, profile):
    try:
        clear_data = clear_src.read(1, window=window)
        clear_nodata = clear_src.nodata or np.nan
        gsr_cloud = np.full((window.height, window.width),clear_nodata, dtype=np.float32)

        cot_reprojected = np.empty((window.height, window.width), dtype=np.float32)
        reproject(
            source=rasterio.band(cot_src, 1),
            destination=cot_reprojected,
            src_transform=cot_src.transform,
            src_crs=cot_src.crs,
            dst_transform=rasterio.windows.transform(window, clear_src.transform),
            dst_crs=clear_src.crs,
            resampling=Resampling.bilinear
        )

        valid_mask = (clear_data != clear_nodata) & (~np.isnan(cot_reprojected))

        if np.any(valid_mask):
            X = pd.DataFrame({
                'Daily_Clear': clear_data[valid_mask],
                'Daily_COT': cot_reprojected[valid_mask]
            })
           
            gsr_cloud[valid_mask] = rf_model.predict(X)

        tile_output_dir = os.path.join(output_dir, f'Day_{day_num}')
        os.makedirs(tile_output_dir, exist_ok=True)
        tile_path = os.path.join(tile_output_dir, f'Daily_Cloud_{day_num}_tile_{tile_idx}.tif')

        tile_profile = profile.copy()
        tile_profile.update({
            'width': window.width,
            'height': window.height,
            'transform': rasterio.windows.transform(window, clear_src.transform),
            'dtype': 'float32',
            'nodata': clear_nodata
        })

        with rasterio.open(tile_path, 'w', **tile_profile) as dst:
            dst.write(gsr_cloud, 1)

    except Exception as e:
        print(f"Error processing tile {tile_idx} for day {day_num}: {e}")

def process_day(day_num, tile_size=2000):
    try:
        clear_path = os.path.join(clear_dir, f'Merged_GSR_{day_num}.tif')
        cot_path = os.path.join(cot_dir, f'Mean_Daily_COT_{day_num}.tif')

        if not all(os.path.exists(p) for p in [clear_path, cot_path]):
            raise FileNotFoundError("Input files not found")

        with rasterio.open(clear_path) as clear_src:
            profile = clear_src.profile
            tiles = get_tile_windows(clear_src.width, clear_src.height, tile_size)

            with rasterio.open(cot_path) as cot_src:
                for tile_idx, window in enumerate(tiles):
                    process_tile(day_num, tile_idx, window, clear_src, cot_src, profile)

    except Exception as e:
        print(f"Error processing day {day_num}: {e}")

for day in range(1,2): 
    process_day(day)
