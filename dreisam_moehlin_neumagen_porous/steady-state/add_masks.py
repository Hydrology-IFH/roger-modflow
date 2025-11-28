from pathlib import Path
import h5netcdf
import xarray as xr
import shutil
import rasterio
from pathlib import Path
import numpy as np


base_path = Path(__file__).parent

# copy file to new file to keep original file unchanged
path1 = str(base_path / "input" / "parameters_modflow_.nc")
path2 = str(base_path / "input" / "parameters_modflow.nc")
shutil.copy(path1, path2)

# add mask of Schoenberg to parameters_modflow.nc
src = rasterio.open(str(base_path / "input" / "schoenberg.tif"))
mask_schoenberg = src.read(1)
# add mask of subatchments to parameters_modflow.nc
src = rasterio.open(str(base_path / "input" / "mask_upper_dreisam.tif"))
mask_upper_dreisam = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_lower_dreisam.tif"))
mask_lower_dreisam = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_upper_moehlin.tif"))
mask_upper_moehlin = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_lower_moehlin.tif"))
mask_lower_moehlin = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_neumagen.tif"))
mask_neumagen = src.read(1)
# add masks of gravel areas to parameters_modflow.nc
src = rasterio.open(str(base_path / "input" / "mask_zarten_brugga.tif"))
mask_zarten_brugga = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_zarten_gravel_north.tif"))
mask_zarten_gravel_north = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_staufen_gravel.tif"))
mask_staufen_gravel = src.read(1)
# add custom masks
src = rasterio.open(str(base_path / "input" / "mask_kf_18e-3_lower_moehlin.tif"))
mask_kf_18e_3_lower_moehlin = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_kf_2e-7_lower_moehlin_and_dreisam.tif"))
mask_kf_2e_7_lower_moehlin_and_dreisam = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_black_forest.tif"))
mask_black_forest = src.read(1)

with xr.open_dataset(base_path / "input" / "parameters_modflow.nc") as ds:
    topography = ds['elevations'].isel(z=0).values
    spatial_ref = ds.spatial_ref
    xcoords = ds.x.values
    ycoords = ds.y.values[::-1]

mask_catchment = np.where(np.isfinite(topography) & (mask_schoenberg == 0) & (mask_black_forest == 0), 1, 0)

path = str(base_path / "input" / "parameters_modflow.nc")
with h5netcdf.File(path, "a", decode_vlen_strings=False) as f:
    var_obj = f.variables.get("y")
    var_obj[:] = ycoords
    kf_layer2 = f.variables.get("kf")[1, :, :]
    mask_zarten_brugga = np.where((mask_zarten_brugga == 1) & (kf_layer2 > 8.64), 1, 0)
    mask_zarten_gravel_north = np.where((mask_zarten_gravel_north == 1) & (kf_layer2 > 8.64), 1, 0)
    mask_staufen_gravel = np.where((mask_staufen_gravel == 1) & (kf_layer2 > 8.64), 1, 0)
    try:
        v = f.create_variable("mask_schoenberg", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_schoenberg
        v.attrs.update(long_name="Mask of Schoenberg", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_schoenberg")
        var_obj[:, :] = mask_schoenberg
    try:
        v = f.create_variable("mask_upper_dreisam", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_upper_dreisam
        v.attrs.update(long_name="Mask of upper Dreisam catchment", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_upper_dreisam")
        var_obj[:, :] = mask_upper_dreisam
    try:
        v = f.create_variable("mask_lower_dreisam", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_lower_dreisam
        v.attrs.update(long_name="Mask of lower Dreisam catchment", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_lower_dreisam")
        var_obj[:, :] = mask_lower_dreisam
    try:
        v = f.create_variable("mask_upper_moehlin", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_upper_moehlin
        v.attrs.update(long_name="Mask of upper Moehlin catchment", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_upper_moehlin")
        var_obj[:, :] = mask_upper_moehlin
    try:
        v = f.create_variable("mask_lower_moehlin", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_lower_moehlin
        v.attrs.update(long_name="Mask of lower Moehlin catchment", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_lower_moehlin")
        var_obj[:, :] = mask_lower_moehlin
    try:
        v = f.create_variable("mask_neumagen", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_neumagen
        v.attrs.update(long_name="Mask of Neumagen catchment", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_neumagen")
        var_obj[:, :] = mask_neumagen
    try:
        v = f.create_variable("mask_zarten_brugga", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_zarten_brugga
        v.attrs.update(long_name="Mask of gravel areas (Brugga)", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_zarten_brugga")
        var_obj[:, :] = mask_zarten_brugga
    try:
        v = f.create_variable("mask_zarten_gravel_north", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_zarten_gravel_north
        v.attrs.update(long_name="Mask of gravel areas (Zarten basin)", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_zarten_gravel_north")
        var_obj[:, :] = mask_zarten_gravel_north
    try:
        v = f.create_variable("mask_staufen_gravel", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_staufen_gravel
        v.attrs.update(long_name="Mask of gravel areas (Staufen)", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_staufen_gravel")
        var_obj[:, :] = mask_staufen_gravel
    try:
        v = f.create_variable("mask_kf_18e_3_lower_moehlin", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_kf_18e_3_lower_moehlin
        v.attrs.update(long_name="Custom mask (Hausen)", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_kf_18e_3_lower_moehlin")
        var_obj[:, :] = mask_kf_18e_3_lower_moehlin
    try:
        v = f.create_variable("mask_kf_2e_7_lower_moehlin_and_dreisam", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_kf_2e_7_lower_moehlin_and_dreisam
        v.attrs.update(long_name="Custom mask (Hausen)", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_kf_2e_7_lower_moehlin_and_dreisam")
        var_obj[:, :] = mask_kf_2e_7_lower_moehlin_and_dreisam
    try:
        v = f.create_variable("mask_black_forest", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_black_forest
        v.attrs.update(long_name="Mask of Black Forest", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_black_forest")
        var_obj[:, :] = mask_black_forest
    try:
        v = f.create_variable("mask_catchment", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_catchment
        v.attrs.update(long_name="Mask of catchment area", units="", grid_mapping="spatial_ref", coordinates="spatial_ref")
    except ValueError:
        var_obj = f.variables.get("mask_catchment")
        var_obj[:, :] = mask_catchment