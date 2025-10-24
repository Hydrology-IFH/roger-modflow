from pathlib import Path
import xarray as xr
import geoxarray
import numpy as onp
import pandas as pd
import datetime


base_path = Path(__file__).parent

params_file = base_path / "input" / "parameters_roger_25m.nc"
ds_params = xr.open_dataset(params_file)
xcoords = ds_params.x.values
ycoords = ds_params.y.values

# load the netcdf file
file = base_path / "input" / "crop_rotations_2018-2022.nc"
ds_cr_2018_2022 = xr.open_dataset(file)
spatial_ref = ds_cr_2018_2022.spatial_ref
lu_ids_2018_2022 = ds_cr_2018_2022['Nutzcode'].values
cond = onp.isnan(lu_ids_2018_2022)
lu_ids_2018_2022[cond] = -9999  # set nan to
lu_ids_2018_2022 = lu_ids_2018_2022.astype(onp.int16)

time_2013_2023 = onp.arange(2013, 2023)
time_2000_2023 = onp.arange(2000, 2023)

lu_ids_2013_2023 = onp.zeros((len(time_2013_2023), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
lu_ids_2013_2023[0:5, :, :] = lu_ids_2018_2022
lu_ids_2013_2023[5:10, :, :] = lu_ids_2018_2022
lu_ids_2013_2023[-1, :, :] = lu_ids_2018_2022[0, :, :]

lu_ids_2000_2023 = onp.zeros((len(time_2000_2023), lu_ids_2018_2022.shape[1], lu_ids_2018_2022.shape[2]), dtype=onp.int16)
lu_ids_2000_2023[0:5, :, :] = lu_ids_2018_2022
lu_ids_2000_2023[5:10, :, :] = lu_ids_2018_2022
lu_ids_2000_2023[10:15, :, :] = lu_ids_2018_2022
lu_ids_2000_2023[15:20, :, :] = lu_ids_2018_2022
lu_ids_2000_2023[-1, :, :] = lu_ids_2018_2022[0, :, :]

# create xarray dataset
attrs = dict(
        date_created=datetime.datetime.today().isoformat(),
        title="lu_id of RoGeR in the Dreisam-Moehlin-Neumagen catchment for the years 2013-2023",
        institution="University of Freiburg, Chair of Hydrology",
    )
coords = {
        "lon": ("lon", xcoords),  # x
        "lat": ("lat", ycoords),  # y
        "Time": ("Time", time_2013_2023),
    }
data_vars=dict(
        crop_type=(["Time", "lat", "lon"], lu_ids_2013_2023),
    )

ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds["crop_type"].attrs["units"] = ""
ds["crop_type"].attrs["long_name"] = "Crop type encoded as lu_id from RoGeR"
# create spatial reference
ds = ds.geo.write_crs("EPSG:25832")
ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
file = base_path / "input" / "crop_rotations_2013-2023.nc"
comp = dict(zlib=True, complevel=1)  # compress data to save storage
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)


# create xarray dataset
attrs = dict(
        date_created=datetime.datetime.today().isoformat(),
        title="lu_id of RoGeR in the Dreisam-Moehlin-Neumagen catchment for the years 2000-2023",
        institution="University of Freiburg, Chair of Hydrology",
    )
coords = {
        "lon": ("lon", xcoords),  # x
        "lat": ("lat", ycoords),  # y
        "Time": ("Time", time_2000_2023),
    }
data_vars=dict(
        crop_type=(["Time", "lat", "lon"], lu_ids_2000_2023),
    )

ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
ds["crop_type"].attrs["units"] = ""
ds["crop_type"].attrs["long_name"] = "Crop type encoded as lu_id from RoGeR"
# create spatial reference
ds = ds.geo.write_crs("EPSG:25832")
ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
file = base_path / "input" / "crop_rotations_2000-2023.nc"
comp = dict(zlib=True, complevel=1)  # compress data to save storage
encoding = {var: comp for var in ds.data_vars}
ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)

