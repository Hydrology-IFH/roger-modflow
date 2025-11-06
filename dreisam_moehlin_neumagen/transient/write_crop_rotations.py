from pathlib import Path
import xarray as xr
import geoxarray
import numpy as onp
import pandas as pd
import datetime
import roger.lookuptables as lut

summer_crops = lut.SUMMER_CROPS.tolist()
winter_crops = lut.WINTER_CROPS.tolist()

base_path = Path(__file__).parent

params_file = base_path / "input" / "parameters_roger_25m.nc"
ds_params = xr.open_dataset(params_file)
xcoords = ds_params.x.values
ycoords = ds_params.y.values
mask = ds_params['maskCatch'].values

# load the netcdf file
file = base_path / "input" / "crop_rotations_2018-2022.nc"
ds_cr_2018_2022 = xr.open_dataset(file)
spatial_ref = ds_cr_2018_2022.spatial_ref
lu_ids_2018_2022 = ds_cr_2018_2022['Nutzcode'].values
cond = onp.isnan(lu_ids_2018_2022)
lu_ids_2018_2022[cond] = -9999  # set nan to
lu_ids_2018_2022 = lu_ids_2018_2022.astype(onp.int16)

# calculate area fractions of each crop type
years = [2018, 2019, 2020, 2021, 2022]
for t, year in enumerate(years):
    lu_ids_year = lu_ids_2018_2022[t, :, :]
    cond = mask == 0
    lu_ids_year[cond] = -9999
    unique, counts = onp.unique(lu_ids_year, return_counts=True)
    total_counts = onp.sum(mask)
    area_fractions = counts / total_counts
    print(f"Year: {year}")
    for i in range(len(unique)):
        print(f"  lu_id: {unique[i]}, area fraction: {area_fractions[i]:.4f}")

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

# create csv file for crop rotations
unit_header = [""]
time_header = ["No"]
for year in time_2013_2023:
    unit_header.append("[year_season]")
    unit_header.append("[year_season]")
    time_header.append(f"{year}_summer")
    time_header.append(f"{year}_winter")
df_crop_rotations = pd.DataFrame(columns=time_header)
df_crop_rotations.loc[:, "No"] = onp.arange(1, lu_ids_2013_2023.shape[1] * lu_ids_2013_2023.shape[2] + 1)
df_crop_rotations.loc[:, time_header[1]:] = -9999  # initialize with -9999
idx_xy = onp.arange(1, lu_ids_2013_2023.shape[1] * lu_ids_2013_2023.shape[2] + 1).reshape((lu_ids_2013_2023.shape[1], lu_ids_2013_2023.shape[2]))
# for t, year in enumerate(time_2013_2023[:-1]):
#     lu_ids_year = lu_ids_2013_2023[t, :, :].flatten()
#     lu_ids_summer = onp.zeros(lu_ids_year.shape, dtype=onp.int16)
#     lu_ids_winter = onp.zeros(lu_ids_year.shape, dtype=onp.int16)
#     lu_ids_year1 = lu_ids_2013_2023[t+1, :, :].flatten()
#     lu_ids_summer1 = onp.zeros(lu_ids_year.shape, dtype=onp.int16)
#     lu_ids_winter1 = onp.zeros(lu_ids_year.shape, dtype=onp.int16)
#     lu_ids_summer[:] = -9999
#     lu_ids_winter[:] = -9999
#     lu_ids_summer1[:] = -9999
#     lu_ids_winter1[:] = -9999
#     cond_summer_crops = onp.isin(lu_ids_year, summer_crops)
#     cond_winter_crops = onp.isin(lu_ids_year, winter_crops)
#     cond_summer1_crops = onp.isin(lu_ids_year1, summer_crops)
#     cond_winter1_crops = onp.isin(lu_ids_year1, winter_crops)
#     lu_ids_summer[cond_summer_crops] = lu_ids_year[cond_summer_crops]
#     lu_ids_winter[cond_summer_crops] = 599
#     lu_ids_summer[cond_winter_crops] = 599
#     lu_ids_winter[cond_winter_crops] = lu_ids_year[cond_winter_crops]
#     lu_ids_summer1[cond_summer1_crops] = lu_ids_year1[cond_summer1_crops]
#     lu_ids_winter1[cond_summer1_crops] = 599
#     lu_ids_summer1[cond_winter1_crops] = 599
#     lu_ids_winter1[cond_winter1_crops] = lu_ids_year1[cond_winter1_crops]
#     # lu_ids_summer[cond_winter_crops & cond_summer1_crops] = lu_ids_year1[cond_winter_crops & cond_summer1_crops]
#     # lu_ids_summer1[cond_winter_crops & cond_summer1_crops] = 599
#     # cond_both_crops = onp.isin(lu_ids_year, [8, 81, 82])
#     # cond_both_crops1 = onp.isin(lu_ids_year1, [8, 81, 82])
#     # lu_ids_summer[cond_both_crops] = lu_ids_year[cond_both_crops]
#     # lu_ids_winter[cond_both_crops] = lu_ids_year[cond_both_crops]
#     # lu_ids_summer1[cond_both_crops1] = lu_ids_year1[cond_both_crops1]
#     # lu_ids_winter1[cond_both_crops1] = lu_ids_year1[cond_both_crops1]
#     # cond_fallow = lu_ids_year == 599
#     # lu_ids_summer[cond_fallow] = 599
#     # lu_ids_winter[cond_fallow] = 599
#     df_crop_rotations.loc[:, f"{year}_summer"] = lu_ids_summer
#     df_crop_rotations.loc[:, f"{year}_winter"] = lu_ids_winter
#     df_crop_rotations.loc[:, f"{year+1}_summer"] = lu_ids_summer1
#     df_crop_rotations.loc[:, f"{year+1}_winter"] = lu_ids_winter1

for t, year in enumerate(time_2013_2023):
    lu_ids_year = lu_ids_2013_2023[t, :, :].flatten()
    lu_ids_summer = onp.zeros(lu_ids_year.shape, dtype=onp.int16)
    lu_ids_winter = onp.zeros(lu_ids_year.shape, dtype=onp.int16)
    lu_ids_summer[:] = -9999
    lu_ids_winter[:] = -9999
    cond_summer_crops = onp.isin(lu_ids_year, summer_crops)
    cond_winter_crops = onp.isin(lu_ids_year, winter_crops)
    lu_ids_summer[cond_summer_crops] = lu_ids_year[cond_summer_crops]
    lu_ids_winter[cond_summer_crops] = 599
    lu_ids_summer[cond_winter_crops] = 599
    lu_ids_winter[cond_winter_crops] = lu_ids_year[cond_winter_crops]
    cond_both_crops = onp.isin(lu_ids_year, [8, 81, 82])
    lu_ids_summer[cond_both_crops] = lu_ids_year[cond_both_crops]
    lu_ids_winter[cond_both_crops] = lu_ids_year[cond_both_crops]
    cond_fallow = lu_ids_year == 599
    lu_ids_summer[cond_fallow] = 599
    lu_ids_winter[cond_fallow] = 599
    df_crop_rotations.loc[:, f"{year}_summer"] = lu_ids_summer
    df_crop_rotations.loc[:, f"{year}_winter"] = lu_ids_winter



file = base_path / "input" / "crop_rotations_2013-2023.csv"
header = [unit_header, time_header]
df_crop_rotations.columns = header
df_crop_rotations.to_csv(file, sep=";", index=False)