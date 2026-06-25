from pathlib import Path
import os
import numpy as np
import xarray as xr
import xesmf as xe

def aggregate_to_finer_resolution(vals, res_coarse, res_fine, method="keep", x_origin=0, y_origin=0):
    """Aggregate raster data to a finer resolution.
        
    Args
    ----
    vals : numpy.ndarray
        2D array with the raster data.

    res_coarse : int
        spatial resolution of the coarse grid in meters.

    res_fine : int
        spatial resolution of the fine grid in meters.

    method : str
        Method to aggregate the data. Options are "keep", "interpolate" and "conservative".

    x_origin : int
        x-coordinate of the grid origin (lower left corner).

    y_origin : int
        y-coordinate of the grid origin (lower left corner).  
    """
    ny_coarse, nx_coarse = vals.shape[0], vals.shape[1]
    nx_fine = int(nx_coarse * (res_coarse / res_fine))
    ny_fine = int(ny_coarse * (res_coarse / res_fine))
    meters_to_latlon = 111195
    if method == "keep":
        lat_fine = np.linspace(y_origin, y_origin + ny_fine*(res_fine/meters_to_latlon), ny_fine)/meters_to_latlon  # boundaries
        lon_fine = np.linspace(x_origin, x_origin + nx_fine*(res_fine/meters_to_latlon), nx_fine)/meters_to_latlon  # boundaries
        lat_coarse = np.linspace(y_origin, y_origin + ny_coarse*(res_coarse/meters_to_latlon), ny_coarse)/meters_to_latlon  # boundaries
        lon_coarse = np.linspace(x_origin, x_origin + nx_coarse*(res_coarse/meters_to_latlon), nx_coarse)/meters_to_latlon  # boundaries
        grid_coarse = {"lon": lon_coarse, "lat": lat_coarse}
        grid_fine = {"lon": lon_fine, "lat": lat_fine}
        regridder = xe.Regridder(grid_coarse, grid_fine, "nearest_s2d")
    elif method == "interpolate":
        lat_fine = np.linspace(y_origin, y_origin + ny_fine*(res_fine/meters_to_latlon), ny_fine)/meters_to_latlon  # boundaries
        lon_fine = np.linspace(x_origin, x_origin + nx_fine*(res_fine/meters_to_latlon), nx_fine)/meters_to_latlon  # boundaries
        lat_coarse = np.linspace(y_origin, y_origin + ny_coarse*(res_coarse/meters_to_latlon), ny_coarse)/meters_to_latlon  # boundaries
        lon_coarse = np.linspace(x_origin, x_origin + nx_coarse*(res_coarse/meters_to_latlon), nx_coarse)/meters_to_latlon  # boundaries
        grid_coarse = {"lon": lon_coarse, "lat": lat_coarse}
        grid_fine = {"lon": lon_fine, "lat": lat_fine}
        regridder = xe.Regridder(grid_coarse, grid_fine, "bilinear")
    elif method == "conservative":
        lat_fine = np.linspace(y_origin, y_origin + (ny_fine + 1) * (res_fine/meters_to_latlon), ny_fine + 1)/meters_to_latlon  # boundaries
        lon_fine = np.linspace(x_origin, x_origin + (nx_fine + 1) * (res_fine/meters_to_latlon), nx_fine + 1)/meters_to_latlon  # boundaries
        lat_coarse = np.linspace(y_origin, y_origin + (ny_coarse + 1) * (res_coarse/meters_to_latlon), ny_coarse + 1)/meters_to_latlon  # boundaries
        lon_coarse = np.linspace(x_origin, x_origin + (nx_coarse + 1) * (res_coarse/meters_to_latlon), nx_coarse + 1)/meters_to_latlon  # boundaries
        lat_fine_centers = (0.5 * (lat_fine[1] - lat_fine[0])) + lat_fine[:-1]  # centers
        lat_coarse_centers = (0.5 * (lat_coarse[1] + lat_coarse[0])) + lat_coarse[:-1]  # centers
        lon_fine_centers = (0.5 * (lon_fine[1] + lon_fine[0])) + lon_fine[:-1]  # centers
        lon_coarse_centers = (0.5 * (lon_coarse[1] + lon_coarse[0])) + lon_coarse[:-1]  # centers
        grid_coarse = {"lon": lon_coarse_centers, "lon_b": lon_coarse, "lat": lat_coarse_centers, "lat_b": lat_coarse}
        grid_fine = {"lon": lon_fine_centers, "lon_b": lon_fine, "lat": lat_fine_centers, "lat_b": lat_fine}
        regridder = xe.Regridder(grid_coarse, grid_fine, "conservative")

    data = regridder(vals)
    return data

# load MODFLOW parameters
path = Path(__file__).parent / "input" / "parameters_modflow.nc"
ds_params_modflow = xr.open_dataset(path, engine="h5netcdf")

path = Path(__file__).parent / "input" / "boundary_conditions.nc"
ds_bc_modflow = xr.open_dataset(path, engine="h5netcdf")

mask = np.isfinite(ds_params_modflow['elevations'].isel(z=0).values)
# set Schoenberg to inactive
mask_schoenberg = (ds_params_modflow["mask_schoenberg"].values == 1)
mask = np.where(mask_schoenberg, False, mask)
mask_boundary_condition_schoenberg = ds_bc_modflow["mask_schoenberg_bc"].values
mask_50 = np.where(mask_boundary_condition_schoenberg, True, mask)

mask_25 = aggregate_to_finer_resolution(mask.astype(np.float32), res_coarse=50, res_fine=25, method="keep", x_origin=ds_params_modflow.x_origin, y_origin=ds_params_modflow.y_origin)
mask_25 = mask_25.astype(bool)

mask_10 = aggregate_to_finer_resolution(mask.astype(np.float32), res_coarse=50, res_fine=10, method="keep", x_origin=ds_params_modflow.x_origin, y_origin=ds_params_modflow.y_origin)
mask_10 = mask_10.astype(bool)

path = str(Path(__file__).parent / "input" / "parameters_roger_25m.nc")
with xr.open_dataset(path) as f:
    f['maskCatch'] = (('y', 'x'), mask_25.astype(np.int32))
    f['maskCatch'].attrs.update(long_name="Mask of the catchment", standard_name="maskCatch", units="")
    _path = str(Path(__file__).parent / "input" / "_parameters_roger_25m.nc")
    f.to_netcdf(_path, mode='w')
# rename file
os.rename(_path, path)

path = str(Path(__file__).parent / "input" / "parameters_roger_10m.nc")
with xr.open_dataset(path) as f:
    f['maskCatch'] = (('y', 'x'), mask_10.astype(np.int32))
    f['maskCatch'].attrs.update(long_name="Mask of the catchment", standard_name="maskCatch", units="")
    _path = str(Path(__file__).parent / "input" / "_parameters_roger_10m.nc")
    f.to_netcdf(_path, mode='w')
# rename file
os.rename(_path, path)