import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import xesmf as xe


def aggregate_to_coarser_resolution(vals, res_fine, res_coarse, method="sum", x_origin=0, y_origin=0):
    """Aggregate raster data to a coarser resolution.
    
    Args
    ----
    vals : numpy.ndarray
        2D array with the raster data.

    res_fine : int
        spatial resolution of the fine grid in meters.

    res_coarse : int
        spatial resolution of the coarse grid in meters.

    method : str
        Method to aggregate the data. Options are "sum" and "average".

    x_origin : int
        x-coordinate of the grid origin (lower left corner).

    y_origin : int
        y-coordinate of the grid origin (lower left corner).  
    """
    ny_fine, nx_fine = vals.shape[0], vals.shape[1]
    nlat_coarse, nlon_coarse = int(res_coarse / res_fine), int(res_coarse / res_fine)
    meters_to_latlon = 111195
    lat_fine = np.linspace(y_origin, y_origin + ny_fine*(res_fine/meters_to_latlon), ny_fine)/meters_to_latlon  # boundaries
    lon_fine = np.linspace(x_origin, x_origin + nx_fine*(res_fine/meters_to_latlon), nx_fine)/meters_to_latlon  # boundaries

    arr_fine = xr.DataArray(vals, coords={"lat": lat_fine, "lon": lon_fine}, dims=["lat", "lon"])

    if method == "sum":
        arr_coarse = xr.DataArray(
            arr_fine.coarsen(lat=nlat_coarse, lon=nlon_coarse).sum().values,
            dims=("lat", "lon"),
        )
        
    elif method == "average":
        arr_coarse = xr.DataArray(
            arr_fine.coarsen(lat=nlat_coarse, lon=nlon_coarse).mean().values,
            dims=("lat", "lon"),
        )
    return arr_coarse.values


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


base_path = Path(__file__).parent
base_path_figs = base_path / "figures"

# aggregate to coarser resolution
x_origin = 0
y_origin = 0
# nx_in, ny_in = 4, 8 # size of input data
# nx_out, ny_out = 2, 4 # size of output data
# res_in = 25
# res_out = 50
# grid_extent = (x_origin, x_origin + nx_in*res_in, y_origin, y_origin + ny_in*res_in)
# data_in = np.arange(nx_in * ny_in).reshape(ny_in, nx_in)


res_in = 0.25
res_out = 5
ds_in = xe.util.grid_global(res_in, res_in)  # input grid
data_in = xe.data.wave_smooth(ds_in["lon"], ds_in["lat"]).values
nx_in, ny_in = data_in.shape[0], data_in.shape[1]
nx_out, ny_out = int(nx_in * (res_out/res_in)), int(ny_in * (res_out/res_in))
grid_extent = (x_origin, x_origin + nx_in*res_in, y_origin, y_origin + ny_in*res_in)

data_out = aggregate_to_coarser_resolution(data_in, res_in, res_out, method="sum")
fig, axes = plt.subplots(2, 1, figsize=(3, 6))
ax1 = axes[0].imshow(data_in, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.max(data_in)))
axes[0].set_xlabel('Distance in y-direction [m]')
axes[0].set_ylabel('Distance in x-direction [m]')
fig.colorbar(ax1, shrink=0.5)
ax2 = axes[1].imshow(data_out, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.max(data_out)))
axes[1].set_xlabel('Distance in y-direction [m]')
axes[1].set_ylabel('Distance in x-direction [m]')
fig.colorbar(ax2, shrink=0.5)
fig.tight_layout()
file = base_path_figs / "test_sum_aggregate_to_coarse.png"
fig.savefig(file, dpi=300)
plt.close("all")

data_out = aggregate_to_coarser_resolution(data_in, res_in, res_out,  method="average")
fig, axes = plt.subplots(2, 1, figsize=(3, 6))
ax1 = axes[0].imshow(data_in, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.max(data_in)))
axes[0].set_xlabel('Distance in y-direction [m]')
axes[0].set_ylabel('Distance in x-direction [m]')
fig.colorbar(ax1, shrink=0.5)
ax2 = axes[1].imshow(data_out, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.max(data_out)))
axes[1].set_xlabel('Distance in y-direction [m]')
axes[1].set_ylabel('Distance in x-direction [m]')
fig.colorbar(ax2, shrink=0.5)
fig.tight_layout()
file = base_path_figs / "test_average_aggregate_to_coarse.png"
fig.savefig(file, dpi=300)
plt.close("all")

# aggregate to finer resolution
x_origin = 0
y_origin = 0
# nx_in, ny_in = 2, 4 # size of input data
# nx_out, ny_out = 4, 8 # size of output data
# res_in = 50
# res_out = 25
# data_in = np.arange(nx_in * ny_in).reshape(ny_in, nx_in)

res_in = 5
res_out = 0.25
ds_in = xe.util.grid_global(res_in, res_in)  # input grid
data_in = xe.data.wave_smooth(ds_in["lon"], ds_in["lat"]).values
nx_in, ny_in = data_in.shape[0], data_in.shape[1]
nx_out, ny_out = int(nx_in * (res_in/res_out)), int(ny_in * (res_in/res_out))
grid_extent = (x_origin, x_origin + nx_in*res_in, y_origin, y_origin + ny_in*res_in)

data_out = aggregate_to_finer_resolution(data_in, res_in, res_out, method="keep")
fig, axes = plt.subplots(2, 1, figsize=(3, 6))
ax1 = axes[0].imshow(data_in, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.max(data_in)))
axes[0].set_xlabel('Distance in y-direction [m]')
axes[0].set_ylabel('Distance in x-direction [m]')
fig.colorbar(ax1, shrink=0.5)
ax2 = axes[1].imshow(data_out, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.max(data_out)))
axes[1].set_xlabel('Distance in y-direction [m]')
axes[1].set_ylabel('Distance in x-direction [m]')
fig.colorbar(ax2, shrink=0.5)
fig.tight_layout()
file = base_path_figs / "test_keep_aggregate_to_finer.png"
fig.savefig(file, dpi=300)
plt.close("all")

data_out = aggregate_to_finer_resolution(data_in, res_in, res_out, method="interpolate")
fig, axes = plt.subplots(2, 1, figsize=(3, 6))
ax1 = axes[0].imshow(data_in, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.max(data_in)))
axes[0].set_xlabel('Distance in y-direction [m]')
axes[0].set_ylabel('Distance in x-direction [m]')
fig.colorbar(ax1, shrink=0.5)
ax2 = axes[1].imshow(data_out, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.max(data_out)))
axes[1].set_xlabel('Distance in y-direction [m]')
axes[1].set_ylabel('Distance in x-direction [m]')
fig.colorbar(ax2, shrink=0.5)
fig.tight_layout()
file = base_path_figs / "test_interpolate_aggregate_to_finer.png"
fig.savefig(file, dpi=300)
plt.close("all")

data_out = aggregate_to_finer_resolution(data_in, res_in, res_out, method="conservative")
fig, axes = plt.subplots(2, 1, figsize=(3, 6))
ax1 = axes[0].imshow(data_in, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.max(data_in)))
axes[0].set_xlabel('Distance in y-direction [m]')
axes[0].set_ylabel('Distance in x-direction [m]')
fig.colorbar(ax1, shrink=0.5)
ax2 = axes[1].imshow(data_out, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.max(data_in)))
axes[1].set_xlabel('Distance in y-direction [m]')
axes[1].set_ylabel('Distance in x-direction [m]')
fig.colorbar(ax2, shrink=0.5)
fig.tight_layout()
file = base_path_figs / "test_conservative_aggregate_to_finer.png"
fig.savefig(file, dpi=300)
plt.close("all")

