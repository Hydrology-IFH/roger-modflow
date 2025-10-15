import numpy as np
import os
import pandas as pd
from pathlib import Path
import geopandas as gpd
import xarray as xr
import rasterio
import matplotlib.pyplot as plt

base_path = Path(__file__).parent


dict_xy_50m = {
    "FALKENSTEIG": [(591, 310), (591, 309)],
    "EBNET": [(430, 207), (430, 207)],
    "OBERAMBRINGEN": [(191, 360), (191, 360)],
    "UNTERMUENSTERTAL": [(218, 488), (218, 488)],
}

dict_xy_5m = {
    "FALKENSTEIG": [(5913, 3104 - i) for i in range(11)],
    "EBNET": [(4295 + i, 2084 - i) for i in range(11)],
    "OBERAMBRINGEN": [(1908 + i, 3607 - i) for i in range(11)],
    "UNTERMUENSTERTAL": [(2180 + i, 4880 - i) for i in range(11)],
}

path = base_path / "input" / "parameters_modflow.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")
dem_50m = ds_params["elevations"].isel(z=0).values

file = base_path / "input" / "dem_5m.tif"
src = rasterio.open(str(file))
dem_5m = src.read(1)
cols, rows = np.meshgrid(np.arange(dem_5m.shape[1]), np.arange(dem_5m.shape[0]))
xs, ys = rasterio.transform.xy(src.transform, rows, cols)
lons = np.array(xs)
lats = np.array(ys)

# plot profiles
for station in dict_xy_50m.keys():
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlabel("Distance along profile (m)")
    ax.set_ylabel("Elevation (m a.s.l.)")

    # plot DEM
    xy_50m = dict_xy_50m[station]
    dem_profile_50m = [dem_50m[y, x] for x, y in xy_50m]
    distance_50m = [0, 50]
    ax.plot(distance_50m, dem_profile_50m, color="black", label="DEM 50 m", linewidth=2)

    xy_5m = dict_xy_5m[station]
    dem_profile_5m = [dem_5m[y, x] for x, y in xy_5m]
    distance_5m = np.arange(len(xy_5m)) * 5
    ax.plot(distance_5m, dem_profile_5m, color="black", label="DEM 5 m", linewidth=1, ls="--")

    fig.tight_layout()
    file = base_path / "figures" / f"stage_profile_{station.lower()}_dem.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

