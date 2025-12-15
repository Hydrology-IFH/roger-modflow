from pathlib import Path
import xarray as xr
import geoxarray
import numpy as onp
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar

base_path = Path(__file__).parent

params_file = base_path / "input" / "parameters_roger_25m_.nc"
ds_params = xr.open_dataset(params_file)
xcoords = ds_params.x.values
ycoords = ds_params.y.values
mask = ds_params['maskCatch'].values
cond_catch = mask == 1
lu_ids = onp.copy(ds_params['lanu'].values)
lu_ids[~cond_catch] = -9999  # set non-catchment area to -9999

# load the netcdf file
file = base_path / "input" / "crops_2018-2022.nc"
ds_cr_2018_2022 = xr.open_dataset(file)
spatial_ref = ds_cr_2018_2022.spatial_ref
lu_ids_2018_2022 = ds_cr_2018_2022['Nutzcode'].values
cond = onp.isnan(lu_ids_2018_2022)
lu_ids_2018_2022[cond] = -9999  # set nan to
lu_ids_2018_2022 = lu_ids_2018_2022.astype(onp.int16)

cond = (lu_ids_2018_2022 == 6).any(axis=0)
lu_ids[cond] = 6
cond = (lu_ids_2018_2022 == 7).any(axis=0)
lu_ids[cond] = 7
ll_ga_ids = onp.arange(501, 598).tolist() + [599, 8, 81, 82]
cond_ga = onp.isin(lu_ids_2018_2022, ll_ga_ids).all(axis=0)
cond5 = (lu_ids == 5)

cropland_gaps = onp.zeros(lu_ids.shape, dtype=float)
cropland_gaps[:, :] = onp.nan
cropland_gaps[~cond_ga & cond5] = 1.0

cropland_ga = onp.zeros(lu_ids.shape, dtype=float)
cropland_ga[:, :] = onp.nan
cropland_5 = onp.zeros(lu_ids.shape, dtype=float)
cropland_5[:, :] = onp.nan
cropland_5[cond5] = 1.0
cropland_ga[cond_ga] = 1.0

# update land use ids in cropland gaps with crop rotations from 2018-2022
lu_ids_updated = lu_ids.copy()
lu_ids_updated[~cond_catch] = ds_params['lanu'].values[~cond_catch]
lu_ids_updated[cond_ga] = 5
cond5_ = (lu_ids_updated == 5)
lu_ids_updated[~cond_ga & cond5_] = 8

file = base_path / "input" / "crop_rotations_2018_2022.nc"
ds_crop_rotations_2018_2022 = xr.open_dataset(file)
summer_crops_2018_2022 = ds_crop_rotations_2018_2022['summer_crops'].values


print(onp.nansum(cropland_gaps))  # in grid cells
print(onp.nansum(cropland_gaps) * 0.0625)  # in ha
print(onp.nansum(cropland_gaps)/onp.nansum(cropland_ga))  # area share of gaps compared to GA cropland
print(onp.nansum(cropland_gaps)/onp.nansum(mask))  # area share of gaps compared to catchment area

grid_extent = (ds_params.x.values[0], ds_params.x.values[-1], ds_params.y.values[-1], ds_params.y.values[0])

fig, axes = plt.subplots(figsize=(6, 5))
axes.imshow(cropland_gaps, extent=grid_extent, cmap='Oranges', alpha=1, aspect='equal', vmin=0, vmax=1, zorder=3)
ctx.add_basemap(axes, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:25832", zorder=1, alpha=0.5)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
north_arrow(
    axes, scale=0.2, location="upper right", rotation={"crs": 25832, "reference": "center"}
)
scale_bar(axes, location="lower right", style="boxes", bar={"projection": 25832, "minor_type": "none"}, labels={"style": "first_last"})
plt.tight_layout()
file = base_path / "figures" / "cropland_gaps.png"
fig.savefig(file, dpi=300)
plt.close("all")

fig, axes = plt.subplots(figsize=(6, 5))
cb = axes.imshow(cropland_ga, extent=grid_extent, cmap='Purples', alpha=1, aspect='equal', vmin=0, vmax=1, zorder=3)
ctx.add_basemap(axes, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:25832", zorder=1, alpha=0.5)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
north_arrow(
    axes, scale=0.2, location="upper right", rotation={"crs": 25832, "reference": "center"}
)
scale_bar(axes, location="lower right", style="boxes", bar={"projection": 25832, "minor_type": "none"}, labels={"style": "first_last"})
plt.tight_layout()
file = base_path / "figures" / "cropland_GA.png"
fig.savefig(file, dpi=300)
plt.close("all")

fig, axes = plt.subplots(figsize=(6, 5))
cb = axes.imshow(cropland_5, extent=grid_extent, cmap='Greys', alpha=1, aspect='equal', vmin=0, vmax=1, zorder=3)
ctx.add_basemap(axes, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:25832", zorder=1, alpha=0.5)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
north_arrow(
    axes, scale=0.2, location="upper right", rotation={"crs": 25832, "reference": "center"}
)
scale_bar(axes, location="lower right", style="boxes", bar={"projection": 25832, "minor_type": "none"}, labels={"style": "first_last"})
plt.tight_layout()
file = base_path / "figures" / "cropland_5.png"
fig.savefig(file, dpi=300)
plt.close("all")

cond5 = lu_ids == 5
lu_ids[~cond_ga & cond5] = 8
cropland_gaps = onp.zeros(lu_ids.shape, dtype=float)
cropland_gaps[:, :] = onp.nan
cropland_gaps[~cond_ga & cond5] = 1.0
fig, axes = plt.subplots(figsize=(6, 5))
axes.imshow(cropland_gaps, extent=grid_extent, cmap='Oranges', alpha=1, aspect='equal', vmin=0, vmax=1, zorder=3)
ctx.add_basemap(axes, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:25832", zorder=1, alpha=0.5)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
north_arrow(
    axes, scale=0.2, location="upper right", rotation={"crs": 25832, "reference": "center"}
)
scale_bar(axes, location="lower right", style="boxes", bar={"projection": 25832, "minor_type": "none"}, labels={"style": "first_last"})
plt.tight_layout()
file = base_path / "figures" / "cropland_gaps_filled_with_8.png"
fig.savefig(file, dpi=300)
plt.close("all")

cond5 = lu_ids_updated == 5
cond_598 = summer_crops_2018_2022[0, :, :] == 598
cropland_gaps = onp.zeros(lu_ids.shape, dtype=float)
cropland_gaps[:, :] = onp.nan
cropland_gaps[cond_598 & cond5] = 1.0
cropland_gaps[~cond_catch] = onp.nan

fig, axes = plt.subplots(figsize=(6, 5))
axes.imshow(cropland_gaps, extent=grid_extent, cmap='Oranges', alpha=1, aspect='equal', vmin=0, vmax=1, zorder=3)
ctx.add_basemap(axes, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:25832", zorder=1, alpha=0.5)
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
north_arrow(
    axes, scale=0.2, location="upper right", rotation={"crs": 25832, "reference": "center"}
)
scale_bar(axes, location="lower right", style="boxes", bar={"projection": 25832, "minor_type": "none"}, labels={"style": "first_last"})
plt.tight_layout()
file = base_path / "figures" / "cropland_gaps_.png"
fig.savefig(file, dpi=300)
plt.close("all")