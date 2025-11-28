from pathlib import Path
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


base_path = Path(__file__).parent

# load MODFLOW parameters
path = Path(__file__).parent / "parameters_modflow.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")

# load the topography and elevation of the aquifer layers
topography = ds_params['elevations'].isel(z=0).values
# derive the model domain from the topography
mask = np.isfinite(topography)
# set Schoenberg to inactive
mask_schoenberg = (ds_params['mask_schoenberg'].values == 1)
mask = np.where(mask_schoenberg, False, mask)

# load observed groundwater heads (average values of the observation wells)
path = base_path / "observations" / "observed_groundwater_heads_average.csv"
observed_groundwater_heads = pd.read_csv(path, sep=";", skiprows=0)

# load interpolated groundwater heads
base_path = Path(__file__).parent
src = rasterio.open(str(base_path / "input" / "groundwater_heads_interpolated_50m.tif"))
gw_heads_interpolated = src.read(1)

grid_extent = (0, 777*50, 0, 621*50)
fig, axes = plt.subplots(figsize=(4, 4))
gw_heads_interpolated[~mask] = np.nan
plt.imshow(gw_heads_interpolated, cmap='terrain', aspect='equal')
plt.colorbar(label='[m]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('x-direction')
plt.ylabel('y-direction')
plt.tight_layout()
file = Path(__file__).parent / "figures" / "groundwater_heads_interpolated.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
gw_depth_interpolated = topography - gw_heads_interpolated
plt.imshow(gw_depth_interpolated, cmap='viridis', aspect='equal', vmin=0, vmax=10)
plt.colorbar(label='[m]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('x-direction')
plt.ylabel('y-direction')
plt.tight_layout()
file = Path(__file__).parent / "figures" / "groundwater_depths_interpolated.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# calculate the metrics
df_metrics = pd.DataFrame(index=range(1))
df_metrics["ME"] = np.nan
df_metrics["MAE"] = np.nan
df_metrics["RBIAS"] = np.nan
df_metrics["r"] = np.nan


# load observed groundwater heads
obs = observed_groundwater_heads.iloc[:, -1].values  # observed groundwater heads
rows = observed_groundwater_heads.iloc[:, -2].values  # row IDs of the observation wells
cols = observed_groundwater_heads.iloc[:, -3].values  # column IDs of the observation wells

# extract the simulated groundwater heads at the location of the observation wells
sim = gw_heads_interpolated[rows, cols].flatten()

# calculate mean error
df_metrics.loc[0, "ME"] = np.mean(sim - obs)
# calculate mean absolute error
df_metrics.loc[0, "MAE"] = np.mean(np.abs(sim - obs))
# calculate relative bias
df_metrics.loc[0, "RBIAS"] = np.mean((sim - obs) / obs)
# calculate spearman correlation
df_metrics.loc[0, "r"] = sp.stats.spearmanr(sim, obs)[0]


diff_sim_obs = sim - obs
cm = plt.get_cmap('PuOr')
grid_extent = (0, 777*50, 0, 621*50)
fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
wells_y = [266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]
wells_x = [66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]
plt.scatter(wells_x, wells_y, marker='x', s=5, c='black')
wells_obs_y = observed_groundwater_heads.iloc[:, -2].values  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads.iloc[:, -3].values  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, c=diff_sim_obs, s=5, cmap=cm, vmin=-10, vmax=10)
plt.colorbar(label='[m]', shrink=0.5)
plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5)
plt.grid(zorder=0)
plt.xlabel('x-direction')
plt.ylabel('y-direction')
plt.tight_layout()
file = Path(__file__).parent / "figures" / "steady-state" / "difference_int_obs.png"
fig.savefig(file, dpi=300)
plt.close(fig)


# save the metrics
path = base_path / "metrics_interpolated.csv"
df_metrics.to_csv(path, sep=";", index=False)
