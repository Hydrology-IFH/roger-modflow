from pathlib import Path
from cftime import num2date
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml


base_path = Path(__file__).parent

# load the config file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    roger_config = yaml.safe_load(file)

model_type = "transient"

output_file = base_path / "output" / model_type / "moehlin.rate.nc"
ds_roger = xr.open_dataset(output_file, engine="h5netcdf")
days = ds_roger["Time"].values / np.timedelta64(24 * 60 * 60, "s")
date = num2date(
    days,
    units=f"days since {ds_roger['Time'].attrs['time_origin']}",
    calendar="standard",
    only_use_cftime_datetimes=False,
)
dates = date[1:]
ds_roger = ds_roger.assign_coords(Time=("Time", date))

grid_extent = (0, roger_config['ny'] * roger_config['dy'], 0, roger_config['nx']*roger_config['dx'])

fig, axes = plt.subplots(1,1, figsize=(6, 6))
vals = np.nansum(ds_roger['q_ss'].values, axis=0)
vals = np.where(vals < 0, np.nan, vals)
plt.imshow(vals, extent=grid_extent, cmap='viridis_r', aspect='equal', vmin=0, vmax=100)
plt.colorbar(label='groundwater recharge [mm]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path / "figures" / "groundwater_recharge_sum_debug_oneD.png"
fig.savefig(file, dpi=300)
plt.close(fig)


fig, axes = plt.subplots(figsize=(4, 2))
axes.plot(dates, ds_roger['q_ss'].values[1:, 198, 198], lw=1.5, color='black')
axes.tick_params(axis="x", rotation=33)
axes.set_xlabel('Time [days]')
axes.set_ylabel('Groundwater recharge \n[mm/day]')
axes.set_xlim(dates[0], dates[-1])
fig.tight_layout()
file = base_path / "figures" / "groundwater_recharge_ts_debug_oneD.png"
fig.savefig(file, dpi=300)

# fig, axes = plt.subplots(1,1, figsize=(6, 6))
# vals = np.nansum(ds_roger['q_sub'].values, axis=0)
# vals = np.where(vals < 0, np.nan, vals)
# plt.imshow(vals, extent=grid_extent, cmap='viridis_r', aspect='equal', vmin=0, vmax=700)
# plt.colorbar(label='subsurface runoff [mm]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path / "figures" / "lateral_subsurface_runoff_sum_debug.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)


# # load the netcdf file
# output_file = base_path / "output" / model_type / "modflow_output.nc"
# ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
# ndays = ds_mf.sizes['time']

# fig, axes = plt.subplots(figsize=(4, 4))
# plt.imshow(np.nanmean(ds_mf['head'].isel(layer=0).values, axis=0), extent=grid_extent, vmin=100, vmax=300, cmap='viridis', aspect='equal')
# plt.colorbar(label='groundwater head [m a.s.l.]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path / "figures" / f"gw_head_transient_layer0_debug_svat.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)