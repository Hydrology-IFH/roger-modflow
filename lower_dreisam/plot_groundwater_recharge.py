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

output_file = base_path / "output" / model_type / "oneD" / "roger_output.nc"
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
vals = np.nanmean(ds_roger['q_ss'].resample(Time='YE').sum().values, axis=-1)
vals = np.where(vals < 0, np.nan, vals)
plt.imshow(vals, extent=grid_extent, cmap='viridis_r', aspect='equal', vmin=0, vmax=700)
plt.colorbar(label='groundwater recharge [mm/year]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path / "figures" / "average_annual_groundwater_recharge.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(1,1, figsize=(6, 6))
vals = np.nanmean(ds_roger['q_sub'].resample(Time='YE').sum().values, axis=-1)
vals = np.where(vals < 0, np.nan, vals)
plt.imshow(vals, extent=grid_extent, cmap='viridis_r', aspect='equal', vmin=0, vmax=700)
plt.colorbar(label='subsurface runoff [mm/year]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path / "figures" / "average_annual_lateral_subsurface_runoff.png"
fig.savefig(file, dpi=300)
plt.close(fig)