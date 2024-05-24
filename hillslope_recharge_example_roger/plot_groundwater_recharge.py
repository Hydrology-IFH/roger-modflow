import os
from pathlib import Path
import matplotlib.pyplot as plt
import xarray as xr
import yaml


base_path = Path(__file__).parent
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# load the config file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    config = yaml.safe_load(file)

# load the netcdf file
output_file = base_path / "output" / "transient" / "roger_output.nc"
ds_roger = xr.open_dataset(output_file, engine="h5netcdf")

grid_extent = (0, config['nx']*config['dx'], 0, config['ny']*config['dy'])

# plot soil water content for all time steps
for t in range(1, ds_roger.sizes['Time'], 30):
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(ds_roger['theta'].isel(Time=t).values.T, extent=grid_extent, cmap='Blues', vmin=0.18, vmax=0.42)
    plt.colorbar(label='[-]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / "transient" / f"theta_{t}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

# plot groundwater recharge for all time steps
for t in range(1, ds_roger.sizes['Time']):
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(ds_roger['q_ss'].isel(Time=t).values.T, extent=grid_extent, cmap='Blues', vmin=0, vmax=5)
    plt.colorbar(label='[mm/day]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / "transient" / f"recharge_{t}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)