from cftime import num2date
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
import imageio


base_path = Path(__file__).parent

# load the netcdf file
params_file = base_path / "parameters.nc"
ds_params = xr.open_dataset(params_file, engine="h5netcdf")
roger_mask = ds_params["GRUND"].values/10 > 0 

# load the config file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    roger_config = yaml.safe_load(file)

res_modflow = 50  # spatial resolution of MODFLOW in meters

modflow_config = {
    'dx': int(roger_config['dx']*(res_modflow/roger_config['dx'])),
    'dy': int(roger_config['dy']*(res_modflow/roger_config['dx'])),
    'nx': int(roger_config['nx']/(res_modflow/roger_config['dx'])),
    'ny': int(roger_config['ny']/(res_modflow/roger_config['dx'])),
    'nz': 4,
}

# load the netcdf file
model_type = "transient"
output_file = base_path / "output" / model_type / "modflow_output.nc"
ds_mf = xr.open_dataset(output_file, engine="h5netcdf")

output_file = base_path / "output" / model_type / "roger_output.nc"
ds_roger = xr.open_dataset(output_file, engine="h5netcdf")
days = ds_roger["Time"].values / np.timedelta64(24 * 60 * 60, "s")
date = num2date(
    days,
    units=f"days since {ds_roger['Time'].attrs['time_origin']}",
    calendar="standard",
    only_use_cftime_datetimes=False,
)
dates = date[1:]

# plot the heads for all time steps
grid_extent = (-2000, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])
frames = []
for t in range(30, ds_mf.sizes['time'], 30):
    fig, axes = plt.subplots(2,1, figsize=(6, 6))

    arr = np.sum(ds_roger['q_ss'].values[:, :, t-30:t], axis=-1)
    arr = np.where(roger_mask, arr, np.nan)
    empty_arr = np.zeros((roger_config['nx'], int(2000/roger_config['dy'])))
    empty_arr[:, :] = np.nan
    arr1 = np.concatenate((empty_arr, arr), axis=1)
    cb0 = axes[0].imshow(arr1, extent=grid_extent, vmin=0, vmax=200, cmap='viridis', aspect='equal')
    fig.colorbar(cb0, ax=axes[0], label='groundwater recharge \n [mm/30 days]', shrink=0.9)
    axes[0].grid(zorder=0)
    axes[0].set_xlabel('Distance in x-direction [m]')
    axes[0].set_ylabel('Distance in y-direction [m]')
    axes[0].set_title(f'{date[t].strftime("%d %B %Y")}', fontsize=10)

    arr = ds_mf['head'].isel(time=t, layer=0).values
    empty_arr = np.zeros((modflow_config['nx'], int(2000/modflow_config['dy'])))
    empty_arr[:, :] = np.nan
    arr1 = np.concatenate((empty_arr, arr), axis=1)
    axes[1].text(-1900, 5400, "observation\nwell", fontsize=9, color='black', bbox=dict(facecolor='white', edgecolor="none", alpha=0.5))
    axes[1].scatter(-1350, 96*50, marker='x', s=20, c='black')  # approximate location of the observation well
    axes[1].scatter(150, 96*50, marker='x', s=20, c='red')  # approximate location of the observation well
    cb1 = axes[1].imshow(arr1, extent=grid_extent, vmin=200, vmax=700, cmap='viridis', aspect='equal')
    fig.colorbar(cb1, ax=axes[1], label='groundwater head \n [m a.s.l.]', shrink=0.9)
    axes[1].grid(zorder=0)
    axes[1].set_xlabel('Distance in x-direction [m]')
    axes[1].set_ylabel('Distance in y-direction [m]')
    fig.tight_layout()
    file = base_path / "figures" / "transient" / f"recharge_gw_heads_{t}_transient_layer0_gif.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)
    img = imageio.v2.imread(file)
    frames.append(img)

file = base_path / "figures" / "transient" / "recharge_gw_heads.gif"
imageio.mimsave(file,
                frames,
                fps = 1)
