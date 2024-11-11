from cftime import num2date
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
import imageio
import seaborn as sns
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

# paper style
mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["legend.title_fontsize"] = 8
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.0,
    },
)

# presentation style
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["legend.title_fontsize"] = 10
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "legend.title_fontsize": 10.0,
    },
)


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
grid_extent = (0, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])


# plot the heads for all time steps
frames = []
for t in range(30, ds_mf.sizes['time'], 30):
    fig, axes = plt.subplots(1,1, figsize=(4, 4))

    arr = ds_mf['head'].isel(time=t, layer=3).values
    cb1 = axes.imshow(arr, extent=grid_extent, vmin=250, vmax=260, cmap='viridis', aspect='equal')
    fig.colorbar(cb1, ax=axes, label='groundwater head \n [m a.s.l.]', shrink=0.5)
    axes.grid(zorder=0)
    axes.set_xlabel('Distance in x-direction [m]')
    axes.set_ylabel('Distance in y-direction [m]')
    axes.set_title(f'{date[t].strftime("%d %B %Y")}', fontsize=10)
    fig.tight_layout()
    file = base_path / "figures" / "transient" / f"gw_heads_{t}_transient_layer3_gif.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)
    img = imageio.v2.imread(file)
    frames.append(img)

file = base_path / "figures" / "transient" / "gw_heads.gif"
imageio.mimsave(file,
                frames,
                fps = 1)

frames = []
for t in range(30, ds_mf.sizes['time'], 30):
    fig, axes = plt.subplots(1,1, figsize=(4, 4))

    arr = ds_roger['theta'].values[:, :, t]
    arr = np.where(roger_mask, arr, np.nan)
    cb1 = axes.imshow(arr, extent=grid_extent, vmin=0.15, vmax=0.45, cmap='viridis_r', aspect='equal')
    fig.colorbar(cb1, ax=axes, label=r'$\theta$ [-]', shrink=0.5)
    axes.grid(zorder=0)
    axes.set_xlabel('Distance in x-direction [m]')
    axes.set_ylabel('Distance in y-direction [m]')
    axes.set_title(f'{date[t].strftime("%d %B %Y")}', fontsize=10)
    fig.tight_layout()
    file = base_path / "figures" / "transient" / f"theta_{t}_gif.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)
    img = imageio.v2.imread(file)
    frames.append(img)

file = base_path / "figures" / "transient" / "theta.gif"
imageio.mimsave(file,
                frames,
                fps = 1)

frames = []
for t in range(1, 366, 1):
    fig, axes = plt.subplots(1,1, figsize=(4, 4))

    arr = ds_roger['theta'].values[:, :, t]
    arr = np.where(roger_mask, arr, np.nan)
    cb1 = axes.imshow(arr, extent=grid_extent, vmin=0.15, vmax=0.45, cmap='viridis_r', aspect='equal')
    fig.colorbar(cb1, ax=axes, label=r'$\theta$ [-]', shrink=0.5)
    axes.grid(zorder=0)
    axes.set_xlabel('Distance in x-direction [m]')
    axes.set_ylabel('Distance in y-direction [m]')
    axes.set_title(f'{date[t].strftime("%d %B %Y")}', fontsize=10)
    fig.tight_layout()
    file = base_path / "figures" / "transient" / f"theta_365_{t}_gif.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)
    img = imageio.v2.imread(file)
    frames.append(img)

file = base_path / "figures" / "transient" / "theta_365.gif"
imageio.mimsave(file,
                frames,
                fps = 3)


# # plot the recharge and heads for all time steps
# frames = []
# for t in range(30, ds_mf.sizes['time'], 30):
#     fig, axes = plt.subplots(2,1, figsize=(6, 6))

#     arr = np.sum(ds_roger['q_ss'].values[:, :, t-30:t], axis=-1)
#     arr = np.where(roger_mask, arr, np.nan)
#     empty_arr = np.zeros((roger_config['nx'], int(2000/roger_config['dy'])))
#     empty_arr[:, :] = np.nan
#     arr1 = np.concatenate((empty_arr, arr), axis=1)
#     cb0 = axes[0].imshow(arr1, extent=grid_extent, vmin=0, vmax=50, cmap='viridis_r', aspect='equal')
#     fig.colorbar(cb0, ax=axes[0], label='groundwater recharge \n [mm/30 days]', shrink=0.9)
#     axes[0].grid(zorder=0)
#     axes[0].set_xlabel('Distance in x-direction [m]')
#     axes[0].set_ylabel('Distance in y-direction [m]')
#     axes[0].set_title(f'{date[t].strftime("%d %B %Y")}', fontsize=10)

#     arr = ds_mf['head'].isel(time=t, layer=3).values
#     empty_arr = np.zeros((modflow_config['nx'], int(2000/modflow_config['dy'])))
#     empty_arr[:, :] = np.nan
#     arr1 = np.concatenate((empty_arr, arr), axis=1)
#     axes[1].text(-1900, 5400, "observation\nwell", fontsize=9, color='black', bbox=dict(facecolor='white', edgecolor="none", alpha=0.5))
#     axes[1].scatter(-1350, 96*50, marker='x', s=20, c='black')  # approximate location of the observation well
#     axes[1].scatter(150, 96*50, marker='x', s=20, c='red')  # approximate location of the observation well
#     cb1 = axes[1].imshow(arr1, extent=grid_extent, vmin=100, vmax=400, cmap='viridis', aspect='equal')
#     fig.colorbar(cb1, ax=axes[1], label='groundwater head \n [m a.s.l.]', shrink=0.9)
#     axes[1].grid(zorder=0)
#     axes[1].set_xlabel('Distance in x-direction [m]')
#     axes[1].set_ylabel('Distance in y-direction [m]')
#     fig.tight_layout()
#     file = base_path / "figures" / "transient" / f"recharge_gw_heads_{t}_transient_layer3_gif.png"
#     fig.savefig(file, dpi=300)
#     plt.close(fig)
#     img = imageio.v2.imread(file)
#     frames.append(img)

# file = base_path / "figures" / "transient" / "recharge_gw_heads.gif"
# imageio.mimsave(file,
#                 frames,
#                 fps = 1)


# frames = []
# for t in range(30, ds_mf.sizes['time'], 30):
#     fig, axes = plt.subplots(2,1, figsize=(6, 6))

#     arr = np.sum(ds_roger['q_ss'].values[:, :, t-30:t], axis=-1)
#     arr = np.where(roger_mask, arr, np.nan)
#     empty_arr = np.zeros((roger_config['nx'], int(2000/roger_config['dy'])))
#     empty_arr[:, :] = np.nan
#     arr1 = np.concatenate((empty_arr, arr), axis=1)
#     cb0 = axes[0].imshow(arr1, extent=grid_extent, vmin=0, vmax=100, cmap='viridis_r', aspect='equal')
#     fig.colorbar(cb0, ax=axes[0], label='Grundwasserneubildung \n [mm/30 Tage]', shrink=0.9)
#     axes[0].grid(zorder=0)
#     axes[0].set_xlabel('Distanz in x-Richtung [m]')
#     axes[0].set_ylabel('Distanz in y-Richtung [m]')
#     axes[0].set_title(f'{date[t].strftime("%d %B %Y")}', fontsize=10)

#     arr = ds_mf['head'].isel(time=t, layer=3).values
#     empty_arr = np.zeros((modflow_config['nx'], int(2000/modflow_config['dy'])))
#     empty_arr[:, :] = np.nan
#     arr1 = np.concatenate((empty_arr, arr), axis=1)
#     axes[1].text(-1900, 5400, "Grundwasser-\nmessstelle", fontsize=9, color='black', bbox=dict(facecolor='white', edgecolor="none", alpha=0.5))
#     axes[1].scatter(-1350, 96*50, marker='x', s=20, c='black')  # approximate location of the observation well
#     axes[1].scatter(150, 96*50, marker='x', s=20, c='red')  # approximate location of the observation well
#     cb1 = axes[1].imshow(arr1, extent=grid_extent, vmin=100, vmax=400, cmap='viridis_r', aspect='equal')
#     fig.colorbar(cb1, ax=axes[1], label='Grundwasserspiegel \n [m ü.NN]', shrink=0.9)
#     axes[1].grid(zorder=0)
#     axes[1].set_xlabel('Distanz in x-Richtung [m]')
#     axes[1].set_ylabel('Distanz in y-Richtung [m]')
#     fig.tight_layout()
#     file = base_path / "figures" / "transient" / f"recharge_gw_heads_{t}_transient_layer3_gif_ger.png"
#     fig.savefig(file, dpi=300)
#     plt.close(fig)
#     img = imageio.v2.imread(file)
#     frames.append(img)

# file = base_path / "figures" / "transient" / "recharge_gw_heads_ger.gif"
# imageio.mimsave(file,
#                 frames,
#                 fps = 1)


# t = 200
# arr = ds_roger['q_ss'].values[:, :, t]
# arr = np.where(roger_mask, arr, np.nan)
# empty_arr = np.zeros((roger_config['nx'], int(2000/roger_config['dy'])))
# empty_arr[:, :] = np.nan
# arr1 = np.concatenate((empty_arr, arr), axis=1)
# fig, axes = plt.subplots(2,1, figsize=(6, 6))
# cb0 = axes[0].imshow(arr1, extent=grid_extent, vmin=0, vmax=10, cmap='viridis_r', aspect='equal')
# axes[0].scatter(15*50, 92*50, marker='x', s=20, c='magenta')
# axes[0].scatter(168*50, 60*50, marker='x', s=20, c='grey') 
# fig.colorbar(cb0, ax=axes[0], label='Grundwasserneubildung \n [mm/Tag]', shrink=0.9)
# axes[0].grid(zorder=0)
# axes[0].set_xlabel('Distanz in x-Richtung [m]')
# axes[0].set_ylabel('Distanz in y-Richtung [m]')
# axes[0].set_title(f'{date[t].strftime("%d %B %Y")}', fontsize=10)

# arr = ds_mf['head'].isel(time=t, layer=3).values
# empty_arr = np.zeros((modflow_config['nx'], int(2000/modflow_config['dy'])))
# empty_arr[:, :] = np.nan
# arr1 = np.concatenate((empty_arr, arr), axis=1)
# axes[1].text(-1900, 5400, "Grundwasser-\nmessstelle", fontsize=9, color='black', bbox=dict(facecolor='white', edgecolor="none", alpha=0.5))
# axes[1].scatter(-1350, 96*50, marker='x', s=20, c='black')  # approximate location of the observation well
# axes[1].scatter(15*50, 92*50, marker='x', s=20, c='magenta')
# axes[1].scatter(168*50, 60*50, marker='x', s=20, c='grey') 
# cb1 = axes[1].imshow(arr1, extent=grid_extent, vmin=100, vmax=400, cmap='viridis_r', aspect='equal')
# fig.colorbar(cb1, ax=axes[1], label='Grundwasserspiegel \n [m ü.NN]', shrink=0.9)
# axes[1].grid(zorder=0)
# axes[1].set_xlabel('Distanz in x-Richtung [m]')
# axes[1].set_ylabel('Distanz in y-Richtung [m]')
# fig.tight_layout()
# file = base_path / "figures" / "transient" / f"recharge_gw_heads_{t}_transient_layer3_gif_ger.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)
    