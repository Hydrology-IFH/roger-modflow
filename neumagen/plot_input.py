import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from flopy.utils import Raster
import yaml

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join("..", ".."))
    sys.path.append(fpth)
    import flopy

base_path = Path(__file__).parent
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

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

# load topography and domain
file = base_path / "input" / "elevation.grd"
layer_elevations = Raster.load(file)
topography = layer_elevations.get_array(1)
elevation_bottom_layer1 = layer_elevations.get_array(2)
topography[topography <= elevation_bottom_layer1] = elevation_bottom_layer1[topography <= elevation_bottom_layer1]
elevation_bottom_layer2 = layer_elevations.get_array(3)
elevation_bottom_layer3 = layer_elevations.get_array(4)
elevation_bottom_layer4 = layer_elevations.get_array(5)
elevation_bottom_layer4[elevation_bottom_layer4 <= 100] = 100
elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

file = base_path / "input" / "domain.grd"
domain = Raster.load(file)
mask = (domain.get_array(1) == 1)
grid_extent = (0, (modflow_config['ny']*modflow_config['dy']) / 1000, 0, (modflow_config['nx']*modflow_config['dx']) / 1000)

basin = np.empty_like(domain.get_array(1))
basin[mask] = 1
fig, axes = plt.subplots(figsize=(4, 4))
plt.imshow(basin, extent=grid_extent, cmap='Greys', aspect='equal')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [km]')
plt.ylabel('Distance in y-direction [km]')
plt.tight_layout()
file = base_path_figs / "basin.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
plt.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal')
plt.colorbar(label='[m a.s.l.]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [km]')
plt.ylabel('Distance in y-direction [km]')
plt.tight_layout()
file = base_path_figs / "topography.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
cb = axes.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal')
plt.scatter(10*50, (178-82)*50, marker='x', s=20, c='black')  # location of the extraction well 1
plt.scatter(40*50, (178-92)*50, marker='x', s=20, c='black')  # location of the extraction well 2
plt.scatter(85*50, (178-62)*50, marker='x', s=20, c='black')  # location of the extraction well 3
fig.colorbar(cb , ax=axes, label='[m a.s.l.]', shrink=0.5)
axes.grid(zorder=0)
axes.set_xlabel('Distance in x-direction [km]')
axes.set_ylabel('Distance in y-direction [km]')
plt.tight_layout()
file = base_path_figs / "topography_well_locations.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(5, 6))
grid_extent1 = (-2000, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])
topography[~mask] = np.nan
empty_arr = np.zeros((modflow_config['nx'], int(2000/modflow_config['dy'])))
empty_arr[:, :] = np.nan
topography1 = np.concatenate((empty_arr, topography), axis=1)
plt.scatter(-1350, 96*50, marker='x', s=20, c='black')  # approximate location of the observation well
plt.imshow(topography1, extent=grid_extent1, cmap='terrain', aspect='equal')
plt.colorbar(label='[m a.s.l.]', shrink=0.34)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [km]')
plt.ylabel('Distance in y-direction [km]')
fig.tight_layout()
file = base_path_figs / "topography1.png"
fig.savefig(file, dpi=300)
plt.close(fig)

for i, elevation_bottom_layer in enumerate(elevation_bottom_layers):
    i = i + 1
    fig, axes = plt.subplots(figsize=(4, 4))
    elevation_bottom_layer[~mask] = np.nan
    plt.imshow(elevation_bottom_layer, extent=grid_extent, cmap='viridis', vmin=-100, vmax=1200, aspect='equal')
    plt.colorbar(label='[m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.tight_layout()
    file = base_path_figs / f"elevation_bottom_layer_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

file = base_path / "input" / "initial_conditions.grd"
initial_conditions = Raster.load(file)
initial_conditions_layer1 = initial_conditions.get_array(1)
initial_conditions_layer2 = initial_conditions.get_array(2)
initial_conditions_layer3 = initial_conditions.get_array(3)
initial_conditions_layer4 = initial_conditions.get_array(4)
initial_conditions_layers = [initial_conditions_layer1, initial_conditions_layer2, initial_conditions_layer3, initial_conditions_layer4]
for i, initial_conditions_layer in enumerate(initial_conditions_layers):
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(initial_conditions_layer, extent=grid_extent, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater head [m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.tight_layout()
    file = base_path_figs / f"initial_conditions_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

file = base_path / "input" / "boundary_condition.grd"
boundary_condition = Raster.load(file).get_array(1)
mask1 = (boundary_condition == -1)
boundary_condition1 = np.zeros((modflow_config['nx'], modflow_config['ny']+1))
boundary_condition1[:, :] = np.nan
boundary_condition1[mask] = 0.2
boundary_condition1[mask1] = 1
fig, axes = plt.subplots(figsize=(4, 4))
plt.imshow(boundary_condition1, extent=grid_extent, cmap='Greys', vmin=0, vmax=1, aspect='equal')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [km]')
plt.ylabel('Distance in y-direction [km]')
plt.tight_layout()
file = base_path_figs / "boundary_condition.png"
fig.savefig(file, dpi=300)
plt.close(fig)

file = base_path / "input" / "river_reach.csv"
river_reach = pd.read_csv(file, delimiter=",")
river_reach_arr = np.zeros((modflow_config['nx'], modflow_config['ny']+1))
river_reach_arr[:, :] = np.nan
for x, y in zip(river_reach['i'], river_reach['j']):
    river_reach_arr[x, y] = 1
river_reach_arr = np.where(mask, river_reach_arr, np.nan)
fig, axes = plt.subplots(figsize=(4, 4))
cb = axes.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal')
fig.colorbar(cb, ax=axes, label='[m a.s.l.]', shrink=0.5)
axes.imshow(river_reach_arr, extent=grid_extent, cmap='Blues', vmin=0, vmax=1.5, aspect='equal')
axes.grid(zorder=0)
axes.set_xlabel('Distance in x-direction [km]')
axes.set_ylabel('Distance in y-direction [km]')
fig.tight_layout()
file = base_path_figs / "river_reach.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
plt.imshow(boundary_condition1, extent=grid_extent, cmap='Greys', vmin=0, vmax=1, aspect='equal')
axes.imshow(river_reach_arr, extent=grid_extent, cmap='Blues', vmin=0, vmax=1.5, aspect='equal')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [km]')
plt.ylabel('Distance in y-direction [km]')
plt.tight_layout()
file = base_path_figs / "boundary_condition_and_river_reach.png"
fig.savefig(file, dpi=300)
plt.close(fig)


# fig, axes = plt.subplots(figsize=(4, 4))
# axes.scatter(river_reach['j'].values[::-1]*modflow_config['dx'], river_reach['i'].values[::-1]*modflow_config['dy'], s=river_reach['rwid'].values*0.02, c='blue')
# cb = axes.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal')
# fig.colorbar(cb, ax=axes, label='[m a.s.l.]', shrink=0.5)
# axes.grid(zorder=0)
# axes.set_xlabel('Distance in x-direction [km]')
# axes.set_ylabel('Distance in y-direction [km]')
# fig.tight_layout()
# file = base_path_figs / "river_reach.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

file = base_path / "input" / "hydraulic_conductivity_layer1.grd"
hydraulic_conductivities_layer1 = Raster.load(file).get_array(1) / (60 * 60 * 24)
file = base_path / "input" / "hydraulic_conductivity_layer2.grd"
hydraulic_conductivities_layer2 = Raster.load(file).get_array(1) / (60 * 60 * 24)
file = base_path / "input" / "hydraulic_conductivity_layer3.grd"
hydraulic_conductivities_layer3 = Raster.load(file).get_array(1) / (60 * 60 * 24)
file = base_path / "input" / "hydraulic_conductivity_layer4.grd"
hydraulic_conductivities_layer4 = Raster.load(file).get_array(1) / (60 * 60 * 24)
hydraulic_conductivities_layers = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]
for i, hydraulic_conductivities_layer in enumerate(hydraulic_conductivities_layers):
    i = i + 1
    fig, axes = plt.subplots(figsize=(4, 4))
    hydraulic_conductivities_layer[~mask] = np.nan
    plt.imshow(hydraulic_conductivities_layer, extent=grid_extent, cmap='Oranges', aspect='equal')
    plt.colorbar(label='$k_f$ [m/s]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.tight_layout()
    file = base_path_figs / f"hydraulic_conductivity_layer_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

file = base_path / "input" / "specific_yield_layer1.grd"
specific_yield_layer1 = Raster.load(file).get_array(1)
file = base_path / "input" / "specific_yield_layer2.grd"
specific_yield_layer2 = Raster.load(file).get_array(1)
file = base_path / "input" / "specific_yield_layer3.grd"
specific_yield_layer3 = Raster.load(file).get_array(1)
file = base_path / "input" / "specific_yield_layer4.grd"
specific_yield_layer4 = Raster.load(file).get_array(1)
specific_yield_layers = [specific_yield_layer1, specific_yield_layer2, specific_yield_layer3, specific_yield_layer4]
for i, specific_yield_layer in enumerate(specific_yield_layers):
    i = i + 1
    fig, axes = plt.subplots(figsize=(4, 4))
    specific_yield_layer[~mask] = np.nan
    plt.imshow(specific_yield_layer, extent=grid_extent, cmap='Oranges', aspect='equal')
    plt.colorbar(label='$n$ [-]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.tight_layout()
    file = base_path_figs / f"specific_yield_layer_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
elevation_bottom_layer1[~mask] = np.nan
thickness_layer1 = topography - elevation_bottom_layer1
plt.imshow(thickness_layer1, extent=grid_extent, cmap='viridis', aspect='equal')
plt.colorbar(label='[m]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [km]')
plt.ylabel('Distance in y-direction [km]')
plt.tight_layout()
file = base_path_figs / f"thickness_layer1.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
elevation_bottom_layer1[~mask] = np.nan
elevation_bottom_layer2[~mask] = np.nan
thickness_layer2 = elevation_bottom_layer1 - elevation_bottom_layer2
plt.imshow(thickness_layer2, extent=grid_extent, cmap='viridis', aspect='equal')
plt.colorbar(label='[m]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [km]')
plt.ylabel('Distance in y-direction [km]')
plt.tight_layout()
file = base_path_figs / f"thickness_layer2.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
elevation_bottom_layer2[~mask] = np.nan
elevation_bottom_layer3[~mask] = np.nan
thickness_layer3 = elevation_bottom_layer2 - elevation_bottom_layer3
plt.imshow(thickness_layer3, extent=grid_extent, cmap='viridis', aspect='equal')
plt.colorbar(label='[m]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [km]')
plt.ylabel('Distance in y-direction [km]')
plt.tight_layout()
file = base_path_figs / f"thickness_layer3.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
elevation_bottom_layer3[~mask] = np.nan
elevation_bottom_layer4[~mask] = np.nan
thickness_layer4 = elevation_bottom_layer3 - elevation_bottom_layer4
plt.imshow(thickness_layer4, extent=grid_extent, cmap='viridis', aspect='equal')
plt.colorbar(label='[m]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [km]')
plt.ylabel('Distance in y-direction [km]')
plt.tight_layout()
file = base_path_figs / f"thickness_layer4.png"
fig.savefig(file, dpi=300)
plt.close(fig)

thickness_layers = [thickness_layer1, thickness_layer2, thickness_layer3, thickness_layer4]
fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=False)
x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
for layer in range(4):
    z = thickness_layers[layer][82, :-1]
    axes[layer].plot(x, z, ls='-', lw=1, color='black')
    axes[layer].set_xlim(0, x[-1])
    axes[layer].set_ylim(0, )
axes[-1].set_xlabel('Distance in x-direction [km]')
axes[0].set_ylabel('layer 1 \n aquifer thickness \n[m]')
axes[1].set_ylabel('layer 2 \n aquifer thickness \n[m]')
axes[2].set_ylabel('layer 3 \n aquifer thickness \n[m]')
axes[3].set_ylabel('layer 4 \n aquifer thickness \n[m]')
fig.tight_layout()
file = base_path_figs / "gw_thickness_x_cross_section_layer.png"
fig.savefig(file, dpi=300)
plt.close("all")

fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=False)
x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
axes.plot(x, topography[82, :-1], ls='-', lw=2, color='black')
axes.plot(x, elevation_bottom_layer1[82, :-1], ls='--', lw=1.75, color='black', alpha=0.9)
axes.plot(x, elevation_bottom_layer2[82, :-1], ls='-.', lw=1.5, color='black', alpha=0.8)
axes.plot(x, elevation_bottom_layer3[82, :-1], ls=':', lw=1.25, color='black', alpha=0.7)
axes.plot(x, elevation_bottom_layer4[82, :-1], ls='-', lw=1, color='black', alpha=0.6)
axes.set_xlim(0, x[-1])
# axes.set_ylim(0, )
axes.set_xlabel('Distance in x-direction [km]')
axes.set_ylabel('elevation \n[m a.s.l.]')
fig.tight_layout()
file = base_path_figs / "gw_layers_cross_section_west-east_layer.png"
fig.savefig(file, dpi=300)
plt.close("all")


transmissivity_layer1 = hydraulic_conductivities_layer1 * thickness_layer1
transmissivity_layer2 = hydraulic_conductivities_layer2 * thickness_layer2
transmissivity_layer3 = hydraulic_conductivities_layer3 * thickness_layer3
transmissivity_layer4 = hydraulic_conductivities_layer4 * thickness_layer4
transmissivity_layers = [transmissivity_layer1, transmissivity_layer2, transmissivity_layer3, transmissivity_layer4]
for i, transmissivity_layer in enumerate(transmissivity_layers):
    i = i + 1
    fig, axes = plt.subplots(figsize=(4, 4))
    transmissivity_layer[~mask] = np.nan
    plt.imshow(transmissivity_layer, extent=grid_extent, cmap='Oranges', aspect='equal')
    plt.colorbar(label='$T$ [$m^2$/s]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.tight_layout()
    file = base_path_figs / f"transmissivity_layer_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


params_file = base_path / "parameters.nc"
ds_params = xr.open_dataset(params_file, engine="h5netcdf")
data1 = ds_params['TP'].values / (1000 * 60 * 60)

fig, axes = plt.subplots(2, 1, figsize=(4, 6))
ax1 = axes[0].imshow(data1, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.nanmax(data1)))
fig.colorbar(ax1, shrink=0.45, label='[m/s]')
axes[0].set_xlabel('Distance in y-direction [km]')
axes[0].set_ylabel('Distance in x-direction [km]')
ax2 = axes[1].imshow(hydraulic_conductivities_layer1, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.nanmax(data1)))
axes[1].set_xlabel('Distance in y-direction [km]')
axes[1].set_ylabel('Distance in x-direction [km]')
fig.colorbar(ax2, shrink=0.45, label='[m/s]')
fig.tight_layout()
file = base_path_figs / "kf_comparison_layer1.png"
fig.savefig(file, dpi=300)
plt.close("all")

fig, axes = plt.subplots(2, 1, figsize=(4, 6))
ax1 = axes[0].imshow(data1, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.nanmax(data1)))
fig.colorbar(ax1, shrink=0.45, label='[m/s]')
axes[0].set_xlabel('Distance in y-direction [km]')
axes[0].set_ylabel('Distance in x-direction [km]')
ax2 = axes[1].imshow(hydraulic_conductivities_layer2, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.nanmax(data1)))
axes[1].set_xlabel('Distance in y-direction [km]')
axes[1].set_ylabel('Distance in x-direction [km]')
fig.colorbar(ax2, shrink=0.45, label='[m/s]')
fig.tight_layout()
file = base_path_figs / "kf_comparison_layer2.png"
fig.savefig(file, dpi=300)
plt.close("all")

fig, axes = plt.subplots(figsize=(4, 4))
data = hydraulic_conductivities_layer2 * (1000 * 60 * 60)
data[~mask] = np.nan
plt.imshow(data, extent=grid_extent, cmap='Oranges', aspect='equal')
plt.colorbar(label='$k_f$ [mm/hour]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [km]')
plt.ylabel('Distance in y-direction [km]')
plt.tight_layout()
file = base_path_figs / f"hydraulic_conductivity_layer_2_mmh.png"
fig.savefig(file, dpi=300)
plt.close(fig)