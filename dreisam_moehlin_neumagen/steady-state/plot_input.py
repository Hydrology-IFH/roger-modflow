import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
import yaml

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join("..", ".."))
    sys.path.append(fpth)
    import flopy

def fudge_hydraulic_conductivities(ds_params, fudge_parameters, model_run=5):
    hydraulic_conductivities_layer1 = ds_params['kf'].isel(layer=0).values
    hydraulic_conductivities_layer2 = ds_params['kf'].isel(layer=1).values
    hydraulic_conductivities_layer3 = ds_params['kf'].isel(layer=2).values
    hydraulic_conductivities_layer4 = ds_params['kf'].isel(layer=3).values

    mask_upper_dreisam = (ds_params['mask_upper_dreisam'].values == 1)
    mask_upper_moehlin = (ds_params['mask_upper_moehlin'].values == 1)
    mask_neumagen = (ds_params['mask_neumagen'].values == 1)
    mask_zarten_brugga = (ds_params['mask_zarten_brugga'].values == 1)
    mask_zarten_gravel_north = (ds_params['mask_zarten_gravel_north'].values == 1)
    mask_staufen_gravel = (ds_params['mask_staufen_gravel'].values == 1)
    mask_gravel = mask_zarten_brugga | mask_zarten_gravel_north | mask_staufen_gravel
    mask_valleys_black_forest = mask_upper_dreisam | mask_upper_moehlin | mask_neumagen
    
    # fudge parameters
    mask1 = (hydraulic_conductivities_layer1 <= 1e-5)
    mask2 = (hydraulic_conductivities_layer2 <= 1e-5)
    mask3 = (hydraulic_conductivities_layer3 <= 1e-5)
    mask4 = (hydraulic_conductivities_layer4 <= 1e-5)  
    hydraulic_conductivities_layer1[mask1] = hydraulic_conductivities_layer1[mask1] * 2000
    hydraulic_conductivities_layer2[mask2] = hydraulic_conductivities_layer2[mask2] * 2000
    hydraulic_conductivities_layer3[mask3] = hydraulic_conductivities_layer3[mask3] * 2000
    hydraulic_conductivities_layer4[mask4] = hydraulic_conductivities_layer4[mask4] * 2000

    mask2 = (topography < 450) & (hydraulic_conductivities_layer2 > 10) & (mask_gravel == False)
    mask3 = (topography < 450) & (hydraulic_conductivities_layer3 > 10)
    mask4 = (topography < 450) & (hydraulic_conductivities_layer4 > 10)

    mask5 = (topography < 450) & (hydraulic_conductivities_layer2 > 1) & (hydraulic_conductivities_layer2 <= 10)
    mask6 = (topography < 450) & (hydraulic_conductivities_layer3 > 1) & (hydraulic_conductivities_layer3 <= 10)
    mask7 = (topography < 450) & (hydraulic_conductivities_layer4 > 1) & (hydraulic_conductivities_layer4 <= 10)

    mask8 = (topography < 450) & (hydraulic_conductivities_layer2 > 0.01) & (hydraulic_conductivities_layer2 <= 1)
    mask9 = (topography < 450) & (hydraulic_conductivities_layer3 > 0.01) & (hydraulic_conductivities_layer3 <= 1)
    mask10 = (topography < 450) & (hydraulic_conductivities_layer4 > 0.01) & (hydraulic_conductivities_layer4 <= 1)

    mask11 = (topography >= 450) & (hydraulic_conductivities_layer2 > 1) & (hydraulic_conductivities_layer2 <= 10)
    mask12 = (topography >= 450) & (hydraulic_conductivities_layer3 > 1) & (hydraulic_conductivities_layer3 <= 10)
    mask13 = (topography >= 450) & (hydraulic_conductivities_layer4 > 1) & (hydraulic_conductivities_layer4 <= 10)

    mask14 = (topography >= 450) & (hydraulic_conductivities_layer2 > 0.01) & (hydraulic_conductivities_layer2 <= 1)
    mask15 = (topography >= 450) & (hydraulic_conductivities_layer3 > 0.01) & (hydraulic_conductivities_layer3 <= 1)
    mask16 = (topography >= 450) & (hydraulic_conductivities_layer4 > 0.01) & (hydraulic_conductivities_layer4 <= 1) 

    # fudge parameters of the upper layer
    hydraulic_conductivities_layer1[topography < 450] = hydraulic_conductivities_layer1[topography < 450] * fudge_parameters['l_1'].values[model_run]

    # fudge parameters in the Rhine aquifer
    hydraulic_conductivities_layer2[mask2] = hydraulic_conductivities_layer2[mask2] * fudge_parameters['rz10_23'].values[model_run]
    hydraulic_conductivities_layer3[mask3] = hydraulic_conductivities_layer3[mask3] * fudge_parameters['rz10_23'].values[model_run]
    hydraulic_conductivities_layer4[mask4] = hydraulic_conductivities_layer4[mask4] * fudge_parameters['r10_4'].values[model_run]

    hydraulic_conductivities_layer2[mask5] = hydraulic_conductivities_layer2[mask5] * fudge_parameters['r110_23'].values[model_run]
    hydraulic_conductivities_layer3[mask6] = hydraulic_conductivities_layer3[mask6] * fudge_parameters['r110_23'].values[model_run]
    hydraulic_conductivities_layer4[mask7] = hydraulic_conductivities_layer4[mask7] * fudge_parameters['r110_4'].values[model_run]

    hydraulic_conductivities_layer2[mask8] = hydraulic_conductivities_layer2[mask8] * fudge_parameters['r0011_23'].values[model_run]
    hydraulic_conductivities_layer3[mask9] = hydraulic_conductivities_layer3[mask9] * fudge_parameters['r0011_23'].values[model_run]
    hydraulic_conductivities_layer4[mask10] = hydraulic_conductivities_layer4[mask10] * fudge_parameters['r0011_4'].values[model_run]

    # fudge parameters in the Zarten aquifer
    hydraulic_conductivities_layer2[mask2 & mask_valleys_black_forest] = np.where(mask2 & mask_valleys_black_forest, hydraulic_conductivities_layer2 * fudge_parameters['rz10_23'].values[model_run], hydraulic_conductivities_layer2)[(mask2 & mask_valleys_black_forest)]
    hydraulic_conductivities_layer3[mask3 & mask_valleys_black_forest] = np.where(mask3 & mask_valleys_black_forest, hydraulic_conductivities_layer3 * fudge_parameters['rz10_23'].values[model_run], hydraulic_conductivities_layer3)[(mask3 & mask_valleys_black_forest)]  
    hydraulic_conductivities_layer4[mask4 & mask_valleys_black_forest] = np.where(mask4 & mask_valleys_black_forest, hydraulic_conductivities_layer4 * fudge_parameters['rz10_23'].values[model_run], hydraulic_conductivities_layer4)[(mask4 & mask_valleys_black_forest)]    

    hydraulic_conductivities_layer2[mask8 & mask_valleys_black_forest] = np.where(mask8 & mask_valleys_black_forest, hydraulic_conductivities_layer2 * fudge_parameters['z0011'].values[model_run], hydraulic_conductivities_layer2)[(mask8 & mask_valleys_black_forest)]
    hydraulic_conductivities_layer3[mask9 & mask_valleys_black_forest] = np.where(mask9 & mask_valleys_black_forest, hydraulic_conductivities_layer3 * fudge_parameters['z0011'].values[model_run], hydraulic_conductivities_layer3)[(mask9 & mask_valleys_black_forest)]  
    hydraulic_conductivities_layer4[mask10 & mask_valleys_black_forest] = np.where(mask10 & mask_valleys_black_forest, hydraulic_conductivities_layer4 * fudge_parameters['z0011'].values[model_run], hydraulic_conductivities_layer4)[(mask10 & mask_valleys_black_forest)]  

    # fudge parameters in the mountain
    hydraulic_conductivities_layer2[mask11] = hydraulic_conductivities_layer2[mask11] * fudge_parameters['m110'].values[model_run]
    hydraulic_conductivities_layer3[mask12] = hydraulic_conductivities_layer3[mask12] * fudge_parameters['m110'].values[model_run] * 0.66
    hydraulic_conductivities_layer4[mask13] = hydraulic_conductivities_layer4[mask13] * fudge_parameters['m110'].values[model_run] * 0.33

    hydraulic_conductivities_layer2[mask14] = hydraulic_conductivities_layer2[mask14] * fudge_parameters['m0011'].values[model_run]
    hydraulic_conductivities_layer3[mask15] = hydraulic_conductivities_layer3[mask15] * fudge_parameters['m0011'].values[model_run] * 0.66
    hydraulic_conductivities_layer4[mask16] = hydraulic_conductivities_layer4[mask16] * fudge_parameters['m0011'].values[model_run] * 0.33
    hydraulic_conductivities_layer = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]

    return hydraulic_conductivities_layer


base_path = Path(__file__).parent
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)


res_modflow = 50  # spatial resolution of MODFLOW in meters
modflow_config = {
    'dx': 50,
    'dy': 50,
    'nx': 621,
    'ny': 777,
    'nz': 4,
}

# load observed groundwater heads (average values of the observation wells)
path = base_path / "observations" / "observed_groundwater_heads_avg.csv"
observed_groundwater_heads = pd.read_csv(path, sep=";", skiprows=0)

# load fudge parameters
path = base_path / "fudge_parameters_modflow.csv"
fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)

# load MODFLOW parameters
path = Path(__file__).parent / "parameters_modflow.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")

path = Path(__file__).parent / "boundary_conditions.nc"
ds_bc = xr.open_dataset(path, engine="h5netcdf")

# load the topography and elevation of the aquifer layers
topography = ds_params['elevations'].isel(z=0).values
elevation_bottom_layer1 = ds_params['elevations'].isel(z=1).values
elevation_bottom_layer2 = ds_params['elevations'].isel(z=2).values
elevation_bottom_layer3 = ds_params['elevations'].isel(z=3).values
elevation_bottom_layer4 = ds_params['elevations'].isel(z=4).values
elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

# derive the model domain from the topography
mask = np.isfinite(topography)
mask_upper_dreisam = (ds_params['mask_upper_dreisam'].values == 1)
mask_upper_moehlin = (ds_params['mask_upper_moehlin'].values == 1)
mask_neumagen = (ds_params['mask_neumagen'].values == 1)
mask_valleys_black_forest = mask_upper_dreisam | mask_upper_moehlin | mask_neumagen
mask_zarten_brugga = (ds_params['mask_zarten_brugga'].values == 1)
mask_zarten_gravel_north = (ds_params['mask_zarten_gravel_north'].values == 1)
mask_staufen_gravel = (ds_params['mask_staufen_gravel'].values == 1)
mask_gravel = mask_zarten_brugga | mask_zarten_gravel_north | mask_staufen_gravel
grid_extent = (0, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])

basin = np.empty_like(topography)
basin[mask] = 1
fig, axes = plt.subplots(figsize=(4, 4))
plt.imshow(basin, extent=grid_extent, cmap='Greys', aspect='equal')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "basin.png"
fig.savefig(file, dpi=300)
plt.close(fig)

grid_extent = (0, modflow_config['ny']*modflow_config['dy'], modflow_config['nx']*modflow_config['dx'], 0)
fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
plt.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal')
plt.colorbar(label='[m a.s.l.]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "topography.png"
fig.savefig(file, dpi=300)
plt.close(fig)

grid_extent = (0, modflow_config['ny']*modflow_config['dy'], modflow_config['nx']*modflow_config['dx'], 0)
fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
plt.imshow(np.where(topography < 300, np.nan, topography), extent=grid_extent, cmap='terrain', aspect='equal')
plt.colorbar(label='[m a.s.l.]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "topography_mountains.png"
fig.savefig(file, dpi=300)
plt.close(fig)

grid_extent = (0, modflow_config['ny']*modflow_config['dy'], modflow_config['nx']*modflow_config['dx'], 0)
fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
wells_y = [266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]
wells_x = [66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]
plt.scatter(wells_x, wells_y, marker='x', s=5, c='black')
plt.imshow(topography, cmap='terrain', aspect='equal')
plt.colorbar(label='[m a.s.l.]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('x-direction')
plt.ylabel('y-direction')
plt.tight_layout()
file = base_path_figs / "topography_and_wells.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]) * 50
wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]) * 50
plt.scatter(wells_x, wells_y, marker='^', s=5, c='black')
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=5, c='purple')
plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
plt.colorbar(label='[m a.s.l.]', shrink=0.45)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "topography_and_wells_observations.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264])
wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496])
plt.scatter(wells_x, wells_y, marker='^', s=5, c='black')
wells_obs_y = observed_groundwater_heads["Zelle_y"].values # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=5, c='purple')
plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5)
plt.colorbar(label='[m a.s.l.]', shrink=0.45)
plt.grid(zorder=0)
plt.xlabel('x-direction')
plt.ylabel('y-direction')
plt.tight_layout()
file = base_path_figs / "topography_and_wells_observations_.png"
fig.savefig(file, dpi=300)
plt.close(fig)


grid_extent = (0, modflow_config['ny']*modflow_config['dy'], modflow_config['nx']*modflow_config['dx'], 0)
fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
topography_schoenberg = topography.copy()
src = rasterio.open(str(base_path / "input" / "schoenberg.tif"))
schoenberg_mask = src.read(1)
topography_schoenberg = np.where(schoenberg_mask == 1, np.nan, topography_schoenberg)
plt.imshow(topography_schoenberg, cmap='terrain', aspect='equal')
plt.colorbar(label='[m a.s.l.]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('x-direction')
plt.ylabel('y-direction')
plt.tight_layout()
file = base_path_figs / "topography_schoenberg.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# fig, axes = plt.subplots(figsize=(4, 4))
# topography[~mask] = np.nan
# cb = axes.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal')
# plt.scatter(10*50, (178-82)*50, marker='x', s=20, c='black')  # location of the extraction well 1
# plt.scatter(40*50, (178-92)*50, marker='x', s=20, c='black')  # location of the extraction well 2
# plt.scatter(85*50, (178-62)*50, marker='x', s=20, c='black')  # location of the extraction well 3
# fig.colorbar(cb , ax=axes, label='[m a.s.l.]', shrink=0.5)
# axes.grid(zorder=0)
# axes.set_xlabel('Distance in x-direction [m]')
# axes.set_ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / "topography_well_locations.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# fig, axes = plt.subplots(figsize=(5, 6))
# grid_extent1 = (-2000, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])
# topography[~mask] = np.nan
# empty_arr = np.zeros((modflow_config['nx'], int(2000/modflow_config['dy'])))
# empty_arr[:, :] = np.nan
# topography1 = np.concatenate((empty_arr, topography), axis=1)
# plt.scatter(-1350, 96*50, marker='x', s=20, c='black')  # approximate location of the observation well
# plt.imshow(topography1, extent=grid_extent1, cmap='terrain', aspect='equal')
# plt.colorbar(label='[m a.s.l.]', shrink=0.34)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# fig.tight_layout()
# file = base_path_figs / "topography1.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)


for i, elevation_bottom_layer in enumerate(elevation_bottom_layers):
    i = i + 1
    fig, axes = plt.subplots(figsize=(4, 4))
    elevation_bottom_layer[~mask] = np.nan
    plt.imshow(elevation_bottom_layer, extent=grid_extent, cmap='viridis', vmin=0, vmax=1200, aspect='equal')
    plt.colorbar(label='[m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"elevation_bottom_layer_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
elevation_bottom_layer[~mask] = np.nan
plt.imshow(elevation_bottom_layer, extent=grid_extent, cmap='viridis', vmin=0, vmax=250, aspect='equal')
plt.colorbar(label='[m a.s.l.]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"elevation_bottom_layer_{i}.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# initial_conditions_layer1 = (topography - elevation_bottom_layer1) * 0.5 + elevation_bottom_layer1
# initial_conditions_layer2 = (elevation_bottom_layer1 - elevation_bottom_layer2) * 0.5 + elevation_bottom_layer2
# initial_conditions_layer3 = (elevation_bottom_layer2 - elevation_bottom_layer3) * 0.5 + elevation_bottom_layer3
# initial_conditions_layer4 = (elevation_bottom_layer3 - elevation_bottom_layer4) * 0.5 + elevation_bottom_layer4
# initial_conditions_layers = [initial_conditions_layer1, initial_conditions_layer2, initial_conditions_layer3, initial_conditions_layer4]
# for i, initial_conditions_layer in enumerate(initial_conditions_layers):
#     fig, axes = plt.subplots(figsize=(4, 4))
#     plt.imshow(initial_conditions_layer, extent=grid_extent, cmap='viridis', aspect='equal')
#     plt.colorbar(label='groundwater head [m a.s.l.]', shrink=0.5)
#     plt.grid(zorder=0)
#     plt.xlabel('Distance in x-direction [m]')
#     plt.ylabel('Distance in y-direction [m]')
#     plt.tight_layout()
#     file = base_path_figs / f"initial_conditions_{i}.png"
#     fig.savefig(file, dpi=300)
#     plt.close(fig)

# file = base_path / "input" / "boundary_condition.grd"
# boundary_condition = Raster.load(file).get_array(1)
# mask1 = (boundary_condition == -1)
# boundary_condition1 = np.zeros((modflow_config['nx'], modflow_config['ny']+1))
# boundary_condition1[:, :] = np.nan
# boundary_condition1[mask] = 0.2
# boundary_condition1[mask1] = 1
# fig, axes = plt.subplots(figsize=(4, 4))
# plt.imshow(boundary_condition1, extent=grid_extent, cmap='Greys', vmin=0, vmax=1, aspect='equal')
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / "boundary_condition.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# file = base_path / "input" / "river_reach.csv"
# river_reach = pd.read_csv(file, delimiter=",")
# river_reach_arr = np.zeros((modflow_config['nx'], modflow_config['ny']))
# river_reach_arr[:, :] = np.nan
# for x, y in zip(np.max(river_reach['i'].values) - river_reach['i'], river_reach['j']):
#     river_reach_arr[x, y] = 1
# river_reach_arr = np.where(mask, river_reach_arr, np.nan)
# fig, axes = plt.subplots(figsize=(4, 4))
# # cb = axes.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal')
# # fig.colorbar(cb, ax=axes, label='[m a.s.l.]', shrink=0.5)
# axes.imshow(river_reach_arr, extent=grid_extent, cmap='Blues', vmin=0, vmax=1, aspect='equal')
# axes.grid(zorder=0)
# axes.set_xlabel('Distance in x-direction [m]')
# axes.set_ylabel('Distance in y-direction [m]')
# fig.tight_layout()
# file = base_path_figs / "river_reach.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# fig, axes = plt.subplots(figsize=(4, 4))
# plt.imshow(boundary_condition1, extent=grid_extent, cmap='Greys', vmin=0, vmax=1, aspect='equal')
# axes.imshow(river_reach_arr, extent=grid_extent, cmap='Blues', vmin=0, vmax=1.5, aspect='equal')
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / "boundary_condition_and_river_reach.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

file = base_path / "input" / "river_reach.csv"
river_reach = pd.read_csv(file, delimiter=",")
fig, axes = plt.subplots(figsize=(4, 4))
axes.scatter(river_reach['j'].values*modflow_config['dx'], (np.max(river_reach['i'].values) - river_reach['i'].values)*modflow_config['dy'], s=0.5, c='blue')
cb = axes.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal')
fig.colorbar(cb, ax=axes, label='[m a.s.l.]', shrink=0.5)
axes.grid(zorder=0)
axes.set_xlabel('Distance in x-direction [m]')
axes.set_ylabel('Distance in y-direction [m]')
fig.tight_layout()
file = base_path_figs / "river_reach_.png"
fig.savefig(file, dpi=300)
plt.close(fig)

_x = []
_y = []
for index in range(0, len(river_reach.index)):
    i = river_reach['i'].values[index]
    j = river_reach['j'].values[index]
    if not mask[i,j]:
        _x.append(j)
        _y.append(i)


file = base_path / "input" / "river_reach.csv"
river_reach = pd.read_csv(file, delimiter=",")
fig, axes = plt.subplots(figsize=(4, 4))
axes.scatter(river_reach['j'].values, river_reach['i'].values, s=0.5, c='blue')
axes.scatter(_x, _y, s=0.5, c='red')
cb = axes.imshow(topography, cmap='terrain', aspect='equal')
fig.colorbar(cb, ax=axes, label='[m a.s.l.]', shrink=0.5)
axes.grid(zorder=0)
axes.set_xlabel('x-direction')
axes.set_ylabel('y-direction')
fig.tight_layout()
file = base_path_figs / "river_reach_.png"
fig.savefig(file, dpi=300)
plt.close(fig)

hydraulic_conductivities_layer1 = ds_params['kf'].isel(layer=0).values
hydraulic_conductivities_layer2 = ds_params['kf'].isel(layer=1).values
hydraulic_conductivities_layer3 = ds_params['kf'].isel(layer=2).values
hydraulic_conductivities_layer4 = ds_params['kf'].isel(layer=3).values
hydraulic_conductivities_layers = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]
for i, hydraulic_conductivities_layer in enumerate(hydraulic_conductivities_layers):
    i = i + 1
    print(f"Kf Layer {i}")
    print(np.unique(hydraulic_conductivities_layer[mask]/(24*60*60)))

for i, hydraulic_conductivities_layer in enumerate(hydraulic_conductivities_layers):
    i = i + 1
    fig, axes = plt.subplots(figsize=(4, 4))
    bounds = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001]
    norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Oranges"].N)
    hydraulic_conductivities_layer[~mask] = np.nan
    plt.imshow(hydraulic_conductivities_layer/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal', norm=norm)
    cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.45)
    cbar.set_ticks(ticks=bounds, labels=[r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$'])
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"hydraulic_conductivity_layer_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)
    print(f"Kf Layer {i} (Min., Max.): ", np.nanmin(hydraulic_conductivities_layer), np.nanmax(hydraulic_conductivities_layer))
    hydraulic_conductivities_layer[~mask] = 0
    print(f"Kf Layer {i} (Number of no data): ", np.isnan(hydraulic_conductivities_layer).sum())
    # print(np.unique(hydraulic_conductivities_layer))
    fig, axes = plt.subplots(figsize=(4, 4))
    hydraulic_conductivities_layer[~mask] = np.nan
    plt.imshow(hydraulic_conductivities_layer/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal')
    cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.45)
    # cbar.set_ticks(ticks=bounds, labels=[r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"hydraulic_conductivity_layer_{i}_.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

hydraulic_conductivities_layers_fudged = fudge_hydraulic_conductivities(ds_params, fudge_parameters, model_run=0)
for i, hydraulic_conductivities_layer in enumerate(hydraulic_conductivities_layers_fudged):
    i = i + 1
    fig, axes = plt.subplots(figsize=(4, 4))
    bounds = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Oranges"].N)
    hydraulic_conductivities_layer[~mask] = np.nan
    plt.imshow(hydraulic_conductivities_layer/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal', norm=norm)
    cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.45)
    cbar.set_ticks(ticks=bounds, labels=[r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"hydraulic_conductivity_layer_fudged_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)
    print(f"Kf Layer {i} (Min., Max.): ", np.nanmin(hydraulic_conductivities_layer), np.nanmax(hydraulic_conductivities_layer))
    hydraulic_conductivities_layer[~mask] = 0
    print(f"Kf Layer {i} (Number of no data): ", np.isnan(hydraulic_conductivities_layer).sum())
    print(np.unique(hydraulic_conductivities_layer))
    fig, axes = plt.subplots(figsize=(4, 4))
    hydraulic_conductivities_layer[~mask] = np.nan
    plt.imshow(hydraulic_conductivities_layer/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal', vmin=0.00000001, vmax=0.1)
    cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.45)
    cbar.set_ticks(ticks=bounds, labels=[r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"hydraulic_conductivity_layer_fudged_{i}_.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


# i = 1
# fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
# i = i + 1
# fig, axes = plt.subplots(figsize=(4, 4))
# fudge_regions_layer[~mask] = np.nan
# mask1 = (topography < 450) & (fudge_regions_layer > 10**(-4)) & (mask_gravel == False)
# mask12 = (topography < 450) & (fudge_regions_layer > 10**(-4)) & (mask_gravel == True)
# mask2 = (topography < 450) & (fudge_regions_layer > 10**(-4)) & (fudge_regions_layer <= 10**(-4))
# mask3 = (topography < 450) & (fudge_regions_layer <= 10**(-5))
# mask4 = (topography >= 450) & (fudge_regions_layer > 10**(-4))
# mask5 = (topography >= 450) & (fudge_regions_layer > 10**(-5)) & (fudge_regions_layer <= 10**(-4))
# mask6 = (topography >= 450) & (fudge_regions_layer <= 10**(-5))
# fudge_regions_layer = np.where(mask1, 100, fudge_regions_layer)
# fudge_regions_layer = np.where(mask2, 200, fudge_regions_layer)
# fudge_regions_layer = np.where(mask3, 300, fudge_regions_layer)
# fudge_regions_layer = np.where(mask12, 400, fudge_regions_layer)
# fudge_regions_layer[mask1 & mask_valleys_black_forest] = np.where(mask1 & mask_valleys_black_forest, 500, fudge_regions_layer)[mask1 & mask_valleys_black_forest]
# fudge_regions_layer[mask3 & mask_valleys_black_forest] = np.where(mask3 & mask_valleys_black_forest, 600, fudge_regions_layer)[mask3 & mask_valleys_black_forest]
# fudge_regions_layer = np.where(mask4, 700, fudge_regions_layer)
# fudge_regions_layer = np.where(mask5, 800, fudge_regions_layer)
# fudge_regions_layer = np.where(mask6, 900, fudge_regions_layer)
# fudge_regions_layer = np.where(np.isin(fudge_regions_layer, [100, 200, 300, 400, 500, 600, 700, 800, 900]), fudge_regions_layer, np.nan)
# plt.imshow(fudge_regions_layer, cmap='Oranges', aspect='equal', extent=grid_extent)
# cbar = plt.colorbar(label='', shrink=0.45)
# wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
# wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
# plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / f"fudge_regions_layer{i}.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)


i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [0, 100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Dark2"].N)
fudge_regions_layer[~mask] = np.nan
mask1 = (topography < 450) & (fudge_regions_layer > 10**(-4)) & (fudge_regions_layer <= 10**(-1)) & (mask_gravel == False) & (mask_valleys_black_forest == False)
fudge_regions_layer = np.where(mask1, 99., fudge_regions_layer)
fudge_regions_layer_r4 = np.where(np.isin(fudge_regions_layer, [99.]), fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_layer_r4, cmap='Dark2', aspect='equal', norm=norm, extent=grid_extent)
cbar = plt.colorbar(label='', shrink=0.45)
cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-5', 'm-4', 'm-6'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_r4.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [0, 100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Dark2"].N)
fudge_regions_layer[~mask] = np.nan
mask2 = (topography < 450) & (fudge_regions_layer > 10**(-5)) & (fudge_regions_layer <= 10**(-4)) & (mask_gravel == False) & (mask_valleys_black_forest == False)
fudge_regions_layer = np.where(mask2, 199., fudge_regions_layer)
fudge_regions_layer_r5 = np.where(np.isin(fudge_regions_layer, [199.]), fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_layer_r5, cmap='Dark2', aspect='equal', norm=norm, extent=grid_extent)
cbar = plt.colorbar(label='', shrink=0.45)
cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-5', 'm-4', 'm-6'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_r5.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [0, 100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Dark2"].N)
fudge_regions_layer[~mask] = np.nan
mask3 = (topography < 450) & (fudge_regions_layer <= 10**(-5)) & (mask_gravel == False) & (mask_valleys_black_forest == False)
fudge_regions_layer = np.where(mask3, 299., fudge_regions_layer)
fudge_regions_layer_r6 = np.where(np.isin(fudge_regions_layer, [299.]), fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_layer_r6, cmap='Dark2', aspect='equal', norm=norm, extent=grid_extent)
cbar = plt.colorbar(label='', shrink=0.45)
cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-5', 'm-4', 'm-6'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_r6.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [0, 100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Dark2"].N)
fudge_regions_layer[~mask] = np.nan
mask12 = (topography < 450) & (mask_gravel == True)
fudge_regions_layer = np.where(mask12, 399., fudge_regions_layer)
fudge_regions_layer_g4 = np.where(np.isin(fudge_regions_layer, [399.]), fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_layer_g4, cmap='Dark2', aspect='equal', norm=norm, extent=grid_extent)
cbar = plt.colorbar(label='', shrink=0.45)
cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-5', 'm-4', 'm-6'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_g4.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [0, 100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Dark2"].N)
fudge_regions_layer[~mask] = np.nan
mask1 = (topography < 450) & (fudge_regions_layer > 10**(-4)) & (fudge_regions_layer <= 10**(-1)) & (mask_gravel == False)
fudge_regions_layer[mask1 & mask_valleys_black_forest] = np.where(mask1 & mask_valleys_black_forest, 499., fudge_regions_layer)[mask1 & mask_valleys_black_forest]
fudge_regions_layer_z4 = np.where(np.isin(fudge_regions_layer, [499.]), fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_layer_z4, cmap='Dark2', aspect='equal', norm=norm, extent=grid_extent)
cbar = plt.colorbar(label='', shrink=0.45)
cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-5', 'm-4', 'm-6'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_z4.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [0, 100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Dark2"].N)
fudge_regions_layer[~mask] = np.nan
mask2 = (topography < 450) & (fudge_regions_layer > 10**(-7)) & (fudge_regions_layer <= 10**(-4)) & (mask_gravel == False)
fudge_regions_layer[mask2 & mask_valleys_black_forest] = np.where(mask2 & mask_valleys_black_forest, 599., fudge_regions_layer)[mask2 & mask_valleys_black_forest]
fudge_regions_layer_z5 = np.where(np.isin(fudge_regions_layer, [599.]), fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_layer_z5, cmap='Dark2', aspect='equal', norm=norm, extent=grid_extent)
cbar = plt.colorbar(label='', shrink=0.45)
cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-5', 'm-4', 'm-6'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_z5.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [0, 100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Dark2"].N)
fudge_regions_layer[~mask] = np.nan
mask4 = (topography >= 450) & (fudge_regions_layer > 10**(-5)) & (fudge_regions_layer <= 10**(-1))
fudge_regions_layer = np.where(mask4, 699., fudge_regions_layer)
fudge_regions_layer_m4 = np.where(np.isin(fudge_regions_layer, [699.]), fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_layer_m4, cmap='Dark2', aspect='equal', norm=norm, extent=grid_extent)
cbar = plt.colorbar(label='', shrink=0.45)
cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-5', 'm-4', 'm-6'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_m4.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [0, 100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Dark2"].N)
fudge_regions_layer[~mask] = np.nan
mask6 = (topography > 450) & (fudge_regions_layer <= 10**(-6)) & (mask_gravel == False)
fudge_regions_layer = np.where(mask6, 799., fudge_regions_layer)
fudge_regions_layer_m6 = np.where(np.isin(fudge_regions_layer, [799.]), fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_layer_m6, cmap='Dark2', aspect='equal', norm=norm, extent=grid_extent)
cbar = plt.colorbar(label='', shrink=0.45)
cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-5', 'm-4', 'm-6'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_m6.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# i = 1
# fudge_regions_layer = np.empty_like(hydraulic_conductivities_layers[i])
# fudge_regions_layer[:, :] = np.nan
# i = i + 1
# fig, axes = plt.subplots(figsize=(4, 4))
# bounds = [0, 100, 200, 300, 400, 500, 600, 700, 800]
# norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["viridis"].N)
# fudge_regions_layer = np.where(np.isfinite(fudge_regions_layer_r4), fudge_regions_layer_r4, fudge_regions_layer)
# fudge_regions_layer = np.where(np.isfinite(fudge_regions_layer_r5), fudge_regions_layer_r5, fudge_regions_layer)
# fudge_regions_layer = np.where(np.isfinite(fudge_regions_layer_r6), fudge_regions_layer_r6, fudge_regions_layer)
# fudge_regions_layer = np.where(np.isfinite(fudge_regions_layer_g4), fudge_regions_layer_g4, fudge_regions_layer)
# fudge_regions_layer = np.where(np.isfinite(fudge_regions_layer_z4), fudge_regions_layer_z4, fudge_regions_layer)
# fudge_regions_layer = np.where(np.isfinite(fudge_regions_layer_z5), fudge_regions_layer_z5, fudge_regions_layer)
# fudge_regions_layer = np.where(np.isfinite(fudge_regions_layer_m4), fudge_regions_layer_m4, fudge_regions_layer)
# fudge_regions_layer = np.where(np.isfinite(fudge_regions_layer_m6), fudge_regions_layer_m6, fudge_regions_layer)
# plt.imshow(fudge_regions_layer, cmap='viridis', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
# cbar = plt.colorbar(label='', shrink=0.45)
# cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-5', 'm-4', 'm-6'])
# wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
# wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
# plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / f"fudge_regions_layer{i}.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# [2.7777778e-11, 1.9722222e-07, 1.9999999e-07, 1.8180555e-05 1.8181944e-04
#  1.0000000e-03 1.8181807e-03 3.0000000e-03 4.0000002e-03]

i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["PuBuGn"].N)
fudge_regions_layer[~mask] = np.nan
mask1 = (fudge_regions_layer == 2.7777778e-11)
mask2 = (fudge_regions_layer == 1.9722222e-07)
mask3 = (fudge_regions_layer == 1.9999999e-07)
mask4 = (fudge_regions_layer == 1.8180555e-05)
mask5 = (fudge_regions_layer == 1.8181944e-04)
mask6 = (fudge_regions_layer == 1.0000000e-03)
mask7 = (fudge_regions_layer == 1.8181807e-03)
mask8 = (fudge_regions_layer == 3.0000000e-03)
mask9 = (fudge_regions_layer == 4.0000002e-03)
fudge_regions_layer = np.where(mask3, 1.9722222e-07, fudge_regions_layer)

fudge_regions_layer = np.where(mask1, np.nan, fudge_regions_layer)
fudge_regions_layer = np.where(mask2, 199, fudge_regions_layer)
fudge_regions_layer = np.where(mask4, 299, fudge_regions_layer)
fudge_regions_layer = np.where(mask5, 399, fudge_regions_layer)
fudge_regions_layer = np.where(mask6, 499, fudge_regions_layer)
fudge_regions_layer = np.where(mask7, 599, fudge_regions_layer)
fudge_regions_layer = np.where(mask8, 699, fudge_regions_layer)
fudge_regions_layer = np.where(mask9, 799, fudge_regions_layer)
plt.imshow(fudge_regions_layer, cmap='PuBuGn', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
cbar = plt.colorbar(label='$k_f$ [$m$ $s^{-1}$]', shrink=0.42)
cbar.set_ticks(ticks=[150, 250, 350, 450, 550, 650, 750], labels=['1.9 x $10^{-7}$', '1.8 x $10^{-5}$', '1.8 x $10^{-4}$', '1.8 x $10^{-3}$', '2 x $10^{-3}$', '3 x $10^{-3}$', '4 x $10^{-3}$'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 2
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["PuBuGn"].N)
fudge_regions_layer[~mask] = np.nan
mask1 = (fudge_regions_layer == 2.7777778e-11)
mask2 = (fudge_regions_layer == 1.9722222e-07)
mask3 = (fudge_regions_layer == 1.9999999e-07)
mask4 = (fudge_regions_layer == 1.8180555e-05)
mask5 = (fudge_regions_layer == 1.8181944e-04)
mask6 = (fudge_regions_layer == 1.0000000e-03)
mask7 = (fudge_regions_layer == 1.8181807e-03)
mask8 = (fudge_regions_layer == 3.0000000e-03)
mask9 = (fudge_regions_layer == 4.0000002e-03)
fudge_regions_layer = np.where(mask3, 1.9722222e-07, fudge_regions_layer)

fudge_regions_layer = np.where(mask1, np.nan, fudge_regions_layer)
fudge_regions_layer = np.where(mask2, 199, fudge_regions_layer)
fudge_regions_layer = np.where(mask4, 299, fudge_regions_layer)
fudge_regions_layer = np.where(mask5, 399, fudge_regions_layer)
fudge_regions_layer = np.where(mask6, 499, fudge_regions_layer)
fudge_regions_layer = np.where(mask7, 599, fudge_regions_layer)
fudge_regions_layer = np.where(mask8, 699, fudge_regions_layer)
fudge_regions_layer = np.where(mask9, 799, fudge_regions_layer)
plt.imshow(fudge_regions_layer, cmap='PuBuGn', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
cbar = plt.colorbar(label='$k_f$ [$m$ $s^{-1}$]', shrink=0.42)
cbar.set_ticks(ticks=[150, 250, 350, 450, 550, 650, 750], labels=['1.9 x $10^{-7}$', '1.8 x $10^{-5}$', '1.8 x $10^{-4}$', '1.8 x $10^{-3}$', '2 x $10^{-3}$', '3 x $10^{-3}$', '4 x $10^{-3}$'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 3
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["PuBuGn"].N)
fudge_regions_layer[~mask] = np.nan
mask1 = (fudge_regions_layer == 2.7777778e-11)
mask2 = (fudge_regions_layer == 1.9722222e-07)
mask3 = (fudge_regions_layer == 1.9999999e-07)
mask4 = (fudge_regions_layer == 1.8180555e-05)
mask5 = (fudge_regions_layer == 1.8181944e-04)
mask6 = (fudge_regions_layer == 1.0000000e-03)
mask7 = (fudge_regions_layer == 1.8181807e-03)
mask8 = (fudge_regions_layer == 3.0000000e-03)
mask9 = (fudge_regions_layer == 4.0000002e-03)
fudge_regions_layer = np.where(mask3, 1.9722222e-07, fudge_regions_layer)

fudge_regions_layer = np.where(mask1, np.nan, fudge_regions_layer)
fudge_regions_layer = np.where(mask2, 199, fudge_regions_layer)
fudge_regions_layer = np.where(mask4, 299, fudge_regions_layer)
fudge_regions_layer = np.where(mask5, 399, fudge_regions_layer)
fudge_regions_layer = np.where(mask6, 499, fudge_regions_layer)
fudge_regions_layer = np.where(mask7, 599, fudge_regions_layer)
fudge_regions_layer = np.where(mask8, 699, fudge_regions_layer)
fudge_regions_layer = np.where(mask9, 799, fudge_regions_layer)
plt.imshow(fudge_regions_layer, cmap="PuBuGn", aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
cbar = plt.colorbar(label='$k_f$ [$m$ $s^{-1}$]', shrink=0.42)
cbar.set_ticks(ticks=[150, 250, 350, 450, 550, 650, 750], labels=['1.9 x $10^{-7}$', '1.8 x $10^{-5}$', '1.8 x $10^{-4}$', '1.8 x $10^{-3}$', '2 x $10^{-3}$', '3 x $10^{-3}$', '4 x $10^{-3}$'])
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Greys"].N)
fudge_regions_layer[~mask] = np.nan
mask1 = (fudge_regions_layer == 2.7777778e-11)
mask2 = (fudge_regions_layer == 1.9722222e-07)
mask3 = (fudge_regions_layer == 1.9999999e-07)
mask4 = (fudge_regions_layer == 1.8180555e-05)
mask5 = (fudge_regions_layer == 1.8181944e-04)
mask6 = (fudge_regions_layer == 1.0000000e-03)
mask7 = (fudge_regions_layer == 1.8181807e-03)
mask8 = (fudge_regions_layer == 3.0000000e-03)
mask9 = (fudge_regions_layer == 4.0000002e-03)
fudge_regions_layer = np.where(mask3, 1.9722222e-07, fudge_regions_layer)

fudge_regions_layer = np.where(mask1, np.nan, fudge_regions_layer)
fudge_regions_layer = np.where(mask2, 199, fudge_regions_layer)
fudge_regions_layer = np.where(mask4, 299, fudge_regions_layer)
fudge_regions_layer = np.where(mask5, 399, fudge_regions_layer)
fudge_regions_layer = np.where(mask6, 499, fudge_regions_layer)
fudge_regions_layer = np.where(mask7, 599, fudge_regions_layer)
fudge_regions_layer = np.where(mask8, 699, fudge_regions_layer)
fudge_regions_layer = np.where(mask9, 799, fudge_regions_layer)
fudge_regions_zone1 = np.where(topography < 300, fudge_regions_layer, np.nan)
fudge_regions_zone2 = np.where(topography >= 300, fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_zone1, cmap='Oranges', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
cbar = plt.colorbar(label='$k_f$ [$m$ $s^{-1}$]', shrink=0.38)
cbar.set_ticks(ticks=[150, 250, 350, 450, 550, 650, 750], labels=['1.9 x $10^{-7}$', '1.8 x $10^{-5}$', '1.8 x $10^{-4}$', '1.8 x $10^{-3}$', '2 x $10^{-3}$', '3 x $10^{-3}$', '4 x $10^{-3}$'])
axes.imshow(fudge_regions_zone2, cmap='Blues', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='magenta')
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_zonal1.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 1
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Greys"].N)
fudge_regions_layer[~mask] = np.nan
mask1 = (fudge_regions_layer == 2.7777778e-11)
mask2 = (fudge_regions_layer == 1.9722222e-07)
mask3 = (fudge_regions_layer == 1.9999999e-07)
mask4 = (fudge_regions_layer == 1.8180555e-05)
mask5 = (fudge_regions_layer == 1.8181944e-04)
mask6 = (fudge_regions_layer == 1.0000000e-03)
mask7 = (fudge_regions_layer == 1.8181807e-03)
mask8 = (fudge_regions_layer == 3.0000000e-03)
mask9 = (fudge_regions_layer == 4.0000002e-03)
fudge_regions_layer = np.where(mask3, 1.9722222e-07, fudge_regions_layer)

fudge_regions_layer = np.where(mask1, np.nan, fudge_regions_layer)
fudge_regions_layer = np.where(mask2, 199, fudge_regions_layer)
fudge_regions_layer = np.where(mask4, 299, fudge_regions_layer)
fudge_regions_layer = np.where(mask5, 399, fudge_regions_layer)
fudge_regions_layer = np.where(mask6, 499, fudge_regions_layer)
fudge_regions_layer = np.where(mask7, 599, fudge_regions_layer)
fudge_regions_layer = np.where(mask8, 699, fudge_regions_layer)
fudge_regions_layer = np.where(mask9, 799, fudge_regions_layer)
fudge_regions_zone1 = np.where(topography < 300, fudge_regions_layer, np.nan)
fudge_regions_zone2 = np.where(topography >= 300, fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_zone2, cmap='Blues', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
cbar = plt.colorbar(label='$k_f$ [$m$ $s^{-1}$]', shrink=0.38)
cbar.set_ticks(ticks=[150, 250, 350, 450, 550, 650, 750], labels=['1.9 x $10^{-7}$', '1.8 x $10^{-5}$', '1.8 x $10^{-4}$', '1.8 x $10^{-3}$', '2 x $10^{-3}$', '3 x $10^{-3}$', '4 x $10^{-3}$'])
axes.imshow(fudge_regions_zone1, cmap='Oranges', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='magenta')
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_zonal2.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 2
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Greys"].N)
fudge_regions_layer[~mask] = np.nan
mask1 = (fudge_regions_layer == 2.7777778e-11)
mask2 = (fudge_regions_layer == 1.9722222e-07)
mask3 = (fudge_regions_layer == 1.9999999e-07)
mask4 = (fudge_regions_layer == 1.8180555e-05)
mask5 = (fudge_regions_layer == 1.8181944e-04)
mask6 = (fudge_regions_layer == 1.0000000e-03)
mask7 = (fudge_regions_layer == 1.8181807e-03)
mask8 = (fudge_regions_layer == 3.0000000e-03)
mask9 = (fudge_regions_layer == 4.0000002e-03)
fudge_regions_layer = np.where(mask3, 1.9722222e-07, fudge_regions_layer)

fudge_regions_layer = np.where(mask1, np.nan, fudge_regions_layer)
fudge_regions_layer = np.where(mask2, 199, fudge_regions_layer)
fudge_regions_layer = np.where(mask4, 299, fudge_regions_layer)
fudge_regions_layer = np.where(mask5, 399, fudge_regions_layer)
fudge_regions_layer = np.where(mask6, 499, fudge_regions_layer)
fudge_regions_layer = np.where(mask7, 599, fudge_regions_layer)
fudge_regions_layer = np.where(mask8, 699, fudge_regions_layer)
fudge_regions_layer = np.where(mask9, 799, fudge_regions_layer)
fudge_regions_zone1 = np.where(topography < 300, fudge_regions_layer, np.nan)
fudge_regions_zone2 = np.where(topography >= 300, fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_zone2, cmap='Blues', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
cbar = plt.colorbar(label='$k_f$ [$m$ $s^{-1}$]', shrink=0.38)
cbar.set_ticks(ticks=[150, 250, 350, 450, 550, 650, 750], labels=['1.9 x $10^{-7}$', '1.8 x $10^{-5}$', '1.8 x $10^{-4}$', '1.8 x $10^{-3}$', '2 x $10^{-3}$', '3 x $10^{-3}$', '4 x $10^{-3}$'])
axes.imshow(fudge_regions_zone1, cmap='Oranges', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='magenta')
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_zonal.png"
fig.savefig(file, dpi=300)
plt.close(fig)

i = 3
fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
i = i + 1
fig, axes = plt.subplots(figsize=(4, 4))
bounds = [100, 200, 300, 400, 500, 600, 700, 800]
norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Greys"].N)
fudge_regions_layer[~mask] = np.nan
mask1 = (fudge_regions_layer == 2.7777778e-11)
mask2 = (fudge_regions_layer == 1.9722222e-07)
mask3 = (fudge_regions_layer == 1.9999999e-07)
mask4 = (fudge_regions_layer == 1.8180555e-05)
mask5 = (fudge_regions_layer == 1.8181944e-04)
mask6 = (fudge_regions_layer == 1.0000000e-03)
mask7 = (fudge_regions_layer == 1.8181807e-03)
mask8 = (fudge_regions_layer == 3.0000000e-03)
mask9 = (fudge_regions_layer == 4.0000002e-03)
fudge_regions_layer = np.where(mask3, 1.9722222e-07, fudge_regions_layer)

fudge_regions_layer = np.where(mask1, np.nan, fudge_regions_layer)
fudge_regions_layer = np.where(mask2, 199, fudge_regions_layer)
fudge_regions_layer = np.where(mask4, 299, fudge_regions_layer)
fudge_regions_layer = np.where(mask5, 399, fudge_regions_layer)
fudge_regions_layer = np.where(mask6, 499, fudge_regions_layer)
fudge_regions_layer = np.where(mask7, 599, fudge_regions_layer)
fudge_regions_layer = np.where(mask8, 699, fudge_regions_layer)
fudge_regions_layer = np.where(mask9, 799, fudge_regions_layer)
fudge_regions_zone1 = np.where(topography < 300, fudge_regions_layer, np.nan)
fudge_regions_zone2 = np.where(topography >= 300, fudge_regions_layer, np.nan)
plt.imshow(fudge_regions_zone2, cmap='Blues', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
cbar = plt.colorbar(label='$k_f$ [$m$ $s^{-1}$]', shrink=0.38)
cbar.set_ticks(ticks=[150, 250, 350, 450, 550, 650, 750], labels=['1.9 x $10^{-7}$', '1.8 x $10^{-5}$', '1.8 x $10^{-4}$', '1.8 x $10^{-3}$', '2 x $10^{-3}$', '3 x $10^{-3}$', '4 x $10^{-3}$'])
axes.imshow(fudge_regions_zone1, cmap='Oranges', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='magenta')
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"fudge_regions_layer{i}_zonal.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# i = 1
# fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
# i = i + 1
# fig, axes = plt.subplots(figsize=(4, 4))
# bounds = [100, 200, 300, 400, 500]
# norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["viridis"].N)
# fudge_regions_layer[~mask] = np.nan
# mask1 = (fudge_regions_layer <= 10**-10)
# mask2 = (fudge_regions_layer >= 10**-7) & (fudge_regions_layer < 10**-6)
# mask3 = (fudge_regions_layer >= 10**-5) & (fudge_regions_layer < 10**-4)
# mask4 = (fudge_regions_layer >= 10**-4) & (fudge_regions_layer < 10**-3)
# mask5 = (fudge_regions_layer >= 10**-3) & (fudge_regions_layer < 10**-2)

# fudge_regions_layer = np.where(mask1, np.nan, fudge_regions_layer)
# fudge_regions_layer = np.where(mask2, 199, fudge_regions_layer)
# fudge_regions_layer = np.where(mask3, 299, fudge_regions_layer)
# fudge_regions_layer = np.where(mask4, 399, fudge_regions_layer)
# fudge_regions_layer = np.where(mask5, 499, fudge_regions_layer)
# plt.imshow(fudge_regions_layer, cmap='viridis', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
# cbar = plt.colorbar(label='', shrink=0.45)
# cbar.set_ticks(ticks=[150, 250, 350, 450], labels=['-7', '-5', '-4', '-3'])
# wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
# wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
# plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / f"fudge_regions_layer{i}.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# i = 2
# fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
# i = i + 1
# fig, axes = plt.subplots(figsize=(4, 4))
# bounds = [0, 100.1, 200.1, 300.1, 400.1, 500.1, 600.1, 700.1, 800.1, 900.1]
# norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["viridis"].N)
# fudge_regions_layer[~mask] = np.nan
# mask1 = (topography < 450) & (fudge_regions_layer > 10**(-4)) & (mask_gravel == False)
# mask12 = (topography < 450) & (fudge_regions_layer > 10**(-4)) & (mask_gravel == True)
# mask2 = (topography < 450) & (fudge_regions_layer > 10**(-6)) & (fudge_regions_layer <= 10**(-5))
# mask3 = (topography < 450) &  (fudge_regions_layer <= 10**(-6))
# mask4 = (topography >= 450) & (fudge_regions_layer > 10**(-4))
# mask5 = (topography >= 450) & (fudge_regions_layer > 10**(-5)) & (fudge_regions_layer <= 10**(-4))
# mask6 = (topography >= 450) & (fudge_regions_layer <= 10**(-5))
# fudge_regions_layer = np.where(mask1, 100, fudge_regions_layer)
# fudge_regions_layer = np.where(mask2, 200, fudge_regions_layer)
# fudge_regions_layer = np.where(mask3, 300, fudge_regions_layer)
# fudge_regions_layer = np.where(mask12, 400, fudge_regions_layer)
# fudge_regions_layer[mask1 & mask_valleys_black_forest] = np.where(mask1 & mask_valleys_black_forest, 500, fudge_regions_layer)[mask1 & mask_valleys_black_forest]
# fudge_regions_layer[mask3 & mask_valleys_black_forest] = np.where(mask3 & mask_valleys_black_forest, 600, fudge_regions_layer)[mask3 & mask_valleys_black_forest]
# fudge_regions_layer = np.where(mask4, 700, fudge_regions_layer)
# fudge_regions_layer = np.where(mask5, 800, fudge_regions_layer)
# fudge_regions_layer = np.where(mask6, 900, fudge_regions_layer)
# fudge_regions_layer = np.where(np.isin(fudge_regions_layer, [100, 200, 300, 400, 500, 600, 700, 800, 900]), fudge_regions_layer, np.nan)
# plt.imshow(fudge_regions_layer, cmap='viridis', aspect='equal', norm=norm, extent=grid_extent, interpolation='nearest')
# cbar = plt.colorbar(label='', shrink=0.45)
# cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750, 850], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-6', 'm-4', 'm-5', 'm-6'])
# wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
# wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
# plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / f"fudge_regions_layer{i}.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# i = 3
# fudge_regions_layer = hydraulic_conductivities_layers[i]/(24*60*60)
# i = i + 1
# fig, axes = plt.subplots(figsize=(4, 4))
# bounds = [0, 100.1, 200.1, 300.1, 400.1, 500.1, 600.1, 700.1, 800.1, 900.1]
# norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Dark2"].N)
# fudge_regions_layer[~mask] = np.nan
# mask1 = (topography < 450) & (fudge_regions_layer > 10**(-4)) & (mask_gravel == False)
# mask12 = (topography < 450) & (fudge_regions_layer > 10**(-4)) & (mask_gravel == True)
# mask2 = (topography < 450) & (fudge_regions_layer > 10**(-6)) & (fudge_regions_layer <= 10**(-5))
# mask3 = (topography < 450) & (fudge_regions_layer > 10**(-6)) & (fudge_regions_layer <= 10**(-6))
# mask4 = (topography >= 450) & (fudge_regions_layer > 10**(-4))
# mask5 = (topography >= 450) & (fudge_regions_layer > 10**(-5)) & (fudge_regions_layer <= 10**(-4))
# mask6 = (topography >= 450) & (fudge_regions_layer <= 10**(-5))
# fudge_regions_layer = np.where(mask1, 100, fudge_regions_layer)
# fudge_regions_layer = np.where(mask2, 200, fudge_regions_layer)
# fudge_regions_layer = np.where(mask3, 300, fudge_regions_layer)
# fudge_regions_layer = np.where(mask12, 400, fudge_regions_layer)
# fudge_regions_layer[mask1 & mask_valleys_black_forest] = np.where(mask1 & mask_valleys_black_forest, 500, fudge_regions_layer)[mask1 & mask_valleys_black_forest]
# fudge_regions_layer[mask3 & mask_valleys_black_forest] = np.where(mask3 & mask_valleys_black_forest, 600, fudge_regions_layer)[mask3 & mask_valleys_black_forest]
# fudge_regions_layer = np.where(mask4, 700, fudge_regions_layer)
# fudge_regions_layer = np.where(mask5, 800, fudge_regions_layer)
# fudge_regions_layer = np.where(mask6, 900, fudge_regions_layer)
# fudge_regions_layer = np.where(np.isin(fudge_regions_layer, [100, 200, 300, 400, 500, 600, 700, 800, 900]), fudge_regions_layer, np.nan)
# plt.imshow(fudge_regions_layer, cmap='Dark2', aspect='equal', norm=norm, extent=grid_extent)
# cbar = plt.colorbar(label='', shrink=0.45)
# cbar.set_ticks(ticks=[50, 150, 250, 350, 450, 550, 650, 750, 850], labels=['r-4', 'r-5', 'r-6', 'g-4', 'z-4', 'z-6', 'm-4', 'm-5', 'm-6'])
# wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
# wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
# plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=2, c='black')
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / f"fudge_regions_layer{i}.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)


# bins = [10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0]
# fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=True)
# axes[0].hist(hydraulic_conductivities_layer1[mask]/(24*60*60), bins=bins, color='black', alpha=1, label='Layer 1')
# axes[0].set_ylabel('layer 1')
# axes[0].set_xscale('log')
# axes[0].set_yscale('log')
# axes[1].hist(hydraulic_conductivities_layer2[mask]/(24*60*60), bins=bins, color='black', alpha=1, label='Layer 2')
# axes[1].set_ylabel('layer 2')
# axes[1].set_xscale('log')
# axes[1].set_yscale('log')
# axes[2].hist(hydraulic_conductivities_layer3[mask]/(24*60*60), bins=bins, color='black', alpha=1, label='Layer 3')
# axes[2].set_ylabel('layer 3')
# axes[2].set_xscale('log')
# axes[2].set_yscale('log')
# axes[3].hist(hydraulic_conductivities_layer4[mask]/(24*60*60), bins=bins, color='black', alpha=1, label='Layer 4')
# axes[3].set_ylabel('layer 4')
# axes[3].set_xlabel('[m/s]')
# axes[3].set_xscale('log')
# axes[3].set_yscale('log')
# fig.tight_layout()
# file = base_path_figs / "hydraulic_conductivities_histogram.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# mask_mountains = (topography > 600)
# mask_valleys = (topography <= 600)
# fig, axes = plt.subplots(4, 3, figsize=(6, 6), sharex=True, sharey=True)
# axes[0, 0].hist(hydraulic_conductivities_layer1[mask]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[0, 0].set_ylabel('layer 1')
# axes[0, 0].set_xscale('log')
# axes[0, 0].set_yscale('log')
# axes[0, 0].set_title('', fontsize=10)
# axes[1, 0].hist(hydraulic_conductivities_layer2[mask]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[1, 0].set_ylabel('layer 2')
# axes[1, 0].set_xscale('log')
# axes[1, 0].set_yscale('log')
# axes[2, 0].hist(hydraulic_conductivities_layer3[mask]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[2, 0].set_ylabel('layer 3')
# axes[2, 0].set_xscale('log')
# axes[2, 0].set_yscale('log')
# axes[3, 0].hist(hydraulic_conductivities_layer4[mask]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[3, 0].set_ylabel('layer 4')
# axes[3, 0].set_xlabel('[m/s]')
# axes[3, 0].set_xscale('log')
# axes[3, 0].set_yscale('log')

# axes[0, 1].hist(hydraulic_conductivities_layer1[mask_valleys]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[0, 1].set_xscale('log')
# axes[0, 1].set_yscale('log')
# axes[0, 1].set_title('Valley region', fontsize=10)
# axes[1, 1].hist(hydraulic_conductivities_layer2[mask_valleys]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[1, 1].set_xscale('log')
# axes[1, 1].set_yscale('log')
# axes[2, 1].hist(hydraulic_conductivities_layer3[mask_valleys]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[2, 1].set_xscale('log')
# axes[2, 1].set_yscale('log')
# axes[3, 1].hist(hydraulic_conductivities_layer4[mask_valleys]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[3, 1].set_xlabel('[m/s]')
# axes[3, 1].set_xscale('log')
# axes[3, 1].set_yscale('log')

# axes[0, 2].hist(hydraulic_conductivities_layer1[mask_mountains]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[0, 2].set_xscale('log')
# axes[0, 2].set_yscale('log')
# axes[0, 2].set_title('Mountain region', fontsize=10)
# axes[1, 2].hist(hydraulic_conductivities_layer2[mask_mountains]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[1, 2].set_xscale('log')
# axes[1, 2].set_yscale('log')
# axes[2, 2].hist(hydraulic_conductivities_layer3[mask_mountains]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[2, 2].set_xscale('log')
# axes[2, 2].set_yscale('log')
# axes[3, 2].hist(hydraulic_conductivities_layer4[mask_mountains]/(24*60*60), bins=bins, color='black', alpha=1)
# axes[3, 2].set_xlabel('[m/s]')
# axes[3, 2].set_xscale('log')
# axes[3, 2].set_yscale('log')
# fig.tight_layout()
# file = base_path_figs / "hydraulic_conductivities_histogram_mountains_valleys.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# specific_yield_layer1 = ds_params['sy'].isel(layer=0).values
# specific_yield_layer2 = ds_params['sy'].isel(layer=1).values
# specific_yield_layer3 = ds_params['sy'].isel(layer=2).values
# specific_yield_layer4 = ds_params['sy'].isel(layer=3).values
# specific_yield_layers = [specific_yield_layer1, specific_yield_layer2, specific_yield_layer3, specific_yield_layer4]
# for i, specific_yield_layer in enumerate(specific_yield_layers):
#     i = i + 1
#     fig, axes = plt.subplots(figsize=(4, 4))
#     specific_yield_layer[~mask] = np.nan
#     plt.imshow(specific_yield_layer, extent=grid_extent, cmap='Oranges', aspect='equal')
#     plt.colorbar(label='$n$ [-]', shrink=0.5)
#     plt.grid(zorder=0)
#     plt.xlabel('Distance in x-direction [m]')
#     plt.ylabel('Distance in y-direction [m]')
#     plt.tight_layout()
#     file = base_path_figs / f"specific_yield_layer_{i}.png"
#     fig.savefig(file, dpi=300)
#     plt.close(fig)
#     specific_yield_layer[~mask] = 0
#     print(f"Specific yield Layer {i} (Number of no data): ", np.isnan(specific_yield_layer).sum())

# fig, axes = plt.subplots(figsize=(4, 4))
# topography[~mask] = np.nan
# elevation_bottom_layer1[~mask] = np.nan
# thickness_layer1 = topography - elevation_bottom_layer1
# thickness_layer1[thickness_layer1 <= 0] = np.nan
# print("Thickness Layer 1 (Minimum): ", np.nanmin(thickness_layer1))
# print("Thickness Layer 1 (Number of grid cells with negative values): ", np.nansum(thickness_layer1 < 0))
# plt.imshow(thickness_layer1, extent=grid_extent, cmap='viridis', aspect='equal')
# plt.colorbar(label='[m]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / f"thickness_layer1.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# fig, axes = plt.subplots(figsize=(4, 4))
# elevation_bottom_layer1[~mask] = np.nan
# elevation_bottom_layer2[~mask] = np.nan
# thickness_layer2 = elevation_bottom_layer1 - elevation_bottom_layer2
# thickness_layer2[thickness_layer2 <= 0] = np.nan
# print("Thickness Layer 2 (Minimum): ", np.nanmin(thickness_layer2))
# print("Thickness Layer 2 (Number of grid cells with negative values): ", np.nansum(thickness_layer2 < 0))
# plt.imshow(thickness_layer2, extent=grid_extent, cmap='viridis', aspect='equal', vmin=5, vmax=25)
# plt.colorbar(label='[m]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / f"thickness_layer2.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# fig, axes = plt.subplots(figsize=(4, 4))
# elevation_bottom_layer2[~mask] = np.nan
# elevation_bottom_layer3[~mask] = np.nan
# thickness_layer3 = elevation_bottom_layer2 - elevation_bottom_layer3
# thickness_layer3[thickness_layer3 <= 0] = np.nan
# print("Thickness Layer 3 (Minimum): ", np.nanmin(thickness_layer3))
# print("Thickness Layer 3 (Number of grid cells with negative values): ", np.nansum(thickness_layer3 < 0))
# plt.imshow(thickness_layer3, extent=grid_extent, cmap='viridis', aspect='equal', vmin=5, vmax=25)
# plt.colorbar(label='[m]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / f"thickness_layer3.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# fig, axes = plt.subplots(figsize=(4, 4))
# elevation_bottom_layer3[~mask] = np.nan
# elevation_bottom_layer4[~mask] = np.nan
# thickness_layer4 = elevation_bottom_layer3 - elevation_bottom_layer4
# thickness_layer4[thickness_layer4 <= 0] = np.nan
# print("Thickness Layer 4 (Minimum): ", np.nanmin(thickness_layer4))
# print("Thickness Layer 4 (Number of grid cells with negative values): ", np.nansum(thickness_layer4 < 0))
# plt.imshow(thickness_layer4, extent=grid_extent, cmap='viridis', aspect='equal', vmin=5, vmax=25)
# plt.colorbar(label='[m]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / f"thickness_layer4.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# thickness_layers = [thickness_layer1, thickness_layer2, thickness_layer3, thickness_layer4]
# fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=False)
# x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
# for layer in range(4):
#     z = thickness_layers[layer][82, :-1]
#     axes[layer].plot(x, z, ls='-', lw=1, color='black')
#     axes[layer].set_xlim(0, x[-1])
#     axes[layer].set_ylim(0, )
# axes[-1].set_xlabel('Distance in x-direction [m]')
# axes[0].set_ylabel('layer 1 \n aquifer thickness \n[m]')
# axes[1].set_ylabel('layer 2 \n aquifer thickness \n[m]')
# axes[2].set_ylabel('layer 3 \n aquifer thickness \n[m]')
# axes[3].set_ylabel('layer 4 \n aquifer thickness \n[m]')
# fig.tight_layout()
# file = base_path_figs / "gw_thickness_x_cross_section_layer.png"
# fig.savefig(file, dpi=300)
# plt.close("all")

# fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=False)
# x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
# axes.plot(x, topography[82, :-1], ls='-', lw=2, color='black')
# axes.plot(x, elevation_bottom_layer1[82, :-1], ls='--', lw=1.75, color='black', alpha=0.9)
# axes.plot(x, elevation_bottom_layer2[82, :-1], ls='-.', lw=1.5, color='black', alpha=0.8)
# axes.plot(x, elevation_bottom_layer3[82, :-1], ls=':', lw=1.25, color='black', alpha=0.7)
# axes.plot(x, elevation_bottom_layer4[82, :-1], ls='-', lw=1, color='black', alpha=0.6)
# axes.set_xlim(0, x[-1])
# # axes.set_ylim(0, )
# axes.set_xlabel('Distance in x-direction [m]')
# axes.set_ylabel('elevation \n[m a.s.l.]')
# fig.tight_layout()
# file = base_path_figs / "gw_layers_cross_section_west-east_layer.png"
# fig.savefig(file, dpi=300)
# plt.close("all")


# transmissivity_layer1 = hydraulic_conductivities_layer1 * thickness_layer1
# transmissivity_layer2 = hydraulic_conductivities_layer2 * thickness_layer2
# transmissivity_layer3 = hydraulic_conductivities_layer3 * thickness_layer3
# transmissivity_layer4 = hydraulic_conductivities_layer4 * thickness_layer4
# transmissivity_layers = [transmissivity_layer1, transmissivity_layer2, transmissivity_layer3, transmissivity_layer4]
# for i, transmissivity_layer in enumerate(transmissivity_layers):
#     i = i + 1
#     fig, axes = plt.subplots(figsize=(4, 4))
#     transmissivity_layer[~mask] = np.nan
#     plt.imshow(transmissivity_layer, extent=grid_extent, cmap='Oranges', aspect='equal')
#     plt.colorbar(label='$T$ [$m^2$/s]', shrink=0.5)
#     plt.grid(zorder=0)
#     plt.xlabel('Distance in x-direction [m]')
#     plt.ylabel('Distance in y-direction [m]')
#     plt.tight_layout()
#     file = base_path_figs / f"transmissivity_layer_{i}.png"
#     fig.savefig(file, dpi=300)
#     plt.close(fig)


# params_file = base_path / "parameters.nc"
# ds_params = xr.open_dataset(params_file, engine="h5netcdf")
# data1 = ds_params['TP'].values / (1000 * 60 * 60)

# fig, axes = plt.subplots(2, 1, figsize=(4, 6))
# ax1 = axes[0].imshow(data1, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.nanmax(data1)))
# fig.colorbar(ax1, shrink=0.45, label='[m/s]')
# axes[0].set_xlabel('Distance in y-direction [m]')
# axes[0].set_ylabel('Distance in x-direction [m]')
# ax2 = axes[1].imshow(hydraulic_conductivities_layer1, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.nanmax(data1)))
# axes[1].set_xlabel('Distance in y-direction [m]')
# axes[1].set_ylabel('Distance in x-direction [m]')
# fig.colorbar(ax2, shrink=0.45, label='[m/s]')
# fig.tight_layout()
# file = base_path_figs / "kf_comparison_layer1.png"
# fig.savefig(file, dpi=300)
# plt.close("all")

# fig, axes = plt.subplots(2, 1, figsize=(4, 6))
# ax1 = axes[0].imshow(data1, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.nanmax(data1)))
# fig.colorbar(ax1, shrink=0.45, label='[m/s]')
# axes[0].set_xlabel('Distance in y-direction [m]')
# axes[0].set_ylabel('Distance in x-direction [m]')
# ax2 = axes[1].imshow(hydraulic_conductivities_layer2, extent=grid_extent, aspect='equal', vmin=0, vmax=int(np.nanmax(data1)))
# axes[1].set_xlabel('Distance in y-direction [m]')
# axes[1].set_ylabel('Distance in x-direction [m]')
# fig.colorbar(ax2, shrink=0.45, label='[m/s]')
# fig.tight_layout()
# file = base_path_figs / "kf_comparison_layer2.png"
# fig.savefig(file, dpi=300)
# plt.close("all")

# fig, axes = plt.subplots(figsize=(4, 4))
# data = hydraulic_conductivities_layer2 * (1000 * 60 * 60)
# data[~mask] = np.nan
# plt.imshow(data, extent=grid_extent, cmap='Oranges', aspect='equal')
# plt.colorbar(label='$k_f$ [mm/hour]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path_figs / f"hydraulic_conductivity_layer_2_mmh.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# constant_head = ds_bc['constant_head'].values
# fig, axes = plt.subplots(figsize=(4, 4))
# topography[~mask] = np.nan
# axes.scatter(np.where(constant_head == 1)[1]*.05, np.where(constant_head == 1)[0]*.05, s=0.5, c='r')
# plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=(0, (modflow_config['ny']*modflow_config['dy'])/1000, (modflow_config['nx']*modflow_config['dx'])/1000, 0))
# plt.colorbar(label='[m a.s.l.]', shrink=0.5, alpha=1)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [km]')
# plt.ylabel('Distance in y-direction [km]')
# plt.tight_layout()
# file = base_path_figs / "boundary_condition.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)

# extract thickness of the wells
wells_y = [266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]
wells_x = [66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]
elevation_layers = [topography, elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

df_wells_elevation = pd.DataFrame(index=['A2', 'A3', 'A4', 'B1', 'B4', 'C1', 'HU1', 'HU2', 'HU3', 'K2', 'K5', 'S2', 'S4'], columns=['z0', 'z1', 'z2', 'z3', 'z4'])
df_wells_kf = pd.DataFrame(index=['A2', 'A3', 'A4', 'B1', 'B4', 'C1', 'HU1', 'HU2', 'HU3', 'K2', 'K5', 'S2', 'S4'], columns=['z1', 'z2', 'z3', 'z4'])

for i in range(5):
    elevation_layer = elevation_layers[i]
    for j in range(len(wells_x)):
        x = wells_x[j]
        y = wells_y[j]
        df_wells_elevation.iloc[j, i] = elevation_layer[y, x]
        if i > 0:
            df_wells_kf.iloc[j, i-1] = hydraulic_conductivities_layers[i-1][y, x]

df_wells_thickness = pd.DataFrame(index=['A2', 'A3', 'A4', 'B1', 'B4', 'C1', 'HU1', 'HU2', 'HU3', 'K2', 'K5', 'S2', 'S4'], columns=['z1', 'z2', 'z3', 'z4'])
for i in range(4):
    thickness_layer = elevation_layers[i] - elevation_layers[i+1]
    for j in range(len(wells_x)):
        x = wells_x[j]
        y = wells_y[j]
        df_wells_thickness.iloc[j, i] = thickness_layer[y, x]
df_wells_thickness_ = df_wells_thickness[['z4', 'z3', 'z2', 'z1']]
df_wells_thickness_.loc[:, 'z4'] = df_wells_thickness_.loc[:, 'z4']
fig, ax = plt.subplots(figsize=(4, 2))
df_wells_thickness_.plot(kind='bar', stacked=True, ax=ax)
ax.set_ylabel('Elevation [m a.s.l.]')
ax.set_xlabel('Well')
ax.set_ylim(0, )
ax.legend(ncol=2, loc='upper left', frameon=False)
fig.tight_layout()
file = base_path_figs / "wells_thickness.png"
fig.savefig(file, dpi=300)
plt.close(fig)