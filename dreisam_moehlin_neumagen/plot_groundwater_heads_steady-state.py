import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from flopy.utils import Raster

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join("..", ".."))
    sys.path.append(fpth)
    import flopy

base_path = Path(__file__).parent

# load the config file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    roger_config = yaml.safe_load(file)

res_modflow = 50  # spatial resolution of MODFLOW in meters

modflow_config = {
    'dx': res_modflow,
    'dy': res_modflow,
    'nx': 621,
    'ny': 777,
    'nz': 4,
}
grid_extent = (0, modflow_config['ny'] * modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])

# load MODFLOW parameters
path = Path(__file__).parent / "parameters_modflow.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")

topography = ds_params['elevations'].isel(z=0).values
elevation_bottom_layer1 = ds_params['elevations'].isel(z=1).values
elevation_bottom_layer2 = ds_params['elevations'].isel(z=2).values
elevation_bottom_layer3 = ds_params['elevations'].isel(z=3).values
elevation_bottom_layer4 = ds_params['elevations'].isel(z=4).values
elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]
elevation_layers = [topography, elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

hydraulic_conductivities_layer1 = ds_params['kf'].isel(layer=0).values
hydraulic_conductivities_layer2 = ds_params['kf'].isel(layer=1).values
hydraulic_conductivities_layer3 = ds_params['kf'].isel(layer=2).values
hydraulic_conductivities_layer4 = ds_params['kf'].isel(layer=3).values
hydraulic_conductivities_layers = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]

mask = np.isfinite(topography)
domain = np.empty_like(topography)
domain[mask] = 1
domain[~mask] = -1

model_type = "steady-state"
base_path_figs = base_path / "figures" / model_type
sim = flopy.mf6.MFSimulation.load(
    sim_ws=base_path / "output" / model_type,
    exe_name="mf6",
    version="mf6",
    verbosity_level=0,
)

ml = sim.get_model("dmn_run_0")

# load the netcdf file
output_file = base_path / "output" / model_type / "modflow_output_run_0.nc"
ds_mf = xr.open_dataset(output_file, engine="h5netcdf")

x = np.cumsum(ds_mf.delr.values)
y = np.cumsum(ds_mf.delc.values)
yr = y[::-1]

# get the groundwater head
h = ds_mf["head"].values[:, 0, :, :]

# We can also use the Flopy PlotMapView capabilities for MODFLOW 6
iper = 0
ra = ml.chd.stress_period_data.get_data(key=iper)

ibd = np.ones((ds_mf.sizes['layer'], ds_mf.sizes['y'], ds_mf.sizes['x']), dtype=int)
for k, i, j in ra["cellid"]:
    ibd[k, i, j] = -1

# plot the heads for all time steps
t = 0
# fig, axes = plt.subplots(figsize=(3, 3))
# c = plt.contour(x, yr, h[t, :, :])
# plt.clabel(c, fmt="%2.1f")
# plt.axis("scaled")
# axes.set_xlabel("distance in x-direction [m]")
# axes.set_ylabel("distance in y-direction [m]")
# fig.tight_layout()
# path = base_path_figs / f"heads_{t}.png"
# fig.savefig(path, dpi=300)

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(1, 1, 1, aspect="equal")
# modelmap = flopy.plot.PlotMapView(model=sim.gwf[0], ax=ax)

# # Then we can use the plot_grid() method to draw the grid
# # The return value for this function is a matplotlib LineCollection object,
# # which could be manipulated (or used) later if necessary.
# quadmesh = modelmap.plot_ibound(ibound=ibd)
# linecollection = modelmap.plot_grid()
# contours = modelmap.contour_array(h[t, :, :])
# ax.set_xlabel("distance in x-direction [m]")
# ax.set_ylabel("distance in y-direction [m]")
# fig.tight_layout()
# path = base_path_figs / f"heads_contrours_grid_{t}.png"
# fig.savefig(path, dpi=300)

# fig = plt.figure(figsize=(6, 6))
# ax = fig.add_subplot(1, 1, 1, aspect="equal")
# # Next we create an instance of the ModelMap class
# modelmap = flopy.plot.PlotMapView(model=sim.gwf[0], ax=ax)

# # Then we can use the plot_grid() method to draw the grid
# # The return value for this function is a matplotlib LineCollection object,
# # which could be manipulated (or used) later if necessary.
# quadmesh = modelmap.plot_ibound(ibound=ibd)
# linecollection = modelmap.plot_grid()
# pa = modelmap.plot_array(h[t, :, :])
# cb = plt.colorbar(pa, shrink=0.5, label="groundwater head [m.a.s.l.]")
# plt.xlabel("distance in x-direction [m]")
# plt.ylabel("distance in y-direction [m]")
# fig.tight_layout()
# path = base_path_figs / f"heads_grid_steady_state.png"
# fig.savefig(path, dpi=300)

ll_levels = [[150, 200, 300, 400, 500, 600],
             [150, 200, 300, 400, 500, 600],
             [150, 200, 300, 400, 500, 600],
             [150, 200, 300, 400, 500, 600]]

for layer in range(4):
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(ds_mf['head'].isel(time=0, layer=layer).values, extent=grid_extent, cmap='viridis', aspect='equal', vmin=100, vmax=600)
    plt.colorbar(label='groundwater head \n[m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"gw_head_steady_state_layer{i}_grid.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    elevation_bottom_layer = elevation_bottom_layers[layer]
    gw_thickness = ds_mf['head'].isel(time=0, layer=layer).values - elevation_bottom_layer
    gw_thickness[gw_thickness <= 0] = 0
    plt.imshow(gw_thickness, extent=grid_extent, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater thickness [m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"gw_thickness_steady_state_layer{i}_grid.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    gw_depth = topography - ds_mf['head'].isel(time=0, layer=layer).values
    plt.imshow(gw_depth, extent=grid_extent, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater depth [m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"gw_depth_steady_state_layer{i}_grid.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    y = np.arange(0, modflow_config['nx']*modflow_config['dx'], modflow_config['dx'])[::-1]
    x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
    X, Y = np.meshgrid(x, y)
    Z = ds_mf['head'].isel(time=0, layer=layer).values
    levels = ll_levels[layer]
    CS = axes.contour(X, Y, Z, levels, colors='black')
    axes.clabel(CS, inline=True, fontsize=8, colors='black')
    axes.imshow(mask, extent=grid_extent, cmap='Greys', alpha=0.25)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.title(f"Groundwater head of layer {layer + 1} [m a.s.l.]", fontsize=8)
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"gw_head_steady_state_layer{i}_contour.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

# wells_y = [266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]
# wells_x = [66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]

# for yy in wells_y:
#     fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=True)
#     x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
#     for layer in range(4):
#         z0 = topography[yy, :]
#         z1 = elevation_bottom_layer1[yy, :]
#         z2 = elevation_bottom_layer2[yy, :]
#         z3 = elevation_bottom_layer3[yy, :]
#         z4 = elevation_bottom_layer4[yy, :]
#         z = ds_mf['head'].isel(time=0, layer=layer).values[yy, :]
#         axes[layer].plot(x, z, ls='-', lw=1, color='black')
#         axes[layer].plot(x, z0, ls='-', lw=1, color='grey')
#         axes[layer].plot(x, z1, ls='-', lw=1, color='grey')
#         axes[layer].plot(x, z2, ls='-', lw=1, color='grey')
#         axes[layer].plot(x, z3, ls='-', lw=1, color='grey')
#         axes[layer].plot(x, z4, ls='-', lw=1, color='grey')
#         axes[layer].set_xlim(0, x[-1])
#         axes[layer].set_ylabel('[m a.s.l.]')
#     axes[-1].set_xlabel('Distance in W-E direction [m]')
#     fig.tight_layout()
#     file = base_path_figs / f"gw_head_x{yy}_cross_section_layer.png"
#     fig.savefig(file, dpi=300)
#     plt.close("all")

# for xx in wells_x:
#     fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=True)
#     x = np.arange(0, modflow_config['nx']*modflow_config['dx'], modflow_config['dx'])
#     for layer in range(4):
#         z0 = topography[:, xx]
#         z1 = elevation_bottom_layer1[:, xx]
#         z2 = elevation_bottom_layer2[:, xx]
#         z3 = elevation_bottom_layer3[:, xx]
#         z4 = elevation_bottom_layer4[:, xx]
#         z = ds_mf['head'].isel(time=0, layer=layer).values[:, xx]
#         axes[layer].plot(x, z, ls='-', lw=1, color='black')
#         axes[layer].plot(x, z0, ls='-', lw=1, color='grey')
#         axes[layer].plot(x, z1, ls='-', lw=1, color='grey')
#         axes[layer].plot(x, z2, ls='-', lw=1, color='grey')
#         axes[layer].plot(x, z3, ls='-', lw=1, color='grey')
#         axes[layer].plot(x, z4, ls='-', lw=1, color='grey')
#         axes[layer].set_xlim(0, x[-1])
#         axes[layer].set_ylabel('[m a.s.l.]')
#     axes[-1].set_xlabel('Distance in N-S direction [m]')
#     fig.tight_layout()
#     file = base_path_figs / f"gw_head_y{xx}_cross_section_layer.png"
#     fig.savefig(file, dpi=300)
#     plt.close("all")


for layer in range(4):
    gw_depth = topography - ds_mf['head'].isel(time=0, layer=layer).values
    gw_depth[~mask] = np.nan
    mask1 = (gw_depth > 0)
    mask2 = (gw_depth < 0)
    gw_depth[mask1] = np.nan
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(gw_depth, extent=grid_extent, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater depth [m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"_gw_depth_steady_state_layer{i}.png"
    fig.savefig(file, dpi=300)
    plt.close("all")
    print(f"Layer {layer} (# GW table above surface): {np.sum(mask2)}")

    thickness = elevation_layers[layer] - elevation_layers[layer + 1]
    thickness[mask1] = np.nan
    thickness[~mask] = np.nan
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(thickness, extent=grid_extent, cmap='viridis', aspect='equal')
    plt.colorbar(label='thickness [m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"_thickness_layer{i}.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    hydraulic_conductivity = hydraulic_conductivities_layers[layer]
    hydraulic_conductivity[mask1] = np.nan
    hydraulic_conductivity[~mask] = np.nan
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(hydraulic_conductivity, extent=grid_extent, cmap='Oranges', aspect='equal')
    plt.colorbar(label=r'$k_f$ [m/day]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"_kf_layer{i}.png"
    fig.savefig(file, dpi=300)
    plt.close("all")