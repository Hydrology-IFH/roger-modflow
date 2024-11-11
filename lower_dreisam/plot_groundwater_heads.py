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
    'dx': int(roger_config['dx']*(res_modflow/roger_config['dx'])),
    'dy': int(roger_config['dy']*(res_modflow/roger_config['dx'])),
    'nx': int(roger_config['nx']/(res_modflow/roger_config['dx'])),
    'ny': int(roger_config['ny']/(res_modflow/roger_config['dx'])),
    'nz': 4,
}

file = base_path / "input" / "domain.grd"
domain = Raster.load(file)
mask = (domain.get_array(1)[:, :-1] == 1)
mask_basin = np.zeros_like(mask)
mask_basin[mask] = 1
grid_extent = (0, modflow_config['ny'] * modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])

# load topography
file = base_path / "input" / "elevation.grd"
layer_elevations = Raster.load(file)
topography = layer_elevations.get_array(1)[:, :-1]
elevation_bottom_layer1 = layer_elevations.get_array(2)[:, :-1]
topography[topography <= elevation_bottom_layer1] = elevation_bottom_layer1[topography <= elevation_bottom_layer1]
elevation_bottom_layer2 = layer_elevations.get_array(3)[:, :-1]
elevation_bottom_layer3 = layer_elevations.get_array(4)[:, :-1]
elevation_bottom_layer4 = layer_elevations.get_array(5)[:, :-1]
elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]


model_type = "transient"
base_path_figs = base_path / "figures" / model_type
sim = flopy.mf6.MFSimulation.load(
    sim_ws=base_path / "output" / model_type,
    exe_name="mf6",
    version="mf6",
    verbosity_level=0,
)

ml = sim.get_model("moehlin")

# load the netcdf file
output_file = base_path / "output" / model_type / "modflow_output.nc"
ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
ndays = ds_mf.sizes['time']


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
for t in range(1, ds_mf.sizes['time'], 30):
    # fig, axes = plt.subplots(figsize=(3, 3))
    # c = plt.contour(x, yr, h[t, :, :])
    # plt.clabel(c, fmt="%2.1f")
    # plt.axis("scaled")
    # axes.set_xlabel("distance in x-direction [m]")
    # axes.set_ylabel("distance in y-direction [m]")
    # fig.tight_layout()
    # path = base_path_figs / f"heads_{t}_transient_layer0.png"
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
    # path = base_path_figs / f"heads_contours_grid_{t}_transient_layer0.png"
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
    # plt.xlabel("distance in y-direction [m]")
    # plt.ylabel("distance in x-direction [m]")
    # fig.tight_layout()
    # path = base_path_figs / f"heads_grid_{t}_transient_layer0.png"
    # fig.savefig(path, dpi=300)
    # plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(ds_mf['head'].isel(time=t, layer=0).values, extent=grid_extent, vmin=100, vmax=300, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater head [m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"gw_head_{t}_transient_layer0.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(ds_mf['head'].isel(time=t, layer=1).values, extent=grid_extent, vmin=100, vmax=300, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater head [m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"gw_head_{t}_transient_layer1.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(ds_mf['head'].isel(time=t, layer=2).values, extent=grid_extent, vmin=100, vmax=300, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater head [m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"gw_head_{t}_transient_layer2.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(ds_mf['head'].isel(time=t, layer=3).values, extent=grid_extent, vmin=100, vmax=300, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater head [m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"gw_head_{t}_transient_layer3.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

model_type = "steady-state"
base_path_figs = base_path / "figures" / model_type
sim = flopy.mf6.MFSimulation.load(
    sim_ws=base_path / "output" / model_type,
    exe_name="mf6",
    version="mf6",
    verbosity_level=0,
)

ml = sim.get_model("moehlin")

# load the netcdf file
output_file = base_path / "output" / model_type / "modflow_output.nc"
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

ll_levels = [[224.5, 245, 250, 300, 400, 500, 600],
             [224.5, 245, 250, 300, 400, 500, 600],
             [224.5, 245, 250, 300, 400, 500, 600],
             [224.5, 245, 250, 300, 400, 500, 600]]

for layer in range(4):
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(ds_mf['head'].isel(time=0, layer=layer).values, extent=grid_extent, cmap='viridis', aspect='equal')
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
    plt.colorbar(label='groundwater thickness [m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"gw_depth_steady_state_layer{i}_grid.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    # fig, axes = plt.subplots(figsize=(4, 4))
    # y = np.arange(0, modflow_config['nx']*modflow_config['dx'], modflow_config['dx'])[::-1]
    # x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
    # X, Y = np.meshgrid(x, y)
    # Z = ds_mf['head'].isel(time=0, layer=layer).values
    # levels = ll_levels[layer]
    # CS = axes.contour(X, Y, Z, levels, colors='black')
    # axes.clabel(CS, inline=True, fontsize=8, colors='black')
    # axes.imshow(mask_basin, extent=grid_extent, cmap='Greys', alpha=0.25)
    # plt.xlabel('Distance in x-direction [m]')
    # plt.ylabel('Distance in y-direction [m]')
    # plt.title(f"Groundwater head of layer {layer + 1} [m a.s.l.]", fontsize=8)
    # plt.tight_layout()
    # i = layer + 1
    # file = base_path_figs / f"gw_head_steady_state_layer{i}_contour.png"
    # fig.savefig(file, dpi=300)
    # plt.close("all")

fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=True)
x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
for layer in range(4):
    z = ds_mf['head'].isel(time=0, layer=layer).values[82, :]
    axes[layer].plot(x, z, ls='-', lw=1, color='black')
    axes[layer].set_xlim(0, x[-1])
    axes[layer].set_ylabel('[m a.s.l.]')
axes[-1].set_xlabel('Distance in x-direction [m]')
fig.tight_layout()
file = base_path_figs / "gw_head_x_cross_section_layer.png"
fig.savefig(file, dpi=300)
plt.close("all")


for layer in range(4):
    grid_extent1 = (-2000, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])
    arr = ds_mf['head'].isel(time=0, layer=layer).values
    empty_arr = np.zeros((modflow_config['nx'], int(2000/modflow_config['dy'])))
    empty_arr[:, :] = np.nan
    arr1 = np.concatenate((empty_arr, arr), axis=1)
    fig, axes = plt.subplots(figsize=(5, 6))
    plt.scatter(-1350, 96*50, marker='x', s=20, c='black')  # approximate location of the observation well
    plt.scatter(150, 96*50, marker='x', s=20, c='red')  # approximate location of the observation well
    plt.imshow(arr1, extent=grid_extent1, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater head \n[m a.s.l.]', shrink=0.34)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"gw_head_steady_state_layer{i}_grid_.png"
    fig.savefig(file, dpi=300)
    plt.close("all")