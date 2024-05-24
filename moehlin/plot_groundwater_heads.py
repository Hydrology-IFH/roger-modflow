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
    config = yaml.safe_load(file)

file = base_path / "input" / "domain.grd"
domain = Raster.load(file)
mask = (domain.get_array(1)[:, :-1] == 1)
mask_basin = np.zeros_like(mask)
mask_basin[mask] = 1
grid_extent = (0, config['ny']*config['dy'], 0, config['nx']*config['dx'])

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

grid_extent = (0, config['nx']*config['dx'], 0, config['ny']*config['dy'])

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
heads = ds_mf['head'].isel(layer=0).values
heads_avg = np.mean(heads, axis=0)
heads_p50 = np.median(heads, axis=0)
heads_p5 = np.percentile(heads, 5, axis=0)
heads_p95 = np.percentile(heads, 95, axis=0)
heads_min = np.min(heads, axis=0)
heads_max = np.max(heads, axis=0)

fig, axes = plt.subplots(figsize=(2, 4))
axes.plot(range(ndays), heads_avg, ls='--', label='average', lw=1.5)
axes.plot(range(ndays), heads_p50, ls='-', label='median', lw=2)
axes.plot(range(ndays), heads_p5, ls='-.', label='5th percentile', alpha=0.5)
axes.plot(range(ndays), heads_p95, ls='-.', label='95th percentile', alpha=0.5)
fig.tight_layout()
path = base_path_figs / "heads_time_series.png"
fig.savefig(path, dpi=300)

# plot the heads for all time steps
# for t in range(1, ds_mf.sizes['time']):
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
    # plt.xlabel("distance in y-direction [m]")
    # plt.ylabel("distance in x-direction [m]")
    # fig.tight_layout()
    # path = base_path_figs / f"heads_grid_{t}.png"
    # fig.savefig(path, dpi=300)
    # plt.close(fig)

    # fig, axes = plt.subplots(figsize=(4, 4))
    # plt.imshow(ds_mf['head'].isel(time=t, layer=0).values.T, extent=grid_extent, cmap='viridis', vmin=190, vmax=202, aspect='equal')
    # plt.colorbar(label='groundwater head [m a.s.l.]', shrink=0.5)
    # plt.grid(zorder=0)
    # plt.xlabel('Distance in y-direction [m]')
    # plt.ylabel('Distance in x-direction [m]')
    # plt.tight_layout()
    # file = base_path_figs / f"gw_head_{t}.png"
    # fig.savefig(file, dpi=300)
    # plt.close(fig)

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

ll_levels = [[242, 242.5, 243, 244],
             [242, 242.5, 243, 244],
             [242, 242.5, 243, 244],
             [130, 230, 242, 242.5, 243, 244]]

for layer in range(4):
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(ds_mf['head'].isel(time=0, layer=layer).values, extent=grid_extent, vmin=100, vmax=250, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater head [m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in y-direction [m]')
    plt.ylabel('Distance in x-direction [m]')
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
    plt.xlabel('Distance in y-direction [m]')
    plt.ylabel('Distance in x-direction [m]')
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"gw_thickness_steady_state_layer{i}_grid.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    y = np.arange(0, config['nx']*config['dx'], config['dx'])[::-1]
    x = np.arange(0, config['ny']*config['dy'], config['dy'])
    X, Y = np.meshgrid(x, y)
    Z = ds_mf['head'].isel(time=0, layer=layer).values
    levels = ll_levels[layer]
    CS = axes.contour(X, Y, Z, levels, colors='black')
    axes.clabel(CS, inline=True, fontsize=8, colors='black')
    axes.imshow(mask_basin, extent=grid_extent, cmap='Greys', alpha=0.25)
    plt.xlabel('Distance in y-direction [m]')
    plt.ylabel('Distance in x-direction [m]')
    plt.title(f"Groundwater head of layer {layer + 1} [m a.s.l.]", fontsize=8)
    plt.tight_layout()
    i = layer + 1
    file = base_path_figs / f"gw_head_steady_state_layer{i}_contour.png"
    fig.savefig(file, dpi=300)
    plt.close("all")