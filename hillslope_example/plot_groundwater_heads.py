import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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

sim = flopy.mf6.MFSimulation.load(
    sim_ws=base_path / "output",
    exe_name="mf6",
    version="mf6",
    verbosity_level=0,
)

ml = sim.get_model("hillslope")

# load the netcdf file
output_file = base_path / "output" / "output.nc"
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
for t in range(1, ds_mf.sizes['time'], 7):
    fig, axes = plt.subplots(figsize=(3, 3))
    c = plt.contour(x, yr, h[t, :, :])
    # plot location of the well
    axes.scatter(x[6], yr[6], color="red", s=20, marker="x")  
    plt.clabel(c, fmt="%2.1f")
    plt.axis("scaled")
    axes.set_xlabel("distance in x-direction [m]")
    axes.set_ylabel("distance in y-direction [m]")
    fig.tight_layout()
    path = base_path_figs / f"heads_{t}.png"
    fig.savefig(path, dpi=300)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    modelmap = flopy.plot.PlotMapView(model=sim.gwf[0], ax=ax)

    # Then we can use the plot_grid() method to draw the grid
    # The return value for this function is a matplotlib LineCollection object,
    # which could be manipulated (or used) later if necessary.
    quadmesh = modelmap.plot_ibound(ibound=ibd)
    linecollection = modelmap.plot_grid()
    contours = modelmap.contour_array(h[t, :, :])
    ax.set_xlabel("distance in x-direction [m]")
    ax.set_ylabel("distance in y-direction [m]")
    fig.tight_layout()
    path = base_path_figs / f"heads_contrours_grid_{t}.png"
    fig.savefig(path, dpi=300)

    # Next we create an instance of the ModelMap class
    modelmap = flopy.plot.PlotMapView(model=sim.gwf[0], ax=ax)

    # Then we can use the plot_grid() method to draw the grid
    # The return value for this function is a matplotlib LineCollection object,
    # which could be manipulated (or used) later if necessary.
    quadmesh = modelmap.plot_ibound(ibound=ibd)
    linecollection = modelmap.plot_grid()
    pa = modelmap.plot_array(h[t, :, :])
    cb = plt.colorbar(pa, shrink=0.5, label="head [m.a.s.l.]")
    plt.xlabel("distance in x-direction [m]")
    plt.ylabel("distance in y-direction [m]")
    fig.tight_layout()
    path = base_path_figs / f"heads_grid_{t}.png"
    fig.savefig(path, dpi=300)