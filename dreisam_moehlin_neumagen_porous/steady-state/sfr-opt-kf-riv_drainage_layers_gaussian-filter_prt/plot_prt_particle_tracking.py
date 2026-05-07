from pathlib import Path
import flopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from flopy.plot.styles import styles
from matplotlib.collections import LineCollection
import yaml
import pickle

base_path = Path(__file__).parent

# load the groundwater extraction data
_groundwater_extraction = pd.read_csv(base_path.parent / "input" / "groundwater_extraction.csv", sep=";")
_groundwater_extraction["cell_y"] = _groundwater_extraction["cell_y"].values - 1
_groundwater_extraction["cell_x"] = _groundwater_extraction["cell_x"].values - 1
_groundwater_extraction["layer"] = _groundwater_extraction["layer"].values - 1

_wells_y = _groundwater_extraction["cell_y"].values.tolist()
_wells_x = _groundwater_extraction["cell_x"].values.tolist()
_wells_layer = _groundwater_extraction["layer"].values.tolist()
WEL_LOCS = []
for i in range(len(_wells_x)):
    WEL_LOCS.append((_wells_layer[i], _wells_y[i], _wells_x[i]))

file_config = base_path.parent / "config.yml"
with open(file_config, "r") as file:
    modflow_config = yaml.safe_load(file)

nrow = modflow_config["nx"]
ncol = modflow_config["ny"]
rowsize = modflow_config["dx"]
colsize = modflow_config["dy"]

def plot_map_view(ax, gwf):
    # plot map view of grid
    mv = flopy.plot.PlotMapView(model=gwf, ax=ax)
    mv.plot_grid(alpha=0.3)
    mv.plot_ibound()  # inactive cells
    mv.plot_bc("WEL", alpha=0.3)  # wells (red)

def plot_grid(gwf, title=None):
    with styles.USGSPlot():
        # setup the plot
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        if title is not None:
            styles.heading(ax=ax, heading=title)

        # add plot features
        plot_map_view(ax, gwf)

        # # add legend
        # ax.legend(
        #     handles=[
        #         mpl.patches.Patch(color="red", label="Wells", alpha=0.3),
        #     ],
        #     bbox_to_anchor=(-1.1, 0, 0.8, 0.1),
        # )

        # plt.subplots_adjust(left=0.43)
        fig.tight_layout()
        fig.savefig(base_path / "figures" / "particle_tracking_grid.png", dpi=300)

def plot_pathlines(ax, grid, hd, pl, title=None):
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    # mm.plot_grid(lw=0.5, alpha=0.5)
    # mm.plot_ibound()
    pc = mm.plot_array(hd, alpha=0.5)
    cb = plt.colorbar(pc, shrink=0.25, pad=0.1)
    cb.ax.set_xlabel(r"GW-Hoehe [m]")
    mm.plot_pathline(pl, layer="all", lw=0.3, colors=["black"])

def plot_pathlines_zoom(ax, grid, hd, pl, title=None):
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title)
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    # mm.plot_grid(lw=0.5, alpha=0.5)
    # mm.plot_ibound()
    pc = mm.plot_array(hd, alpha=0.5, vmin=195, vmax=210)
    cb = plt.colorbar(pc, shrink=0.25, pad=0.1)
    cb.ax.set_xlabel(r"GW-Hoehe [m]")
    mm.plot_pathline(pl, layer="all", lw=0.3, colors=["black"])
    pl["x"] = grid.xoffset + pl["x"].values
    pl["y"] = grid.yoffset + pl["y"].values
    xmin = pl.x.min() - 500
    xmax = pl.x.max() + 500
    ymin = pl.y.min() - 500
    ymax = pl.y.max() + 500
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)


def plot_all_pathlines(grid, heads, prtpl, well_id, title=None):
    with styles.USGSPlot():
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        if title is not None:
            styles.heading(ax=ax, heading=title)

        plot_pathlines(
            ax,
            grid,
            heads,
            prtpl
        )

        fig.tight_layout()
        fig.savefig(base_path / "figures" / f"particle_tracking_prt_well{well_id}.png", dpi=300)

def plot_all(gwf, well_ids=[6], well_names=["A4"]):
    # extract grid
    grid = gwf.modelgrid

    # load mf6 gwf head results
    hf = flopy.utils.HeadFile(base_path / "output" / "dmn_run_1806.hds")
    hds = hf.get_data()

    plot_grid(gwf, title="Model grid and boundary conditions")

    for well_id, well_name in zip(well_ids, well_names):
        # load mf6 prt pathline results
        prt_pl = pd.read_csv(base_path / "output"  / f"well{well_id}_prt.trk.csv")

        # plot_all_pathlines_prt(grid, hds, prt_pl, well_id, title="Head and pathlines")
        plot_all_pathlines(grid, hds, prt_pl, well_name, title="Head and pathlines")

# load the MODFLOW 6 model using pickle
with open(base_path / "output" / "dmn_run_1806.pkl", "rb") as f:
    gwfsim = pickle.load(f)
gwf = gwfsim.get_model("dmn_run_1806")

# well_ids = [2, 5, 7, 8, 10, 11]
# well_names = ["A4", "A2", "B1", "A3", "B4", "C1"]
well_ids = [2]
well_names = ["A4"]
plot_all(gwf, well_ids=well_ids, well_names=well_names)