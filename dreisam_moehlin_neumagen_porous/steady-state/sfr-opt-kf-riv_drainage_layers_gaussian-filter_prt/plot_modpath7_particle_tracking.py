from pathlib import Path
import flopy
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from flopy.plot.styles import styles
import geopandas as gpd
import contextily as ctx
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar
import numpy as np
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

# load catchment boundary
path = base_path.parent / "input" / "mask_catchment.gpkg"
catchment_boundary_porous = gpd.read_file(path)

def plot_map_view(ax, gwf):
    # plot map view of grid
    catchment_boundary_porous.boundary.plot(ax=ax, color='green', linewidth=0.5, zorder=1)
    mv = flopy.plot.PlotMapView(model=gwf, ax=ax)
    # mv.plot_grid(alpha=0.3)
    # mv.plot_ibound()  # inactive cells
    mv.plot_bc("WEL", plotAll=True)  # wells (red)
    mv.plot_bc("SFR", plotAll=True)
    mv.plot_bc("GHB", plotAll=True)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")
    ax.set_ylim(5.3e6, 5.327e6)
    ax.set_xlim(395000, 427000)

def plot_grid(gwf):
    with styles.USGSPlot():
        # setup the plot
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")

        # add plot features
        plot_map_view(ax, gwf)

        # plt.subplots_adjust(left=0.43)
        fig.tight_layout()
        fig.savefig(base_path / "figures" / "particle_tracking_grid.png", dpi=300)

def plot_pathlines_grid(ax, grid, hd, pl):
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    pc = mm.plot_array(hd, alpha=0.5)
    cb = plt.colorbar(pc, shrink=0.35, pad=0.1, label=r"GW-Hoehe [m]")
    mm.plot_pathline(pl, layer="all", lw=0.3, colors=["purple"])
    catchment_boundary_porous.boundary.plot(ax=ax, color='green', linewidth=1.5, zorder=1)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")
    ax.set_ylim(5.3e6, 5.327e6)
    ax.set_xlim(395000, 427000)
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})

def plot_pathlines_contours(ax, grid, hd, pl):
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    levels = np.arange(200, 425, 25)
    CS = mm.contour_array(hd, levels=levels, colors="black")
    ax.clabel(CS, fontsize=8)
    mm.plot_pathline(pl, layer="all", lw=0.3, colors=["purple"])
    catchment_boundary_porous.boundary.plot(ax=ax, color='green', linewidth=2)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.DE, crs=catchment_boundary_porous.crs)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")
    ax.set_ylim(5.3e6, 5.327e6)
    ax.set_xlim(395000, 427000)
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})

def plot_pathlines_grid_zoom(ax, grid, hd, pl):
    x1 = int((pl.x.min() - 500)/50)
    x2 = int((pl.x.max() + 500)/50)
    y2 = grid.shape[1] - int((pl.y.min() - 500)/50)
    y1 = grid.shape[1] - int((pl.y.max() + 500)/50)
    zmin = int(np.nanmin(hd[0, y1:y2, x1:x2]) - 1)
    zmax = int(np.nanmax(hd[0, y1:y2, x1:x2]) + 1)
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    pc = mm.plot_array(hd, alpha=0.5, vmin=zmin, vmax=zmax, cmap="viridis_r")
    cb = plt.colorbar(pc, shrink=0.35, pad=0.1, label=r"GW-Hoehe [m]")
    mm.plot_pathline(pl, layer="all", lw=0.3, colors=["purple"])
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})
    xx = grid.xoffset + pl["x"].values
    yy = grid.yoffset + pl["y"].values
    xmin = xx.min() - 500
    xmax = xx.max() + 500
    ymin = yy.min() - 500
    ymax = yy.max() + 500
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")

def plot_pathlines_contours_zoom(ax, grid, hd, pl):
    x1 = int((pl.x.min() - 500)/50)
    x2 = int((pl.x.max() + 500)/50)
    y2 = grid.shape[1] - int((pl.y.min() - 500)/50)
    y1 = grid.shape[1] - int((pl.y.max() + 500)/50)
    zmin = int(np.nanmin(hd[0, y1:y2, x1:x2]) - 1)
    zmax = int(np.nanmax(hd[0, y1:y2, x1:x2]) + 1)
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    levels = np.arange(zmin, zmax, 2.5)
    CS = mm.contour_array(hd, levels=levels, colors="black")
    ax.clabel(CS, fontsize=8)
    mm.plot_pathline(pl, layer="all", lw=0.3, colors=["purple"])
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.DE, crs=catchment_boundary_porous.crs, zoom_adjust=1)
    xx = grid.xoffset + pl["x"].values
    yy = grid.yoffset + pl["y"].values
    xmin = xx.min() - 500
    xmax = xx.max() + 500
    ymin = yy.min() - 500
    ymax = yy.max() + 500
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})


def plot_all_pathlines(grid, heads, mp7pl, well_name):
    with styles.USGSPlot():

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_grid(
            ax,
            grid,
            heads,
            mp7pl
        )
        fig.tight_layout()
        fig.savefig(base_path / "figures" / f"particle_tracking_grid_mp7_well{well_name}.png", dpi=300)

        fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_grid_zoom(ax1,
            grid,
            heads,
            mp7pl)
        fig1.tight_layout()
        fig1.savefig(base_path / "figures" / f"particle_tracking_grid_mp7_well{well_name}_zoom.png", dpi=300)
        
        fig2, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_contours(ax2,
            grid,
            heads,
            mp7pl)
        fig2.tight_layout()
        fig2.savefig(base_path / "figures" / f"particle_tracking_contours_mp7_well{well_name}.png", dpi=300)
        
        fig3, ax3 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_contours_zoom(ax3,
            grid,
            heads,
            mp7pl)
        fig3.tight_layout()
        fig3.savefig(base_path / "figures" / f"particle_tracking_contours_mp7_well{well_name}_zoom.png", dpi=300)


def plot_all(gwf, well_ids=[6], well_names=["A4"]):
    # extract grid
    grid = gwf.modelgrid

    # load mf6 gwf head results
    hf = flopy.utils.HeadFile(base_path / "output" / "dmn_run_1806.hds")
    hds = hf.get_data()
    cond_na = (hds > 1000) | (hds < 0)
    hds[cond_na] = np.nan

    plot_grid(gwf)

    for well_id, well_name in zip(well_ids, well_names):
        # load mp7 pathline results
        plf = flopy.utils.PathlineFile(base_path / "output" / f"well{well_id}_mp7.mppth")
        mp7_pl = pd.DataFrame(
            plf.get_destination_pathline_data(range(grid.nnodes), to_recarray=True)
        )

        plot_all_pathlines(grid, hds, mp7_pl, well_name)

# load the MODFLOW 6 model using pickle
with open(base_path / "output" / "dmn_run_1806.pkl", "rb") as f:
    gwfsim = pickle.load(f)
gwf = gwfsim.get_model("dmn_run_1806")

well_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
well_names = ["HU1", "HU2", "A4", "K5", "HU3", "A2", "K2", "B1", "A3", "S2", "B4", "C1"]
# well_ids = [5]
# well_names = ["A2"]
plot_all(gwf, well_ids=well_ids, well_names=well_names)