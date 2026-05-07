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

# load drinking water protection areas
path = base_path.parent / "input" / "wsg_hausen.gpkg"
wsg_hausen = gpd.read_file(path)

def plot_map_view(ax, gwf):
    # plot map view of grid
    catchment_boundary_porous.boundary.plot(ax=ax, color='green', linewidth=0.5, zorder=1)
    mv = flopy.plot.PlotMapView(model=gwf, ax=ax)
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
    well_names = pl["well_name"].unique()
    colors = plt.get_cmap("Purples", len(well_names) + 1)
    for i, well_name in enumerate(well_names):
        pl_subset = pl[pl["well_name"] == well_name]
        mm.plot_pathline(pl_subset, layer="all", lw=0.3, alpha=0.5, colors=[colors(i)], label=well_name)
    catchment_boundary_porous.boundary.plot(ax=ax, color='green', linewidth=1.5, zorder=1)
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")
    ax.set_ylim(5.3e6, 5.327e6)
    ax.set_xlim(395000, 427000)

def plot_pathlines_contours(ax, grid, hd, pl):
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    levels = np.arange(200, 425, 25)
    CS = mm.contour_array(hd, levels=levels, colors="black")
    ax.clabel(CS, fontsize=8)
    well_names = pl["well_name"].unique()
    colors = plt.get_cmap("Purples", len(well_names) + 1)
    for i, well_name in enumerate(well_names):
        pl_subset = pl[pl["well_name"] == well_name]
        mm.plot_pathline(pl_subset, layer="all", lw=0.3, alpha=0.5, colors=[colors(i+1)], label=well_name)
    wsg_hausen.plot(ax=ax, color='none', edgecolor='blue', linewidth=1, hatch='////', label='WSG Hausen', alpha=0.5)
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
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    pc = mm.plot_array(hd, alpha=0.5, vmin=190, vmax=275)
    cb = plt.colorbar(pc, shrink=0.35, pad=0.1, label=r"GW-Hoehe [m]")
    well_names = pl["well_name"].unique()
    colors = plt.get_cmap("Purples", len(well_names) + 1)
    for i, well_name in enumerate(well_names):
        pl_subset = pl[pl["well_name"] == well_name]
        mm.plot_pathline(pl_subset, layer="all", lw=0.3, alpha=0.5, colors=[colors(i+1)], label=well_name)
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})
    xx = grid.xoffset + pl["x"].values
    yy = grid.yoffset + pl["y"].values
    xmin = xx.min() - 1000
    xmax = xx.max() + 2000
    ymin = yy.min() - 1000
    ymax = yy.max() + 1000
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")

def plot_pathlines_contours_zoom(ax, grid, hd, pl):
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    levels = np.arange(190, 275, 5.0)
    CS = mm.contour_array(hd, levels=levels, colors="black")
    ax.clabel(CS, fontsize=8)
    well_names = pl["well_name"].unique()
    colors = plt.get_cmap("Purples", len(well_names) + 1)
    for i, well_name in enumerate(well_names):
        pl_subset = pl[pl["well_name"] == well_name]
        mm.plot_pathline(pl_subset, layer="all", lw=0.3, alpha=0.5, colors=[colors(i+1)], label=well_name)
    wsg_hausen.plot(ax=ax, color='none', edgecolor='blue', linewidth=1, hatch='////', label='WSG Hausen', alpha=0.5, zorder=1)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.DE, crs=catchment_boundary_porous.crs, zoom_adjust=1)
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})
    xx = grid.xoffset + pl["x"].values
    yy = grid.yoffset + pl["y"].values
    xmin = xx.min() - 1000
    xmax = xx.max() + 2000
    ymin = yy.min() - 1000
    ymax = yy.max() + 1000
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")


def plot_all_pathlines(grid, heads, mp7pl):
    with styles.USGSPlot():

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_grid(
            ax,
            grid,
            heads,
            mp7pl
        )
        fig.tight_layout()
        fig.savefig(base_path / "figures" / "particle_tracking_grid_mp7_wsg_hausen.png", dpi=300)

        fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_grid_zoom(ax1,
            grid,
            heads,
            mp7pl)
        fig1.tight_layout()
        fig1.savefig(base_path / "figures" / "particle_tracking_grid_mp7_wsg_hausen_zoom.png", dpi=300)
        
        fig2, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_contours(ax2,
            grid,
            heads,
            mp7pl)
        fig2.tight_layout()
        fig2.savefig(base_path / "figures" / "particle_tracking_contours_mp7_wsg_hausen.png", dpi=300)
        
        fig3, ax3 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_contours_zoom(ax3,
            grid,
            heads,
            mp7pl)
        fig3.tight_layout()
        fig3.savefig(base_path / "figures" / "particle_tracking_contours_mp7_wsg_hausen_zoom.png", dpi=300)


def plot_all(gwf, well_ids=[6], well_names=["A4"]):
    # extract grid
    grid = gwf.modelgrid

    # load mf6 gwf head results
    hf = flopy.utils.HeadFile(base_path / "output" / "dmn_run_1806.hds")
    hds = hf.get_data()
    cond_na = (hds > 1000) | (hds < 0)
    hds[cond_na] = np.nan

    plot_grid(gwf)

    ll_pathlines = []
    for well_id, well_name in zip(well_ids, well_names):
        # load mp7 pathline results
        plf = flopy.utils.PathlineFile(base_path / "output" / f"well{well_id}_mp7.mppth")
        mp7_pl = pd.DataFrame(
            plf.get_destination_pathline_data(range(grid.nnodes), to_recarray=True)
        )
        mp7_pl["well_id"] = well_id
        mp7_pl["well_name"] = well_name
        ll_pathlines.append(mp7_pl)

    pl = pd.concat(ll_pathlines, ignore_index=True)

    plot_all_pathlines(grid, hds, pl)

# load the MODFLOW 6 model using pickle
with open(base_path / "output" / "dmn_run_1806.pkl", "rb") as f:
    gwfsim = pickle.load(f)
gwf = gwfsim.get_model("dmn_run_1806")

# well_ids = [2, 5, 7, 8, 10, 11]
# well_names = ["A4", "A2", "B1", "A3", "B4", "C1"]
well_ids = [5, 7, 10]
well_names = ["A2", "B1", "B4"]
plot_all(gwf, well_ids=well_ids, well_names=well_names)