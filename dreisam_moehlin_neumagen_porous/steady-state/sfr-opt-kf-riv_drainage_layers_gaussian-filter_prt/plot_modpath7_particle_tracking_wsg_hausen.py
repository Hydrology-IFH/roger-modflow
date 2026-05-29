from pathlib import Path
import flopy
import matplotlib.pyplot as plt
import pandas as pd
from flopy.plot.styles import styles
import geopandas as gpd
import contextily as ctx
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar
from adjustText import adjust_text
import numpy as np
import pickle

base_path = Path(__file__).parent
base_path_external = Path("/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/steady-state/sfr-opt-kf-riv_drainage_layers_gaussian-filter_prt")

# load catchment boundary
path = base_path.parent / "input" / "mask_catchment.gpkg"
catchment_boundary_porous = gpd.read_file(path)

# load drinking water protection areas
path = base_path.parent / "input" / "wsg_hausen.gpkg"
wsg_hausen = gpd.read_file(path)

# load groundwater extraction wells
path = base_path.parent / "input" / "groundwater_extraction.gpkg"
gw_extraction_wells = gpd.read_file(path)

def plot_pathlines_grid(ax, grid, hd, pl, wells):
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    pc = mm.plot_array(hd, alpha=0.5)
    cb = plt.colorbar(pc, shrink=0.35, pad=0.1, label=r"GW-Hoehe [m]")
    well_names = pl["well_name"].unique()
    colors = plt.get_cmap("Purples", len(well_names) + 1)
    for i, well_name in enumerate(well_names):
        pl_subset = pl[pl["well_name"] == well_name]
        mm.plot_pathline(pl_subset, layer="all", lw=0.3, alpha=0.5, colors=[colors(i)], label=well_name)
    ax.scatter(wells['x-coordinate'], wells['y-coordinate'], c='orange', s=6, marker='^', label=wells["ID"].values, zorder=3)
    catchment_boundary_porous.boundary.plot(ax=ax, color='green', linewidth=1.5, zorder=1)
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")
    ax.set_ylim(5.3e6, 5.327e6)
    ax.set_xlim(395000, 427000)

def plot_pathlines_contours(ax, grid, hd, pl, wells):
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    levels = np.arange(200, 425, 25)
    CS = mm.contour_array(hd, levels=levels, colors="black")
    ax.clabel(CS, fontsize=8)
    well_names = pl["well_name"].unique()
    colors = plt.get_cmap("Purples", len(well_names) + 1)
    for i, well_name in enumerate(well_names):
        pl_subset = pl[pl["well_name"] == well_name]
        mm.plot_pathline(pl_subset, layer="all", lw=0.3, alpha=0.5, colors=[colors(i+1)])
    ax.scatter(wells['x-coordinate'], wells['y-coordinate'], c='orange', s=6, marker='^', label=wells["ID"].values, zorder=3)
    wsg_hausen.plot(ax=ax, color='none', edgecolor='blue', linewidth=1, hatch='////', label='WSG Hausen', alpha=0.5, zorder=1)
    catchment_boundary_porous.boundary.plot(ax=ax, color='green', linewidth=2)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")
    ax.set_ylim(5.3e6, 5.327e6)
    ax.set_xlim(395000, 427000)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.DE, crs=catchment_boundary_porous.crs)
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})

def plot_pathlines_grid_zoom(ax, grid, hd, pl, wells):
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    pc = mm.plot_array(hd, alpha=0.5, vmin=190, vmax=275)
    cb = plt.colorbar(pc, shrink=0.35, pad=0.1, label=r"GW-Hoehe [m]")
    well_names = pl["well_name"].unique()
    colors = plt.get_cmap("Purples", len(well_names) + 1)
    for i, well_name in enumerate(well_names):
        pl_subset = pl[pl["well_name"] == well_name]
        mm.plot_pathline(pl_subset, layer="all", lw=0.3, alpha=0.5, colors=[colors(i+1)])
    ax.scatter(wells['x-coordinate'], wells['y-coordinate'], c='orange', s=6, marker='^', label=wells["ID"].values, zorder=3)
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})
    xmin = wsg_hausen.bounds.minx[0] - 1000
    xmax = wsg_hausen.bounds.maxx[0] + 1000
    ymin = wsg_hausen.bounds.miny[0] - 1000
    ymax = wsg_hausen.bounds.maxy[0] + 1000
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")

    TEXTS = []
    bprops = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='none')
    for i in range(len(wells)):
        x = wells['x-coordinate'].iloc[i]
        y = wells['y-coordinate'].iloc[i]
        text = wells["ID"].iloc[i]
        TEXTS.append(ax.text(x, y, text, fontsize=7, bbox=bprops))

    adjust_text(
        TEXTS, 
        expand=(1, 1),
        ax=ax
    )

def plot_pathlines_contours_zoom(ax, grid, hd, pl, wells):
    ax.set_aspect("equal")
    mm = flopy.plot.PlotMapView(modelgrid=grid, ax=ax, layer=2)
    well_names = pl["well_name"].unique()
    colors = plt.get_cmap("Purples", len(well_names) + 1)
    for i, well_name in enumerate(well_names):
        pl_subset = pl[pl["well_name"] == well_name]
        mm.plot_pathline(pl_subset, layer="all", lw=0.3, alpha=0.5, colors=[colors(i+1)])
    levels = np.arange(190, 275, 5.0)
    CS = mm.contour_array(hd, levels=levels, colors="black")
    ax.clabel(CS, fontsize=8)
    ax.scatter(wells['x-coordinate'], wells['y-coordinate'], c='orange', s=6, marker='^', label=wells["ID"].values, zorder=3)
    wsg_hausen.plot(ax=ax, color='none', edgecolor='blue', linewidth=1, hatch='////', label='WSG Hausen', alpha=0.5, zorder=1)
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": catchment_boundary_porous.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": catchment_boundary_porous.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 9})
    xmin = wsg_hausen.bounds.minx[0] - 1000
    xmax = wsg_hausen.bounds.maxx[0] + 1000
    ymin = wsg_hausen.bounds.miny[0] - 1000
    ymax = wsg_hausen.bounds.maxy[0] + 1000
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.DE, crs=catchment_boundary_porous.crs)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")

    TEXTS = []
    bprops = dict(boxstyle='round', facecolor='white', alpha=0.75, edgecolor='none')
    for i in range(len(wells)):
        x = wells['x-coordinate'].iloc[i]
        y = wells['y-coordinate'].iloc[i]
        text = wells["ID"].iloc[i]
        TEXTS.append(ax.text(x, y, text, fontsize=7, bbox=bprops))

    adjust_text(
        TEXTS, 
        expand=(1, 1),
        ax=ax
    )


def plot_all_pathlines(grid, heads, mp7pl, wells, figures_dir=base_path_external / "figures"):
    with styles.USGSPlot():

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_grid(
            ax,
            grid,
            heads,
            mp7pl,
            wells
        )
        fig.tight_layout()
        fig.savefig(figures_dir / "particle_tracking_grid_mp7_wsg_hausen.png", dpi=300)

        fig1, ax1 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_grid_zoom(ax1,
            grid,
            heads,
            mp7pl,
            wells)
        fig1.tight_layout()
        fig1.savefig(figures_dir / "particle_tracking_grid_mp7_wsg_hausen_zoom.png", dpi=300)
        
        fig2, ax2 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_contours(ax2,
            grid,
            heads,
            mp7pl,
            wells)
        fig2.tight_layout()
        fig2.savefig(figures_dir / "particle_tracking_contours_mp7_wsg_hausen.png", dpi=300)
        
        fig3, ax3 = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
        plot_pathlines_contours_zoom(ax3,
            grid,
            heads,
            mp7pl,
            wells)
        fig3.tight_layout()
        fig3.savefig(figures_dir / "particle_tracking_contours_mp7_wsg_hausen_zoom.png", dpi=300)


def plot_all(gwf, well_ids=[6], well_names=["A4"]):
    # extract grid
    grid = gwf.modelgrid

    # load mf6 gwf head results
    hf = flopy.utils.HeadFile(base_path_external / "output" / "dmn_run_1806.hds")
    hds = hf.get_data()
    cond_na = (hds > 1000) | (hds < 0)
    hds[cond_na] = np.nan

    release_scenarios = ["near_surface", "pump_installation_depth", "deep"]
    for release_scenario in release_scenarios:
        ll_pathlines = []
        for well_id, well_name in zip(well_ids, well_names):
            # load mp7 pathline results
            plf = flopy.utils.PathlineFile(base_path_external / "output" / f"{release_scenario}" / f"well_{well_id}" / f"well{well_id}_mp7.mppth")
            mp7_pl = pd.DataFrame(
                plf.get_destination_pathline_data(range(grid.nnodes), to_recarray=True)
            )
            mp7_pl["well_id"] = well_id
            mp7_pl["well_name"] = well_name
            cond_time = (mp7_pl["time"] < (365.25 * 5))
            mp7_pl = mp7_pl[cond_time]
            ll_pathlines.append(mp7_pl)

        pl = pd.concat(ll_pathlines, ignore_index=True)

        wells = gw_extraction_wells[gw_extraction_wells["ID"].isin(well_names)]

        figures_dir = base_path_external / "figures" / f"{release_scenario}"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_all_pathlines(grid, hds, pl, wells, figures_dir=figures_dir)

# load the MODFLOW 6 model using pickle
with open(base_path_external / "output" / "dmn_run_1806.pkl", "rb") as f:
    gwfsim = pickle.load(f)
gwf = gwfsim.get_model("dmn_run_1806")

# well_ids = [2, 5, 7, 8, 10, 11]
# well_names = ["A4", "A2", "B1", "A3", "B4", "C1"]
well_ids = [5, 7, 10]
well_names = ["A2", "B1", "B4"]
plot_all(gwf, well_ids=well_ids, well_names=well_names)