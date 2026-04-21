from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import scipy as sp
import matplotlib.pyplot as plt
import click
import math
import os

dict_pseudowells_fissured = {"Au": (347, 280), # (x, y)
                           "Conventwald": (531, 135),
                           "Wagensteig": (649, 217),
                           "Falkensteig": (579, 303),
                           "Spielweg": (336, 453),
                           "Leimbach": (275, 373),
                           "Aubach": (312, 379),
                           "Eschbach": (553, 179),
                           "Hintereschbach": (554, 158),
                           "Molzhof": (451, 584),
                           "Zastler": (557, 352),
                           "Sankt_Wilhelm": (502, 416),
                           "Breitnau": (687, 313),
                           }

# dict_pseudowells_sfr = {"Falkensteig": (347, 280),
#                         "Ebnet": (430, 207),
#                         "Oberambringen": (191, 360),
#                         "Untermuenstertal": (218, 487),
#                         "Wiesneck": (578, 259),
#                         "SanktWilhelm": (479, 400),
#                         "Oberried": (507, 319),
#                         "Zastler": (557, 351)}

dict_pseudowells_sfr = {"Falkensteig": (347, 280),
                        "Ebnet": (430, 207),
                        "Oberambringen": (191, 360),
                        "Untermuenstertal": (218, 487),
                        "Oberried": (507, 319)}

def xy_to_rowcol(x, y, x0, y0):
    """
    Convert map coordinates (x, y) to array indices (row, col) for a north-up raster.

    x0, y0: map coordinates of the upper-left corner of pixel (0, 0)

    Returns: (row, col) as integers (0-based)
    """
    col = math.floor((x - x0) / 50)
    row = math.floor((y0 - y) / 50)
    return row, col

@click.option("-mr", "--model-run", type=int, default=944)
@click.command("main", short_help="Evaluate the transient simulation")
def main(model_run):
    base_path = Path(__file__).parent

    date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
    years = np.unique(date_time.year.values)

    # load the simulated groundwater depths
    click.echo("Loading simulated groundwater depths...")
    ll_groundwater_depths = []
    ll_groundwater_heads = []
    for year in years:
        output_file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / f"gw_depth_run{model_run}_year{year}.nc"
        ds_gw_depth_sim = xr.open_dataset(output_file, engine="h5netcdf")
        groundwater_depths_year = ds_gw_depth_sim["depth"].values[:, 1, :, :]
        ll_groundwater_depths.append(groundwater_depths_year)
        output_file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / f"gw_head_run{model_run}_year{year}.nc"
        ds_gw_head_sim = xr.open_dataset(output_file, engine="h5netcdf")
        groundwater_heads_year = ds_gw_head_sim["head"].values[:, 1, :, :]
        ll_groundwater_heads.append(groundwater_heads_year)
    groundwater_depths = np.concatenate(ll_groundwater_depths, axis=0)
    groundwater_heads = np.concatenate(ll_groundwater_heads, axis=0)

    # load topography
    click.echo("Loading topography...")
    path = base_path.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    topography = ds_params['elevations'].isel(z=0).values
    mask_schoenberg = (ds_params["mask_schoenberg"].values == 1)
    mask = np.isfinite(topography)
    mask = np.where(mask_schoenberg, False, mask)
    topography = np.where(mask, topography, np.nan)

    # load SFR parameters
    reaches = pd.read_csv(base_path.parent / "input" / "sfr_packagedata_modified.csv", sep=";")

    # load fudge parameters
    path = base_path.parent / "fudge_parameters_modflow.csv"
    fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)

    figure_dir = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures"
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    for station_id in dict_pseudowells_fissured.keys():
        # get row and column index based on ccordinate of the station
        click.echo(f"Evaluating station {station_id}...")
        col = dict_pseudowells_fissured[station_id][0]
        row = dict_pseudowells_fissured[station_id][1]
        simulated_depth = groundwater_depths[:, row, col]
        df_sim = pd.DataFrame({"simulated": simulated_depth})
        df_sim.index = date_time
        sim_vals = df_sim["simulated"].values

        # plot simulated time series of groundwater depths for the station
        fig, axes = plt.subplots(figsize=(6, 2))
        axes.plot(df_sim.index, df_sim["simulated"], label="Simuliert", linewidth=1, color="red")
        axes.set_xlim(df_sim.index[0], df_sim.index[-1])
        axes.set_ylim(0,)
        axes.invert_yaxis()
        axes.set_xlabel("Zeit")
        axes.set_ylabel("GWFA [m]")
        fig.tight_layout()
        file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"ts_gw_depths_{station_id}_run{model_run}.png"
        fig.savefig(file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return

if __name__ == "__main__":
    main()
