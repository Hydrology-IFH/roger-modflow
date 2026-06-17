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

dict_sfr_rno = {"EBNET": 872,
                "OBERAMBRINGEN": 4946,}


def xy_to_rowcol(x, y, x0, y0):
    """
    Convert map coordinates (x, y) to array indices (row, col) for a north-up raster.

    x0, y0: map coordinates of the upper-left corner of pixel (0, 0)

    Returns: (row, col) as integers (0-based)
    """
    col = math.floor((x - x0) / 50)
    row = math.floor((y0 - y) / 50)
    return row, col

@click.option("-mr", "--model-run", type=int, default=1806)
@click.command("main", short_help="Evaluate the transient simulation")
def main(model_run):
    base_path = Path(__file__).parent

    date_time = pd.date_range(start="2014-01-01", end="2023-12-31", freq="D")
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
    xcoords = ds_params.x.values + 25
    ycoords = ds_params.y.values - 25
    x0 = xcoords[0]
    y0 = ycoords[0]

    click.echo("Loading indirect recharge...")
    ll_indirect_recharge = []
    for year in years:
        output_file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / f"indirect_recharge_run{model_run}_year{year}.nc"
        # output_file = base_path_output / f"{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
        ds_indirect_recharge = xr.open_dataset(output_file, engine="h5netcdf")
        indirect_recharge_year = ds_indirect_recharge["indirect_recharge"].values
        ds_indirect_recharge.close()
        indirect_recharge_year = np.where(mask[np.newaxis, :, :], indirect_recharge_year, 0)
        ll_indirect_recharge.append(indirect_recharge_year)
    indirect_recharge = np.concatenate(ll_indirect_recharge, axis=0)

    # load SFR parameters
    reaches = pd.read_csv(base_path.parent / "input" / "sfr_packagedata_modified.csv", sep=";")

    # load fudge parameters
    path = base_path.parent / "fudge_parameters_modflow.csv"
    fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)

    # load observed groundwater heads (average values of the observation wells)
    click.echo("Loading observed groundwater depths...")
    path = base_path.parent / "observations" / "groundwater_observation_wells.gpkg"
    groundwater_observation_wells = gpd.read_file(path)
    groundwater_observation_wells.index = groundwater_observation_wells["station_id"]

    path = base_path.parent / "observations" / "groundwater_depth_time_series_filled.csv"
    observed_groundwater_depths = pd.read_csv(path, index_col=0, sep=";")
    observed_groundwater_depths.index = pd.to_datetime(observed_groundwater_depths.index, format="%Y-%m-%d")

    figure_dir = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures"
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)    

    # load the SFR output file
    output_file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / f"dmn_run_{model_run}_sfr.obs.csv"
    df_sfr_ = pd.read_csv(output_file, sep=",")
    df_sfr_ = df_sfr_.iloc[366:, :]
    date_time = pd.date_range(start="2014-01-01", end="2023-12-31", freq="D")
    df_sfr_.index = date_time

    streamflow_gauges = ["EBNET", "OBERAMBRINGEN", "FALKENSTEIG", "UNTERMUENSTERTAL", "OBERRIED"]
    for gauge in streamflow_gauges:
        # make lowercase
        _gauge = gauge.lower()
        rno = dict_sfr_rno[gauge]
        row = reaches.loc[reaches["rno"] == rno, "i"].values[0] - 1
        col = reaches.loc[reaches["rno"] == rno, "j"].values[0] - 1
        df_streamflow_sim = pd.DataFrame(index=date_time, columns=["sim"])
        df_streamflow_sim.loc[:, "sim"] = df_sfr_[f"{gauge}_FLOW"].values * (-1/86400) # convert from m3/d to m3/s
        df_indirect_recharge_sim = pd.DataFrame(index=date_time, columns=["sim"])
        df_indirect_recharge_sim.loc[:, "sim"] = indirect_recharge[:, row, col].flatten()
        df_stage_gw = pd.DataFrame(index=date_time, columns=["stage", "gw_head", "delta_head"])
        df_stage_gw.loc[:, "stage"] = df_sfr_[f"{gauge}_STAGE"].values
        df_stage_gw.loc[:, "gw_head"] = groundwater_heads[:, row, col].flatten()
        df_stage_gw.loc[:, "delta_head"] = df_stage_gw["gw_head"] - df_stage_gw["stage"]

        fig, axes = plt.subplots(figsize=(6, 2), nrows=1, ncols=1)
        axes.plot(df_stage_gw.index, df_stage_gw["delta_head"], linewidth=1, color="black")
        axes.set_xlim(df_stage_gw.index[0], df_stage_gw.index[-1])
        axes.set_xlabel("Zeit")
        axes.set_ylabel("$\Delta$ GW-OW [m]")
        fig.tight_layout()
        file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"ts_delta_gw_ow_{_gauge}_run{model_run}.png"
        fig.savefig(file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # plot simulated vs observed streamflow for the gauge and assign metrics to the title
        fig, axes = plt.subplots(figsize=(6, 4), nrows=2, ncols=1)
        axes[0].plot(df_indirect_recharge_sim.index, df_indirect_recharge_sim["sim"], label="Simuliert", linewidth=1, color="red")
        axes[0].set_xlim(df_indirect_recharge_sim.index[0], df_indirect_recharge_sim.index[-1])
        axes[0].set_ylabel("Indirekte GWN [m³/s]")
        axes[1].plot(df_streamflow_sim.index, df_streamflow_sim["sim"], label="Simuliert", linewidth=1, color="red")
        axes[1].set_xlim(df_streamflow_sim.index[0], df_streamflow_sim.index[-1])
        axes[1].set_xlabel("Zeit")
        axes[1].set_ylabel("Durchfluss [m³/s]")
        axes[1].set_yscale("log")
        fig.tight_layout()
        file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"ts_recharge_streamflow_{_gauge}_run{model_run}.png"
        fig.savefig(file, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # plot simulated vs observed streamflow for the gauge and assign metrics to the title
        fig, axes = plt.subplots(figsize=(4, 4))
        axes.scatter(df_indirect_recharge_sim["sim"], df_streamflow_sim["sim"], color="black", s=10)
        axes.set_xlabel("Indirekte GWN [m³/s]")
        axes.set_ylabel("Durchfluss [m³/s]")
        fig.tight_layout()
        file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"scatter_recharge_streamflow_{_gauge}_run{model_run}.png"
        fig.savefig(file, dpi=300, bbox_inches="tight")
        plt.close(fig)


    return

if __name__ == "__main__":
    main()
