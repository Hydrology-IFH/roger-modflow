from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import scipy as sp
import rasterio
import matplotlib.pyplot as plt
import scipy
import click
import math
import os
import yaml

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

    date_time = pd.date_range(start="2013-01-01", end="2017-12-31", freq="D")
    years = np.unique(date_time.year.values)
    timesteps = np.arange(len(date_time))

    # load the simulated groundwater depths
    click.echo("Loading simulated groundwater depths...")
    ll_groundwater_depths = []
    for year in years:
        output_file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / f"gw_depth_run{model_run}_year{year}.nc"
        ds_gw_depth_sim = xr.open_dataset(output_file, engine="h5netcdf")
        groundwater_depths_year = ds_gw_depth_sim["depth"].values[:, 1, :, :]
        ll_groundwater_depths.append(groundwater_depths_year)
    groundwater_depths = np.concatenate(ll_groundwater_depths, axis=0)

    # load topography
    click.echo("Loading topography...")
    path = base_path.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    xcoords = ds_params["x"].values
    ycoords = ds_params["y"].values
    x0 = xcoords[0] - 25
    y0 = ycoords[0] + 25

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

    ll_observed_depths = []
    ll_simulated_depths = []

    df_metrics = pd.DataFrame(index=observed_groundwater_depths.columns, columns=["NSE", "MAE", "r"])

    for station_id in observed_groundwater_depths.columns:
        # get row and column index based on ccordinate of the station
        _station_id = str(station_id).replace("_", "/")
        click.echo(f"Evaluating station {station_id}({_station_id})...")
        xcoord = groundwater_observation_wells.loc[_station_id, "xcoord"]
        ycoord = groundwater_observation_wells.loc[_station_id, "ycoord"]
        # check if the station is within the bounds of the model grid
        if xcoord >= xcoords[0] and xcoord <= xcoords[-1] and ycoord >= ycoords[-1] and ycoord <= ycoords[0]:
            row, col = xy_to_rowcol(xcoord, ycoord, x0, y0)
            simulated_depth = groundwater_depths[:, row, col]
            observed_depth = observed_groundwater_depths[station_id].values
            df_sim = pd.DataFrame({"simulated": simulated_depth})
            df_sim.index = date_time
            df_obs = pd.DataFrame({"observed": observed_depth})
            df_obs.index = observed_groundwater_depths.index
            df_sim_obs = df_sim.join(df_obs, how="inner")
            df_sim_obs = df_sim_obs.dropna()
            if len(df_sim_obs) > 24:
                ll_observed_depths.append(df_sim_obs["observed"].values)
                ll_simulated_depths.append(df_sim_obs["simulated"].values)
                # calculate metrics
                sim_vals = df_sim_obs["simulated"].values
                obs_vals = df_sim_obs["observed"].values
                nse_depth = 1.0 - np.sum((obs_vals - sim_vals) ** 2) / np.sum((obs_vals - np.mean(obs_vals)) ** 2)
                mae_depth = np.mean(np.abs(obs_vals - sim_vals))
                r_rank = sp.stats.spearmanr(sim_vals, obs_vals)[0]
                df_metrics.loc[station_id, "NSE"] = nse_depth
                df_metrics.loc[station_id, "MAE"] = mae_depth
                df_metrics.loc[station_id, "r"] = r_rank

                # plot simulated vs observed groundwater depths for the station and assign metrics to the title
                fig, axes = plt.subplots(figsize=(4, 4))
                axes.scatter(df_sim_obs["observed"], df_sim_obs["simulated"], alpha=0.8)
                axes.plot([0, np.max(df_sim_obs["observed"])], [0, np.max(df_sim_obs["observed"])], "k--")
                axes.set_xlabel("Gemessener GWFA [m]")
                axes.set_ylabel("Simulierter GWFA [m]")
                axes.set_xlim(0, np.max(df_sim_obs["observed"]))
                axes.set_ylim(0, np.max(df_sim_obs["observed"]))
                axes.set_title(f"{station_id}\nNSE: {nse_depth:.2f}, MAE: {mae_depth:.2f} m, r: {r_rank:.2f}")
                fig.tight_layout()
                file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"scatter_gw_depths_{station_id}_run{model_run}.png"
                fig.savefig(file, dpi=300, bbox_inches="tight")
                plt.close(fig)

                # compare simulated and observed groundwater depths in a time series plot
                fig, axes = plt.subplots(figsize=(6, 2))
                axes.plot(df_sim_obs.index, df_sim_obs["observed"], label="Gemessen", linewidth=1.2, color="blue")
                axes.plot(df_sim_obs.index, df_sim_obs["simulated"], label="Simuliert", linewidth=1, color="red")
                axes.set_xlabel("Zeit")
                axes.set_ylabel("GWFA [m]")
                fig.tight_layout()
                file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"ts_gw_depths_{station_id}_run{model_run}.png"
                fig.savefig(file, dpi=300, bbox_inches="tight")
                plt.close(fig)

    # drop rows with missing metrics
    df_metrics = df_metrics.dropna(subset=["NSE", "MAE", "r"])

    # calculate overall metrics
    df_metrics.loc["avg", "NSE"] = np.mean(df_metrics["NSE"].astype(float))
    df_metrics.loc["avg", "MAE"] = np.mean(df_metrics["MAE"].astype(float))
    df_metrics.loc["avg", "r"] = np.mean(df_metrics["r"].astype(float))
    df_metrics.loc["median", "NSE"] = np.median(df_metrics["NSE"].astype(float))
    df_metrics.loc["median", "MAE"] = np.median(df_metrics["MAE"].astype(float))
    df_metrics.loc["median", "r"] = np.median(df_metrics["r"].astype(float))
    df_metrics.loc["std", "NSE"] = np.std(df_metrics["NSE"].astype(float))
    df_metrics.loc["std", "MAE"] = np.std(df_metrics["MAE"].astype(float))
    df_metrics.loc["std", "r"] = np.std(df_metrics["r"].astype(float))
    df_metrics.loc["min", "NSE"] = np.min(df_metrics["NSE"].astype(float))
    df_metrics.loc["min", "MAE"] = np.min(df_metrics["MAE"].astype(float))
    df_metrics.loc["min", "r"] = np.min(df_metrics["r"].astype(float))
    df_metrics.loc["max", "NSE"] = np.max(df_metrics["NSE"].astype(float))
    df_metrics.loc["max", "MAE"] = np.max(df_metrics["MAE"].astype(float))
    df_metrics.loc["max", "r"] = np.max(df_metrics["r"].astype(float))

    # write metrics to csv
    click.echo("Writing evaluation metrics to csv...")
    file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"evaluation_metrics_run{model_run}.csv"
    df_metrics.to_csv(file)

    # make scatter plot of simulated vs observed groundwater depths
    click.echo("Making scatter plot of simulated vs observed groundwater depths...")
    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(np.concatenate(ll_observed_depths), np.concatenate(ll_simulated_depths), alpha=0.8, color="black", s=10)
    axes.plot([0, np.max(np.concatenate(ll_observed_depths))], [0, np.max(np.concatenate(ll_observed_depths))], "k--")
    axes.set_xlabel("Gemessener GWFA [m]")
    axes.set_ylabel("Simulierter GWFA [m]")
    axes.set_xlim(0, np.max(np.concatenate(ll_observed_depths)))
    axes.set_ylim(0, np.max(np.concatenate(ll_simulated_depths)))
    fig.tight_layout()
    file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / f"scatter_gw_depths_run{model_run}.png"
    fig.savefig(file, dpi=300, bbox_inches="tight")
    return

if __name__ == "__main__":
    main()
