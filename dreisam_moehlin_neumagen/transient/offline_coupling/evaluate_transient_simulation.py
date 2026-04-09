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

dict_obs_wells_fissured = {"Au": (347, 280),
                           "Conventwalde": (531, 135),
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
    topography = ds_params['elevations'].isel(z=0).values
    mask_schoenberg = (ds_params["mask_schoenberg"].values == 1)
    mask = np.isfinite(topography)
    mask = np.where(mask_schoenberg, False, mask)
    topography = np.where(mask, topography, np.nan)
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


    for station_id in dict_obs_wells_fissured.keys():
        # get row and column index based on ccordinate of the station
        click.echo(f"Evaluating station {station_id}...")
        col = dict_obs_wells_fissured[station_id][0]
        row = dict_obs_wells_fissured[station_id][1]
        simulated_depth = groundwater_depths[:, row, col]
        df_sim = pd.DataFrame({"simulated": simulated_depth})
        df_sim.index = date_time
        sim_vals = df_sim_obs["simulated"].values

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

    ll_observed_depths = []
    ll_simulated_depths = []

    df_metrics_gw = pd.DataFrame(index=observed_groundwater_depths.columns, columns=["NSE", "MAE", "r"])

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
                df_metrics_gw.loc[station_id, "xcoord"] = xcoord
                df_metrics_gw.loc[station_id, "ycoord"] = ycoord
                df_metrics_gw.loc[station_id, "NSE"] = nse_depth
                df_metrics_gw.loc[station_id, "MAE"] = mae_depth
                df_metrics_gw.loc[station_id, "r"] = r_rank

                # plot simulated vs observed groundwater depths for the station and assign metrics to the title
                fig, axes = plt.subplots(figsize=(4, 4))
                axes.scatter(df_sim_obs["observed"], df_sim_obs["simulated"], alpha=0.8, color="black", s=5)
                axes.plot([0, 30], [0, 30], "k--")
                axes.set_xlabel("Gemessener GWFA [m]")
                axes.set_ylabel("Simulierter GWFA [m]")
                axes.set_xlim(0, 30)
                axes.set_ylim(0, 30)
                axes.set_title(f"{station_id}\nNSE: {nse_depth:.2f}, MAE: {mae_depth:.2f} m, r: {r_rank:.2f}")
                fig.tight_layout()
                file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"scatter_gw_depths_{station_id}_run{model_run}.png"
                fig.savefig(file, dpi=300, bbox_inches="tight")
                plt.close(fig)

                # compare simulated and observed groundwater depths in a time series plot
                fig, axes = plt.subplots(figsize=(6, 2))
                axes.plot(df_sim_obs.index, df_sim_obs["observed"], label="Gemessen", linewidth=1.2, color="blue")
                axes.plot(df_sim_obs.index, df_sim_obs["simulated"], label="Simuliert", linewidth=1, color="red")
                axes.set_xlim(df_sim_obs.index[0], df_sim_obs.index[-1])
                axes.set_ylim(0,)
                axes.invert_yaxis()
                axes.set_xlabel("Zeit")
                axes.set_ylabel("GWFA [m]")
                fig.tight_layout()
                file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"ts_gw_depths_{station_id}_run{model_run}.png"
                fig.savefig(file, dpi=300, bbox_inches="tight")
                plt.close(fig)

    # drop rows with missing metrics
    df_metrics_gw = df_metrics_gw.dropna(subset=["NSE", "MAE", "r"])

    # calculate overall metrics
    df_metrics_gw.loc["avg", "NSE"] = np.mean(df_metrics_gw["NSE"].astype(float))
    df_metrics_gw.loc["avg", "MAE"] = np.mean(df_metrics_gw["MAE"].astype(float))
    df_metrics_gw.loc["avg", "r"] = np.mean(df_metrics_gw["r"].astype(float))
    df_metrics_gw.loc["median", "NSE"] = np.median(df_metrics_gw["NSE"].astype(float))
    df_metrics_gw.loc["median", "MAE"] = np.median(df_metrics_gw["MAE"].astype(float))
    df_metrics_gw.loc["median", "r"] = np.median(df_metrics_gw["r"].astype(float))
    df_metrics_gw.loc["std", "NSE"] = np.std(df_metrics_gw["NSE"].astype(float))
    df_metrics_gw.loc["std", "MAE"] = np.std(df_metrics_gw["MAE"].astype(float))
    df_metrics_gw.loc["std", "r"] = np.std(df_metrics_gw["r"].astype(float))
    df_metrics_gw.loc["min", "NSE"] = np.min(df_metrics_gw["NSE"].astype(float))
    df_metrics_gw.loc["min", "MAE"] = np.min(df_metrics_gw["MAE"].astype(float))
    df_metrics_gw.loc["min", "r"] = np.min(df_metrics_gw["r"].astype(float))
    df_metrics_gw.loc["max", "NSE"] = np.max(df_metrics_gw["NSE"].astype(float))
    df_metrics_gw.loc["max", "MAE"] = np.max(df_metrics_gw["MAE"].astype(float))
    df_metrics_gw.loc["max", "r"] = np.max(df_metrics_gw["r"].astype(float))

    # write metrics to csv
    click.echo("Writing evaluation metrics to csv...")
    file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"evaluation_metrics_run{model_run}.csv"
    df_metrics_gw.to_csv(file, sep=";")

    # make scatter plot of simulated vs observed groundwater depths
    click.echo("Making scatter plot of simulated vs observed groundwater depths...")
    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(np.concatenate(ll_observed_depths), np.concatenate(ll_simulated_depths), alpha=0.5, color="black", s=2.5)
    axes.plot([0, 30], [0, 30], "k--", alpha=0.8)
    axes.set_xlabel("Gemessener GWFA [m]")
    axes.set_ylabel("Simulierter GWFA [m]")
    axes.set_xlim(0, 30)
    axes.set_ylim(0, 30)
    axes.set_title(f"MAE: {df_metrics_gw.loc['avg', 'MAE']:.2f} m, r: {df_metrics_gw.loc['avg', 'r']:.2f}")
    fig.tight_layout()
    file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"scatter_gw_depths_run{model_run}.png"
    fig.savefig(file, dpi=300, bbox_inches="tight")
    
    cm = plt.get_cmap('Greens_r')
    cm.set_bad(color='grey')
    grid_extent = (xcoords[0], xcoords[-1], ycoords[-1], ycoords[0])
    wells_obs_x = df_metrics_gw["xcoord"].values.astype(float)
    wells_obs_y = df_metrics_gw["ycoord"].values.astype(float)
    metric_values = df_metrics_gw["MAE"].values.astype(float)
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.scatter(wells_obs_x, wells_obs_y, c=metric_values, s=5, cmap=cm, vmin=0, vmax=3)
    plt.colorbar(label='MAE [m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('X-Koordinate')
    plt.ylabel('Y-Koordinate')
    fig.tight_layout()
    file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"map_mae_run{model_run}.png"
    fig.savefig(file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    cm = plt.get_cmap('Greens')
    cm.set_bad(color='grey')
    grid_extent = (xcoords[0], xcoords[-1], ycoords[-1], ycoords[0])
    wells_obs_x = df_metrics_gw["xcoord"].values.astype(float)
    wells_obs_y = df_metrics_gw["ycoord"].values.astype(float)
    metric_values = df_metrics_gw["r"].values.astype(float)
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.scatter(wells_obs_x, wells_obs_y, c=metric_values, s=5, cmap=cm, vmin=0, vmax=1)
    plt.colorbar(label='r', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('X-Koordinate')
    plt.ylabel('Y-Koordinate')
    fig.tight_layout()
    file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"map_r_run{model_run}.png"
    fig.savefig(file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    

    # load the SFR output file
    output_file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / f"dmn_run_{model_run}_sfr.obs.csv"
    df_sfr_ = pd.read_csv(output_file, sep=",")
    date_time = pd.date_range(start="2013-01-01", end="2024-01-01", freq="D")
    df_sfr_.index = date_time

    streamflow_gauges = ["EBNET", "OBERAMBRINGEN", "FALKENSTEIG", "UNTERMUENSTERTAL", "OBERRIED"]
    df_metrics_sfr = pd.DataFrame(index=streamflow_gauges, columns=["NSE", "MAE", "r"])
    for gauge in streamflow_gauges:
        # make lowercase
        _gauge = gauge.lower()
        file = base_path.parent / "observations" / f"streamflow_{_gauge}.csv"
        df_streamflow_obs = pd.read_csv(file, index_col=0, sep=";", skiprows=1)
        df_streamflow_obs.columns = ["obs"]
        df_streamflow_obs.index = pd.to_datetime(df_streamflow_obs.index, format="%Y-%m-%d")
        df_streamflow_sim = pd.DataFrame(index=date_time, columns=["sim"])
        df_streamflow_sim.loc[:, "sim"] = df_sfr_[f"{gauge}_FLOW"].values * (-1/86400) # convert from m3/d to m3/s
        # join observed and simulated streamflow
        df_streamflow_sim_obs = df_streamflow_sim.join(df_streamflow_obs)
        df_streamflow_sim_obs = df_streamflow_sim_obs.dropna()
        sim_vals = df_streamflow_sim_obs["sim"].values
        obs_vals = df_streamflow_sim_obs["obs"].values
        nse_sfr = 1.0 - np.sum((obs_vals - sim_vals) ** 2) / np.sum((obs_vals - np.mean(obs_vals)) ** 2)
        mae_sfr = np.mean(np.abs(obs_vals - sim_vals))
        r_rank = sp.stats.spearmanr(sim_vals, obs_vals)[0]
        df_metrics_sfr.loc[gauge, "NSE"] = nse_sfr
        df_metrics_sfr.loc[gauge, "MAE"] = mae_sfr
        df_metrics_sfr.loc[gauge, "r"] = r_rank

        # plot simulated vs observed streamflow for the gauge and assign metrics to the title
        fig, axes = plt.subplots(figsize=(6, 2))
        axes.plot(df_streamflow_sim_obs.index, df_streamflow_sim_obs["obs"], label="Gemessen", linewidth=1.2, color="blue")
        axes.plot(df_streamflow_sim_obs.index, df_streamflow_sim_obs["sim"], label="Simuliert", linewidth=1, color="red")
        axes.set_xlim(df_streamflow_sim_obs.index[0], df_streamflow_sim_obs.index[-1])
        axes.set_title(f"MAE: {df_metrics_sfr.loc[gauge, 'MAE']:.2f} m³/s, r: {df_metrics_sfr.loc[gauge, 'r']:.2f}")
        axes.set_xlabel("Zeit")
        axes.set_ylabel("Durchfluss [m³/s]")
        axes.set_yscale("log")
        fig.tight_layout()
        file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"ts_streamflow_{_gauge}_run{model_run}.png"
        fig.savefig(file, dpi=300, bbox_inches="tight")
        plt.close(fig)

    streamflow_gauges = ["E2"]
    for gauge in streamflow_gauges:
        # make lowercase
        file = base_path.parent / "observations" / f"{gauge}_streamflow.csv"
        df_streamflow_obs = pd.read_csv(file, index_col=0, sep=";", skiprows=1)
        df_streamflow_obs.columns = ["obs"]
        df_streamflow_obs.index = pd.to_datetime(df_streamflow_obs.index, format="%Y-%m-%d")
        _df_streamflow_sim = pd.DataFrame(index=date_time, columns=["sim"])
        _df_streamflow_sim.loc[:, "sim"] = df_sfr_[f"{gauge}_FLOW"].values * (-1/86400) # convert from m3/d to m3/s
        date_time_2020_2021 = pd.date_range(start="2020-01-01", end="2021-12-31", freq="D")
        df_streamflow_sim = pd.DataFrame(index=date_time_2020_2021)
        df_streamflow_sim = df_streamflow_sim.join(_df_streamflow_sim)
        # join observed and simulated streamflow
        df_streamflow_sim_obs = df_streamflow_sim.join(df_streamflow_obs)

        # plot simulated vs observed streamflow for the gauge and assign metrics to the title
        fig, axes = plt.subplots(figsize=(6, 2))
        axes.plot(df_streamflow_sim_obs.index, df_streamflow_sim_obs["obs"], label="Gemessen", linewidth=1.2, color="blue")
        axes.plot(df_streamflow_sim_obs.index, df_streamflow_sim_obs["sim"], label="Simuliert", linewidth=1, color="red")
        axes.set_xlim(df_streamflow_sim_obs.index[0], df_streamflow_sim_obs.index[-1])
        axes.set_xlabel("Zeit")
        axes.set_ylabel("Durchfluss [m³/s]")
        axes.set_yscale("log")
        fig.tight_layout()
        file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"ts_streamflow_{_gauge}_run{model_run}.png"
        fig.savefig(file, dpi=300, bbox_inches="tight")
        plt.close(fig)


    # write metrics to csv
    click.echo("Writing evaluation metrics to csv...")
    file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / "figures" / f"evaluation_metrics_sfr_run{model_run}.csv"
    df_metrics_sfr.to_csv(file, sep=";")




    return

if __name__ == "__main__":
    main()
