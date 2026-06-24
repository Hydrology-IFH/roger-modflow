from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import click

def aggregate_to_coarser_resolution(vals, res_fine, res_coarse, method="sum", x_origin=0, y_origin=0):
    """Aggregate raster data to a coarser resolution.
    
    Args
    ----
    vals : numpy.ndarray
        2D array with the raster data.

    res_fine : int
        spatial resolution of the fine grid in meters.

    res_coarse : int
        spatial resolution of the coarse grid in meters.

    method : str
        Method to aggregate the data. Options are "sum" and "average".

    x_origin : int
        x-coordinate of the grid origin (lower left corner).

    y_origin : int
        y-coordinate of the grid origin (lower left corner).  
    """
    ny_fine, nx_fine = vals.shape[0], vals.shape[1]
    nlat_coarse, nlon_coarse = int(res_coarse / res_fine), int(res_coarse / res_fine)
    meters_to_latlon = 111195
    lat_fine = np.linspace(y_origin, y_origin + ny_fine*(res_fine/meters_to_latlon), ny_fine)/meters_to_latlon  # boundaries
    lon_fine = np.linspace(x_origin, x_origin + nx_fine*(res_fine/meters_to_latlon), nx_fine)/meters_to_latlon  # boundaries

    arr_fine = xr.DataArray(vals, coords={"lat": lat_fine, "lon": lon_fine}, dims=["lat", "lon"])

    if method == "sum":
        arr_coarse = xr.DataArray(
            arr_fine.coarsen(lat=nlat_coarse, lon=nlon_coarse).sum().values,
            dims=("lat", "lon"),
        )
        
    elif method == "average":
        arr_coarse = xr.DataArray(
            arr_fine.coarsen(lat=nlat_coarse, lon=nlon_coarse).mean().values,
            dims=("lat", "lon"),
        )
    return arr_coarse.values


@click.option("-mr", "--model-run", type=int, default=1806)
@click.command("main", short_help="Evaluate the transient simulation")
def main(model_run):
    base_path = Path(__file__).parent
    # base_path_output = Path("/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline-coupling/output")

    areas = ["dmn", "wsg_hausen", "wsg_zartener_becken", "wsg_boetzingen", "wsg_breisach", "wsg_ebringen", "wsg_eichstetten", "wsg_gottenheim", "wsg_krozinger_berg", "wsg_march", "wsg_schlatt", "wsg_tuniberg", "wsg_umkirch"]

    base = "base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"

    stress_test_scenarios = ["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction",
                             "base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude0-duration3_no-irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude0-duration3_irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude0-duration3_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "summer-drought-magnitude0-duration3_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude2-duration3_irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "summer-drought-magnitude2-duration3_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "long-term-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction",
                             "long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction",
                             "long-term-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress"]
    
    
    date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
    years = np.unique(date_time.year.values)

    # load modflow parameters to get the coordinates of the grid
    click.echo("Loading coordinates...")
    path = base_path.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    xcoords = ds_params.x.values + 25
    ycoords = ds_params.y.values - 25

    click.echo(f"Processing scenario {base}...")
    # load the groundwater depths
    click.echo("Loading groundwater depths (base)...")
    ll_gw_depths = []
    for year in years:
        output_file = base_path / "output" / f"modflow_{base}" / f"gw_depth_run{model_run}_year{year}.nc"
        # output_file = base_path_output / f"modflow_{base}" / f"gw_depth_run{model_run}_year{year}.nc"
        ds_gw_depths_base = xr.open_dataset(output_file, engine="h5netcdf")
        gw_depths_year_base = ds_gw_depths_base["depth"].values[:, 1, :, :]
        ll_gw_depths.append(gw_depths_year_base)
    gw_depths_base = np.concatenate(ll_gw_depths, axis=0)
    # # create xarray data array for groundwater depths
    # da_gw_depths_base = xr.DataArray(
    #     data=gw_depths_base,
    #     dims=["time", "y", "x"],
    #     coords={
    #         "time": date_time,
    #         "y": ds_gw_depths_base["lat"].values,
    #         "x": ds_gw_depths_base["lon"].values,
    #     },
    # )
    # # resample to monthly
    # da_gw_depths_base_monthly = da_gw_depths_base.resample(time="ME").mean()
    # # resample to annual
    # da_gw_depths_base_annual = da_gw_depths_base.resample(time="YE").mean()
    # # calculate the average groundwater depth for the base scenario
    # da_gw_depths_base_average = da_gw_depths_base.mean(dim="time")

    dict_depths_avg = {}
    dict_depths_anomalies_abs_avg = {}
    dict_depths_anomalies_percent_avg = {}

    for stress_test_scenario in stress_test_scenarios:
        click.echo(f"Processing scenario {stress_test_scenario}...")
        # load the indirect recharge
        click.echo("Loading groundwater depths (stress test)...")
        ll_gw_depths = []
        for year in years:
            output_file = base_path / "output" / f"modflow_{stress_test_scenario}" / f"gw_depth_run{model_run}_year{year}.nc"
            # output_file = base_path_output / f"{stress_test_scenario}" / f"gw_depth_run{model_run}_year{year}.nc"
            ds_gw_depths = xr.open_dataset(output_file, engine="h5netcdf")
            gw_depths_year = ds_gw_depths["depth"].values[:, 1, :, :]
            ll_gw_depths.append(gw_depths_year)
        gw_depths = np.concatenate(ll_gw_depths, axis=0)
        # create xarray data array for groundwater depths
        da_gw_depths = xr.DataArray(
            data=gw_depths,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_gw_depths["lat"].values,
                "x": ds_gw_depths["lon"].values,
            },
        )       

        # # resample to monthly
        # da_gw_depths_monthly = da_gw_depths.resample(time="ME").mean()
        # resample to annual
        da_gw_depths_annual = da_gw_depths.resample(time="YE").mean()

        # make figures directory if it does not exist
        figures_dir = base_path.parent / "figures" / "groundwater_anomalies" / stress_test_scenario
        figures_dir.mkdir(exist_ok=True)

        dict_depths_avg[stress_test_scenario] = {}
        dict_depths_anomalies_abs_avg[stress_test_scenario] = {}
        dict_depths_anomalies_percent_avg[stress_test_scenario] = {}

        for area in areas:
            if area == "dmn":
                mask = ds_params["mask_porous_aquifer"].values
                x1 = np.where(mask)[1].min()
                x2 = np.where(mask)[1].max()
                y1 = np.where(mask)[0].min()
                y2 = np.where(mask)[0].max()
                grid_extent = (xcoords[x1], xcoords[x2], ycoords[y1], ycoords[y2])
            else:
                file = base_path.parent / "input" / f"{area}_.tif"
                with rasterio.open(file) as src:
                    mask = src.read(1)
                    mask = np.where(mask == 1, True, False)
                x1 = np.where(mask)[1].min()
                x2 = np.where(mask)[1].max()
                y1 = np.where(mask)[0].min()
                y2 = np.where(mask)[0].max()
                grid_extent = (xcoords[x1], xcoords[x2], ycoords[y1], ycoords[y2])

            gw_depths_avg = np.nanmean(gw_depths_base, axis=0)[np.newaxis, :, :]
            # calculate daily anomalies of groundwater depths for the stress test scenario compared to the base scenario
            gw_depths_anomalies_abs = (gw_depths - gw_depths_avg) * (-1)
            gw_depths_anomalies_percent = (gw_depths_anomalies_abs / gw_depths_avg) * 100
            # calculate time series of average anomalies
            gw_depths_avg_ = np.zeros(gw_depths_anomalies_abs.shape[0])
            gw_depths_anomalies_abs_avg = np.zeros(gw_depths_anomalies_abs.shape[0])
            gw_depths_anomalies_percent_avg = np.zeros(gw_depths_anomalies_percent.shape[0])
            click.echo(f"Calculating time series of average groundwater depth anomalies for {area}...")
            for t in range(gw_depths_anomalies_abs.shape[0]):
                click.echo(f"Calculating time step {t+1} of {gw_depths_anomalies_abs.shape[0]}...")
                gw_depths_t = np.where(mask, gw_depths[t, :, :], np.nan)
                gw_depths_anomalies_abs_t = np.where(mask, gw_depths_anomalies_abs[t, :, :], np.nan)
                gw_depths_anomalies_percent_t = np.where(mask, gw_depths_anomalies_percent[t, :, :], np.nan)
                gw_depths_avg_[t] = np.nanmean(gw_depths_t)
                gw_depths_anomalies_abs_avg[t] = np.nanmean(gw_depths_anomalies_abs_t)
                gw_depths_anomalies_percent_avg[t] = np.nanmean(gw_depths_anomalies_percent_t)

            dict_depths_avg[stress_test_scenario][area] = gw_depths_avg_
            dict_depths_anomalies_abs_avg[stress_test_scenario][area] = gw_depths_anomalies_abs_avg
            dict_depths_anomalies_percent_avg[stress_test_scenario][area] = gw_depths_anomalies_percent_avg

            # # calculate monthly anomalies of groundwater depths for the stress test scenario compared to the base scenario
            # gw_depths_monthly_anomalies_abs = (da_gw_depths_monthly.values - gw_depths_avg) * (-1)
            # gw_depths_monthly_anomalies_percent = (da_gw_depths_monthly.values - gw_depths_avg) / gw_depths_avg * 100 * (-1)

            # calculate annual anomalies of groundwater depths for the stress test scenario compared to the base scenario
            gw_depths_annual_anomalies_abs = (da_gw_depths_annual.values - gw_depths_avg) * (-1)
            gw_depths_annual_anomalies_percent = (da_gw_depths_annual.values - gw_depths_avg) / gw_depths_avg * 100 * (-1)
            gw_depths_annual_anomalies_abs = np.where(mask, gw_depths_annual_anomalies_abs, np.nan)
            gw_depths_annual_anomalies_percent = np.where(mask, gw_depths_annual_anomalies_percent, np.nan)

            # plot map of annual anomalies of groundwater depths for the year 2018
            for year in years:
                click.echo(f"Plotting groundwater depth anomalies for {area} ({year})...")
                cond = (da_gw_depths_annual.time.dt.year == year).values
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(gw_depths_annual_anomalies_abs[cond, y1:y2, x1:x2][0], cmap="RdBu", vmin=-3, vmax=3, extent=grid_extent)
                if area == "dmn":
                    # plot every second xticklabels
                    xticks = ax.get_xticks()
                    ax.set_xticks(xticks[::2])
                    xticklabels = [f"{int(tick)}" for tick in xticks[::2]]
                    ax.set_xticklabels(xticklabels)
                ax.set_xlabel("X-Koordinate")
                ax.set_ylabel("Y-Koordinate")
                ax.axis('equal')
                ax.grid(True, alpha=0.3)
                fig.colorbar(im, ax=ax, label="GWFA Anomalie [m]")
                fig.tight_layout()
                fig.savefig(figures_dir / f"gw_depth_anomalies_abs_annual_{year}_{area}.pdf", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(gw_depths_annual_anomalies_percent[cond, y1:y2, x1:x2][0], cmap="RdBu", vmin=-50, vmax=50, extent=grid_extent)
                if area == "dmn":
                    # plot every second xticklabels
                    xticks = ax.get_xticks()
                    ax.set_xticks(xticks[::2])
                    xticklabels = [f"{int(tick)}" for tick in xticks[::2]]
                    ax.set_xticklabels(xticklabels)
                ax.set_xlabel("X-Koordinate")
                ax.set_ylabel("Y-Koordinate")
                ax.axis('equal')
                ax.grid(True, alpha=0.3)
                fig.colorbar(im, ax=ax, label="GWFA Anomalie [%]")
                fig.tight_layout()
                fig.savefig(figures_dir / f"gw_depth_anomalies_percent_annual_{year}_{area}.pdf", dpi=300)
                plt.close(fig)

        for area in areas:
            # click.echo(f"Plotting time series of groundwater depth anomalies for {area}...")
            # # plot time series of average groundwater depth anomalies for the three stress test scenarios
            # fig, ax = plt.subplots(figsize=(6, 2.5))
            # ax.axhline(0, color="grey", linestyle="-", alpha=0.8)
            # ax.plot(date_time, dict_depths_anomalies_abs_avg["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"][area], label="Base", color="#737373", lw=2.0)
            # ax.set_xlim(date_time[0], date_time[-1])
            # ax.set_xlabel("Zeit")
            # ax.set_ylabel("Mittlere GWFA Anomalie [m]")
            # # set y-axis limits to -10 to 10        
            # ax.set_ylim(-3, 3)
            # # turn legend off
            # ax.legend().set_visible(False)
            # fig.tight_layout()
            # fig.savefig(figures_dir / f"gw_depth_anomalies_abs_time_series_{area}_1.pdf", dpi=300)
            # plt.close(fig)

            # plot time series of average groundwater depth
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(date_time, dict_depths_avg["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"][area], label="Base", color="#737373", lw=2.0)
            ax.plot(date_time, dict_depths_avg[stress_test_scenario][area], color="#fd8d3c", lw=1.5)
            ax.set_xlim(date_time[0], date_time[-1])
            ax.set_xlabel("Zeit")
            ax.set_ylabel("Mittlere GWFA [m]")
            # set y-axis limits to -10 to 10
            # get ylimits of the current plot
            ylims = ax.get_ylim()
            _ylim = max(abs(ylims[0]), abs(ylims[1]))
            ylim = np.ceil(_ylim)  
            ax.set_ylim(0, ylim)
            # turn legend off
            ax.reverse_yaxis()
            ax.legend().set_visible(False)
            fig.tight_layout()
            fig.savefig(figures_dir / f"gw_depth_avg_time_series_{area}.pdf", dpi=300)
            plt.close(fig)

            # plot time series of average groundwater depth anomalies
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.axhline(0, color="black", linestyle="-", alpha=0.8)
            ax.plot(date_time, dict_depths_anomalies_abs_avg["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"][area], label="Base", color="#737373", lw=2.0)
            ax.plot(date_time, dict_depths_anomalies_abs_avg[stress_test_scenario][area], color="#fd8d3c", lw=1.5)
            ax.set_xlim(date_time[0], date_time[-1])
            ax.set_xlabel("Zeit")
            ax.set_ylabel("Mittlere GWFA Anomalie [m]")
            # set y-axis limits to -10 to 10
            # get ylimits of the current plot
            ylims = ax.get_ylim()
            _ylim = max(abs(ylims[0]), abs(ylims[1]))
            ylim = np.ceil(_ylim)  
            ax.set_ylim(-ylim, ylim)
            # turn legend off
            ax.legend().set_visible(False)
            fig.tight_layout()
            fig.savefig(figures_dir / f"gw_depth_anomalies_abs_time_series_{area}.pdf", dpi=300)
            plt.close(fig)

            # # plot time series of average groundwater depth anomalies for the three stress test scenarios
            # fig, ax = plt.subplots(figsize=(6, 2.5))
            # ax.axhline(0, color="black", linestyle="-", alpha=0.8)
            # ax.plot(date_time, dict_depths_anomalies_abs_avg["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"][area], label="Base", color="#737373", lw=2.0)
            # ax.plot(date_time, dict_depths_anomalies_abs_avg["summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction"][area], label="Summer Drought", color="#fd8d3c", lw=1.5)
            # ax.plot(date_time, dict_depths_anomalies_abs_avg["summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress"][area], label="Well Extraction Stress", color="#a63603", lw=0.9)
            # ax.set_xlim(date_time[0], date_time[-1])
            # ax.set_xlabel("Zeit")
            # ax.set_ylabel("Mittlere GWFA Anomalie [m]")
            # # set y-axis limits to -3 to 3        
            # ax.set_ylim(-3, 3)
            # # turn legend off
            # ax.legend().set_visible(False)
            # fig.tight_layout()
            # fig.savefig(figures_dir / f"gw_depth_anomalies_abs_time_series_{area}_3.pdf", dpi=300)
            # plt.close(fig)

            # # plot time series of average groundwater depth anomalies for the three stress test scenarios
            # fig, ax = plt.subplots(figsize=(6, 2.5))
            # ax.plot(date_time, dict_depths_anomalies_percent_avg["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"][area], label="Base", color="#737373")
            # ax.set_xlim(date_time[0], date_time[-1])
            # ax.set_xlabel("Zeit")
            # ax.set_ylabel("Mittlere GWFA Anomalie [%]")
            # ax.set_ylim(-100, 100)
            # ax.axhline(0, color="grey", linestyle="-", alpha=0.8)
            # # turn legend off
            # ax.legend().set_visible(False)
            # fig.tight_layout()
            # fig.savefig(figures_dir / f"gw_depth_anomalies_percent_time_series_{area}_1.pdf", dpi=300)
            # plt.close(fig)

            # plot time series of average groundwater depth anomalies for the three stress test scenarios
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.plot(date_time, dict_depths_anomalies_percent_avg["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"][area], label="Base", color="#737373")
            ax.plot(date_time, dict_depths_anomalies_percent_avg[stress_test_scenario][area], color="#fd8d3c", lw=1.2)
            ax.set_xlim(date_time[0], date_time[-1])
            ax.axhline(0, color="grey", linestyle="-", alpha=0.8)    
            ax.set_xlabel("Zeit")
            ax.set_ylabel("Mittlere GWFA Anomalie [%]")
            ax.set_ylim(-100, 100)
            # turn legend off        
            ax.legend().set_visible(False)
            fig.tight_layout()
            fig.savefig(figures_dir / f"gw_depth_anomalies_percent_time_series_{area}.pdf", dpi=300)
            plt.close(fig)

            # # plot time series of average groundwater depth anomalies for the three stress test scenarios
            # fig, ax = plt.subplots(figsize=(6, 2.5))
            # ax.plot(date_time, dict_depths_anomalies_percent_avg["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"][area], label="Base", color="#737373")
            # ax.plot(date_time, dict_depths_anomalies_percent_avg["summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction"][area], label="Summer Drought", color="#fd8d3c", lw=1.2)
            # ax.plot(date_time, dict_depths_anomalies_percent_avg["summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress"][area], label="Well Extraction Stress", color="#a63603", lw=1.2)
            # ax.set_xlim(date_time[0], date_time[-1])
            # ax.set_xlabel("Zeit")
            # ax.set_ylabel("Mittlere GWFA Anomalie [%]")
            # ax.set_ylim(-100, 100)
            # ax.axhline(0, color="grey", linestyle="-", alpha=0.8)
            # # turn legend off
            # ax.legend().set_visible(False)
            # fig.tight_layout()
            # fig.savefig(figures_dir / f"gw_depth_anomalies_percent_time_series_{area}_3.pdf", dpi=300)
            # plt.close(fig)

    return

if __name__ == "__main__":
    main()
