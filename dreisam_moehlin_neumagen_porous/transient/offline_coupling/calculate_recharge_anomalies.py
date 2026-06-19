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

    areas = ["dmn", "wsg_hausen", "wsg_zartener_becken", "wsg_boetzingen", "wsg_breisach", "wsg_ebringen", "wsg_eichstetten", "wsg_gottenheim", "wsg_krozinger_berg", "wsg_march", "wsg_schlatt", "wsg_tuniberg", "wsg_umkirch"]

    areas = ["dmn", "wsg_hausen"]

    base = "base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"

    # stress_test_scenarios = ["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction",
    #                          "base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction",
    #                          "summer-drought-magnitude0-duration3_no-irrigation_no-yellow-mustard_soil-compaction",
    #                          "summer-drought-magnitude0-duration3_irrigation_no-yellow-mustard_soil-compaction",
    #                          "summer-drought-magnitude0-duration3_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
    #                          "summer-drought-magnitude0-duration3_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
    #                          "summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction",
    #                          "summer-drought-magnitude2-duration3_irrigation_no-yellow-mustard_soil-compaction",
    #                          "summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
    #                          "summer-drought-magnitude2-duration3_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
    #                          "long-term-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction",
    #                          "long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction",
    #                          "long-term-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
    #                          "long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress"]
    

    stress_test_scenarios = ["summer-drought-magnitude0-duration3_no-irrigation_no-yellow-mustard_soil-compaction"]


    date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
    years = np.unique(date_time.year.values)
    timesteps = np.arange(len(date_time))

    # load modflow parameters to get the coordinates of the grid
    click.echo("Loading topography...")
    path = base_path.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    xcoords = ds_params.x.values + 25
    ycoords = ds_params.y.values - 25

    for area in areas:
        if area == "dmn":
            mask = ds_params["mask_porous_aquifer"].values
        else:
            file = base_path.parent / "input" / f"{area}_.tif"
            with rasterio.open(file) as src:
                mask = src.read(1)
                mask = np.where(mask == 1, True, False)

        click.echo(f"Processing scenario {base}...")
        # load the indirect recharge
        click.echo("Loading indirect recharge (base)...")
        ll_indirect_recharge = []
        for year in years:
            output_file = base_path / "output" / f"modflow_{base}" / f"indirect_recharge_run{model_run}_year{year}.nc"
            ds_indirect_recharge_base = xr.open_dataset(output_file, engine="h5netcdf")
            indirect_recharge_year_base = ds_indirect_recharge_base["indirect_recharge"].values * 86400  # convert from m3/s to m3/day
            indirect_recharge_year_base[indirect_recharge_year_base > 0] = 0  # set positive values to zero
            indirect_recharge_year_base = np.abs(indirect_recharge_year_base)
            indirect_recharge_year_base = np.where(mask[np.newaxis, :, :], indirect_recharge_year_base, 0)
            ll_indirect_recharge.append(indirect_recharge_year_base)
        indirect_recharge_base = np.concatenate(ll_indirect_recharge, axis=0)
        # create xarray data array for indirect recharge
        da_indirect_recharge_base = xr.DataArray(
            data=indirect_recharge_base,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_indirect_recharge_base["lat"].values,
                "x": ds_indirect_recharge_base["lon"].values,
            },
        )
        # resample to monthly
        da_indirect_recharge_base_monthly = da_indirect_recharge_base.resample(time="ME").sum()
        # dataframe with monthly total sum of indirect recharge
        df_indirect_recharge_base_monthly = pd.DataFrame(index=da_indirect_recharge_base_monthly.time.values, data=da_indirect_recharge_base_monthly.sum(dim=["y", "x"]).values, columns=["indirect_recharge"])
        # resample to annual
        da_indirect_recharge_base_annual = da_indirect_recharge_base.resample(time="YE").sum()
        # dataframe with annual total sum of indirect recharge
        df_indirect_recharge_base_annual = pd.DataFrame(index=da_indirect_recharge_base_annual.time.values, data=da_indirect_recharge_base_annual.sum(dim=["y", "x"]).values, columns=["indirect_recharge"])

        # load direct recharge
        click.echo("Loading direct recharge (base)...")
        ll_direct_recharge = []
        for year in years:
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"recharge_{base}_year{year}.nc"
            # output_file = base_path_output / f"{base}" / f"recharge_{base}_year{year}.nc"
            ds_direct_recharge_base = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _direct_recharge_year_base = ds_direct_recharge_base["recharge"].values
            _direct_recharge_year_base[_direct_recharge_year_base < 0] = 0  # set negative values to zero
            _direct_recharge_year_base[_direct_recharge_year_base > 100] = 100  # set values above 100 mm/day to 100 mm/day
            for i in range(_direct_recharge_year_base.shape[0]):
                direct_recharge_day = aggregate_to_coarser_resolution(_direct_recharge_year_base[i, :, :], 25, 50, method="average")
                direct_recharge_day = np.where(mask, direct_recharge_day, 0)
                ll_direct_recharge.append(direct_recharge_day)
        direct_recharge_base = np.stack(ll_direct_recharge, axis=0)
        # convert from mm/day to m3/day
        # get the area of each grid cell in m2
        _area = 50 * 50  # 50 m x 50 m grid cells
        # multiply direct recharge by area to get m3/day
        direct_recharge_base = direct_recharge_base * _area / 1000
        # create xarray data array for direct recharge
        da_direct_recharge_base = xr.DataArray(
            data=direct_recharge_base,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ycoords,
                "x": xcoords,
            },
        )
        # resample to monthly
        da_direct_recharge_base_monthly = da_direct_recharge_base.resample(time="ME").sum()
        # dataframe with monthly total sum of direct recharge
        df_direct_recharge_base_monthly = pd.DataFrame(index=da_direct_recharge_base_monthly.time.values, data=da_direct_recharge_base_monthly.sum(dim=["y", "x"]).values, columns=["direct_recharge"])

        # resample to annual
        da_direct_recharge_base_annual = da_direct_recharge_base.resample(time="YE").sum()
        # dataframe with annual total sum of direct recharge
        df_direct_recharge_base_annual = pd.DataFrame(index=da_direct_recharge_base_annual.time.values, data=da_direct_recharge_base_annual.sum(dim=["y", "x"]).values, columns=["direct_recharge"])

        recharge_base = indirect_recharge_base + direct_recharge_base
        # create xarray data array for total recharge
        da_recharge_base = da_indirect_recharge_base + da_direct_recharge_base
        # resample to monthly
        da_recharge_base_monthly = da_recharge_base.resample(time="ME").sum()
        # resample to annual
        da_recharge_base_annual = da_recharge_base.resample(time="YE").sum()

        df_recharge_base_monthly = pd.DataFrame(index=da_recharge_base_monthly.time.values, data=da_recharge_base_monthly.sum(dim=["y", "x"]).values, columns=["recharge"])
        df_recharge_base_annual = pd.DataFrame(index=da_recharge_base_annual.time.values, data=da_recharge_base_annual.sum(dim=["y", "x"]).values, columns=["recharge"])

        for stress_test_scenario in stress_test_scenarios:
            click.echo(f"Processing scenario {stress_test_scenario}...")
            # load the indirect recharge
            click.echo("Loading indirect recharge (stress test)...")
            ll_indirect_recharge = []
            for year in years:
                output_file = base_path / "output" / f"modflow_{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
                # output_file = base_path_output / f"modflow_{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
                ds_indirect_recharge = xr.open_dataset(output_file, engine="h5netcdf")
                indirect_recharge_year = ds_indirect_recharge["indirect_recharge"].values * 86400  # convert from m3/s to m3/day
                indirect_recharge_year[indirect_recharge_year > 0] = 0  # set positive values to zero
                indirect_recharge_year = np.abs(indirect_recharge_year)
                indirect_recharge_year = np.where(mask[np.newaxis, :, :], indirect_recharge_year, 0)
                ll_indirect_recharge.append(indirect_recharge_year)
            indirect_recharge = np.concatenate(ll_indirect_recharge, axis=0)
            # create xarray data array for indirect recharge
            da_indirect_recharge = xr.DataArray(
                data=indirect_recharge,
                dims=["time", "y", "x"],
                coords={
                    "time": date_time,
                    "y": ds_indirect_recharge["lat"].values,
                    "x": ds_indirect_recharge["lon"].values,
                },
            )
            # resample to monthly
            da_indirect_recharge_monthly = da_indirect_recharge.resample(time="ME").sum()
            # dataframe with monthly total sum of indirect recharge
            df_indirect_recharge_monthly = pd.DataFrame(index=da_indirect_recharge_monthly.time.values, data=da_indirect_recharge_monthly.sum(dim=["y", "x"]).values, columns=["indirect_recharge"])
            # resample to annual
            da_indirect_recharge_annual = da_indirect_recharge.resample(time="YE").sum()
            # dataframe with annual total sum of indirect recharge
            df_indirect_recharge_annual = pd.DataFrame(index=da_indirect_recharge_annual.time.values, data=da_indirect_recharge_annual.sum(dim=["y", "x"]).values, columns=["indirect_recharge"])

            indirect_recharge_base_avg = np.nanmean(df_indirect_recharge_base_monthly.values.flatten())
            indirect_recharge_avg = np.nanmean(df_indirect_recharge_monthly.values.flatten())
            click.echo(f"indirect recharge average (base; monthly): {indirect_recharge_base_avg}")
            click.echo(f"indirect recharge average (stress test; monthly): {indirect_recharge_avg}")
            df_indirect_recharge_anomalies_monthly_abs = df_indirect_recharge_monthly - indirect_recharge_base_avg
            df_indirect_recharge_anomalies_monthly_percent = ((df_indirect_recharge_monthly - indirect_recharge_base_avg) / indirect_recharge_base_avg) * 100
            indirect_recharge_base_avg_annual = np.nanmean(df_indirect_recharge_base_annual.values.flatten())
            indirect_recharge_avg_annual = np.nanmean(df_indirect_recharge_annual.values.flatten())
            click.echo(f"indirect recharge average (base; annual): {indirect_recharge_base_avg_annual}")
            click.echo(f"indirect recharge average (stress test; annual): {indirect_recharge_avg_annual}")
            df_indirect_recharge_anomalies_annual_abs = df_indirect_recharge_annual - indirect_recharge_base_avg_annual
            df_indirect_recharge_anomalies_annual_percent = ((df_indirect_recharge_annual - indirect_recharge_base_avg_annual) / indirect_recharge_base_avg_annual) * 100

            df_indirect_recharge_anomalies_monthly_abs.columns = ["anomaly"]
            df_indirect_recharge_anomalies_monthly_percent.columns = ["anomaly"]
            df_indirect_recharge_anomalies_annual_abs.columns = ["anomaly"]
            df_indirect_recharge_anomalies_annual_percent.columns = ["anomaly"]

            # load direct recharge
            click.echo("Loading direct recharge (stress test)...")
            ll_direct_recharge = []
            for year in years:
                _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"recharge_{_stress_test_scenario}_year{year}.nc"
                # output_file = base_path_output / f"{stress_test_scenario}" / f"recharge_{_stress_test_scenario}_year{year}.nc"
                ds_direct_recharge = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _direct_recharge_year = ds_direct_recharge["recharge"].values
                _direct_recharge_year[_direct_recharge_year < 0] = 0  # set negative values to zero
                _direct_recharge_year[_direct_recharge_year > 100] = 100  # set values above 100 mm/day to 100 mm/day
                for i in range(_direct_recharge_year.shape[0]):
                    direct_recharge_day = aggregate_to_coarser_resolution(_direct_recharge_year[i, :, :], 25, 50, method="average")
                    direct_recharge_day = np.where(mask, direct_recharge_day, 0)
                    ll_direct_recharge.append(direct_recharge_day)
            direct_recharge = np.stack(ll_direct_recharge, axis=0)
            # convert from mm/day to m3/day
            # get the area of each grid cell in m2
            _area = 50 * 50  # 50 m x 50 m grid cells
            # multiply direct recharge by area to get m3/day
            direct_recharge = direct_recharge * _area / 1000
            # create xarray data array for direct recharge
            da_direct_recharge = xr.DataArray(
                data=direct_recharge,
                dims=["time", "y", "x"],
                coords={
                    "time": date_time,
                    "y": ycoords,
                    "x": xcoords,
                },
            )
            # resample to monthly
            da_direct_recharge_monthly = da_direct_recharge.resample(time="ME").sum()
            # dataframe with monthly total sum of direct recharge
            df_direct_recharge_monthly = pd.DataFrame(index=da_direct_recharge_monthly.time.values, data=da_direct_recharge_monthly.sum(dim=["y", "x"]).values, columns=["direct_recharge"])

            # resample to annual
            da_direct_recharge_annual = da_direct_recharge.resample(time="YE").sum()
            # dataframe with annual total sum of direct recharge
            df_direct_recharge_annual = pd.DataFrame(index=da_direct_recharge_annual.time.values, data=da_direct_recharge_annual.sum(dim=["y", "x"]).values, columns=["direct_recharge"])

            direct_recharge_base_avg = np.nanmean(df_direct_recharge_base_monthly.values.flatten())
            direct_recharge_avg = np.nanmean(df_direct_recharge_monthly.values.flatten())
            click.echo(f"direct recharge average (base; monthly): {direct_recharge_base_avg}")
            click.echo(f"direct recharge average (stress test; monthly): {direct_recharge_avg}")
            df_direct_recharge_anomalies_monthly_abs = df_direct_recharge_monthly - direct_recharge_base_avg
            df_direct_recharge_anomalies_monthly_percent = ((df_direct_recharge_monthly - direct_recharge_base_avg) / direct_recharge_base_avg) * 100
            direct_recharge_base_avg_annual = np.nanmean(df_direct_recharge_base_annual.values.flatten())
            direct_recharge_avg_annual = np.nanmean(df_direct_recharge_annual.values.flatten())
            click.echo(f"direct recharge average (annual): {direct_recharge_base_avg_annual}")
            click.echo(f"direct recharge average (stress test; annual): {direct_recharge_avg_annual}")
            df_direct_recharge_anomalies_annual_abs = df_direct_recharge_annual - direct_recharge_base_avg_annual
            df_direct_recharge_anomalies_annual_percent = ((df_direct_recharge_annual - direct_recharge_base_avg_annual) / direct_recharge_base_avg_annual) * 100

            df_direct_recharge_anomalies_monthly_abs.columns = ["anomaly"]
            df_direct_recharge_anomalies_monthly_percent.columns = ["anomaly"]
            df_direct_recharge_anomalies_annual_abs.columns = ["anomaly"]
            df_direct_recharge_anomalies_annual_percent.columns = ["anomaly"]

            # create xarray data array for total recharge
            da_recharge = xr.DataArray(
                data=da_indirect_recharge.values + da_direct_recharge.values,
                dims=["time", "y", "x"],
                coords={
                    "time": date_time,
                    "y": ycoords,
                    "x": xcoords,
                },
            )

            # resample to monthly
            da_recharge_monthly = da_recharge.resample(time="ME").sum()
            # resample to annual
            da_recharge_annual = da_recharge.resample(time="YE").sum()

            df_recharge_monthly = pd.DataFrame(index=da_recharge_monthly.time.values, data=da_recharge_monthly.sum(dim=["y", "x"]).values, columns=["recharge"])
            # df_recharge_annual = pd.DataFrame(index=da_recharge_annual.time.values, data=da_recharge_annual.sum(dim=["y", "x"]).values, columns=["recharge"])

            recharge_base_avg = np.nanmean(df_recharge_base_monthly.values.flatten())
            df_recharge_anomalies_monthly_abs = pd.DataFrame(index=df_recharge_monthly.index, data=df_recharge_monthly["recharge"].values - recharge_base_avg, columns=["anomaly"])
            df_recharge_anomalies_monthly_percent = pd.DataFrame(index=df_recharge_monthly.index, data=(df_recharge_monthly["recharge"].values - recharge_base_avg) / recharge_base_avg * 100, columns=["anomaly"])
            recharge_base_avg_annual = np.nanmean(df_recharge_base_annual.values.flatten())
            # df_recharge_anomalies_annual_abs = pd.DataFrame(index=df_recharge_annual.index, data=df_recharge_annual["recharge"].values - recharge_base_avg_annual, columns=["anomaly"])
            # df_recharge_anomalies_annual_percent = pd.DataFrame(index=df_recharge_annual.index, data=(df_recharge_annual["recharge"].values - recharge_base_avg_annual) / recharge_base_avg_annual * 100, columns=["anomaly"])

            # make figures directory if it does not exist
            figures_dir = base_path.parent / "figures" / "recharge_anomalies" / stress_test_scenario
            figures_dir.mkdir(exist_ok=True)
            click.echo(f"Saving figures to {figures_dir}...")
    
            # plot monthly absolute anomalies of the recharge using a bar plot, make bars with negative values orange and bars with positive values blue
            fig, ax = plt.subplots(figsize=(6, 2.5))
            # use blue for positive anomalies and orange for negative anomalies
            ax.bar(df_recharge_monthly.index, df_recharge_monthly["recharge"], color="black", width=20)
            ax.set_xticks(df_recharge_monthly.index)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_recharge_monthly.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("GWN\n[Mio. m³/Monat]")
            ax.set_xlim(df_recharge_monthly.index[0] - pd.Timedelta(days=15), df_recharge_monthly.index[-1] + pd.Timedelta(days=15))
            ylims = ax.get_ylim()
            _ylim = max(abs(ylims[0]), abs(ylims[1]))
            ylim = np.ceil(_ylim)  
            ax.set_ylim(-ylim, ylim)
            fig.tight_layout()
            fig.savefig(figures_dir / f"recharge_monthly_{area}.pdf", dpi=300)
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(6, 2.5))
            # use blue for positive anomalies and orange for negative anomalies
            colors = ["orange" if x < 0 else "blue" for x in df_recharge_anomalies_monthly_abs["anomaly"]]
            ax.bar(df_recharge_anomalies_monthly_abs.index, df_recharge_anomalies_monthly_abs["anomaly"], color=colors, width=20)
            ax.set_xticks(df_recharge_anomalies_monthly_abs.index)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_recharge_anomalies_monthly_abs.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("GWN-Anomalie\n[Mio. m³/Monat]")
            ax.set_xlim(df_recharge_anomalies_monthly_abs.index[0] - pd.Timedelta(days=15), df_recharge_anomalies_monthly_abs.index[-1] + pd.Timedelta(days=15))
            ylims = ax.get_ylim()
            _ylim = max(abs(ylims[0]), abs(ylims[1]))
            ylim = np.ceil(_ylim)  
            ax.set_ylim(-ylim, ylim)
            fig.tight_layout()
            fig.savefig(figures_dir / f"recharge_anomalies_abs_monthly_{area}.pdf", dpi=300)
            plt.close(fig)

            # plot monthly relative anomalies of the recharge using a bar plot, make bars with negative values orange and bars with positive values blue
            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors = ["orange" if x < 0 else "blue" for x in df_recharge_anomalies_monthly_percent["anomaly"]]
            ax.bar(df_recharge_anomalies_monthly_percent.index, df_recharge_anomalies_monthly_percent["anomaly"], color=colors, width=20)
            ax.set_xticks(df_recharge_anomalies_monthly_percent.index)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_recharge_anomalies_monthly_percent.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("GWN-Anomalie\n[%]")
            ax.set_ylim(-100, 100)
            ax.set_xlim(df_recharge_anomalies_monthly_percent.index[0] - pd.Timedelta(days=15), df_recharge_anomalies_monthly_percent.index[-1] + pd.Timedelta(days=15))
            fig.tight_layout()
            fig.savefig(figures_dir / f"recharge_anomalies_percent_monthly_{area}.pdf", dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6, 2.5))
            df_recharge_anomalies_monthly_percent = df_recharge_anomalies_monthly_percent[(df_recharge_anomalies_monthly_percent.index.year >= 2016) & (df_recharge_anomalies_monthly_percent.index.year <= 2018)]
            colors = ["orange" if x < 0 else "blue" for x in df_recharge_anomalies_monthly_percent["anomaly"]]
            ax.bar(df_recharge_anomalies_monthly_percent.index, df_recharge_anomalies_monthly_percent["anomaly"], color=colors, width=20)
            ax.set_xticks(df_recharge_anomalies_monthly_percent.index)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_recharge_anomalies_monthly_percent.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("GWN-Anomalie\n[%]")
            ax.set_ylim(-100, 100)
            ax.set_xlim(df_recharge_anomalies_monthly_percent.index[0] - pd.Timedelta(days=15), df_recharge_anomalies_monthly_percent.index[-1] + pd.Timedelta(days=15))
            fig.tight_layout()
            fig.savefig(figures_dir / f"recharge_anomalies_percent_monthly_2016-2018_{area}.pdf", dpi=300)
            plt.close(fig)

            # plot monthly absolute anomalies of the indirect recharge using a bar plot, make bars with negative values orange and bars with positive values blue
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.bar(df_indirect_recharge_monthly.index, df_indirect_recharge_monthly["indirect_recharge"], color="black", width=20)
            ax.set_xticks(df_indirect_recharge_monthly.index)
            ax.set_xlim(df_indirect_recharge_monthly.index[0] - pd.Timedelta(days=15), df_indirect_recharge_monthly.index[-1] + pd.Timedelta(days=15))
            # ax.set_ylim(-10, 10)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_indirect_recharge_monthly.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ylims = ax.get_ylim()
            _ylim = max(abs(ylims[0]), abs(ylims[1]))
            ylim = np.ceil(_ylim)  
            ax.set_ylim(-ylim, ylim)
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("Indir. GWN\n[Mio. m³/Monat]")
            fig.tight_layout()
            fig.savefig(figures_dir / f"indirect_recharge_monthly_{area}.pdf", dpi=300)
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors = ["orange" if x < 0 else "blue" for x in df_indirect_recharge_anomalies_monthly_abs["anomaly"]]
            ax.bar(df_indirect_recharge_anomalies_monthly_abs.index, df_indirect_recharge_anomalies_monthly_abs["anomaly"], color=colors, width=20)
            ax.set_xticks(df_indirect_recharge_anomalies_monthly_abs.index)
            ax.set_xlim(df_indirect_recharge_anomalies_monthly_abs.index[0] - pd.Timedelta(days=15), df_indirect_recharge_anomalies_monthly_abs.index[-1] + pd.Timedelta(days=15))
            # ax.set_ylim(-10, 10)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_indirect_recharge_anomalies_monthly_abs.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ylims = ax.get_ylim()
            _ylim = max(abs(ylims[0]), abs(ylims[1]))
            ylim = np.ceil(_ylim)  
            ax.set_ylim(-ylim, ylim)
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("Indir. GWN-Anomalie\n[Mio. m³/Monat]")
            fig.tight_layout()
            fig.savefig(figures_dir / f"indirect_recharge_anomalies_abs_monthly_{area}.pdf", dpi=300)
            plt.close(fig)

            # plot mmonthly relative anomalies of the indirect recharge using a bar plot, make bars with negative values orange and bars with positive values blue
            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors = ["orange" if x < 0 else "blue" for x in df_indirect_recharge_anomalies_monthly_percent["anomaly"]]
            ax.bar(df_indirect_recharge_anomalies_monthly_percent.index, df_indirect_recharge_anomalies_monthly_percent["anomaly"], color=colors, width=20)
            ax.set_xticks(df_indirect_recharge_anomalies_monthly_percent.index)
            ax.set_xlim(df_indirect_recharge_anomalies_monthly_percent.index[0] - pd.Timedelta(days=15), df_indirect_recharge_anomalies_monthly_percent.index[-1] + pd.Timedelta(days=15))
            ax.set_ylim(-100, 100)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_indirect_recharge_anomalies_monthly_percent.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("Indir. GWN-Anomalie\n[%]")
            fig.tight_layout()
            fig.savefig(figures_dir / f"indirect_recharge_anomalies_percent_monthly_{area}.pdf", dpi=300)
            plt.close(fig)

            # plot mmonthly relative anomalies of the direct recharge using a bar plot, make bars with negative values orange and bars with positive values blue
            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.bar(df_direct_recharge_monthly.index, df_direct_recharge_monthly["direct_recharge"], color="black", width=20)
            ax.set_xticks(df_direct_recharge_monthly.index)
            ax.set_xlim(df_direct_recharge_monthly.index[0] - pd.Timedelta(days=15), df_direct_recharge_monthly.index[-1] + pd.Timedelta(days=15))
            # ax.set_ylim(-10, 10)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_direct_recharge_monthly.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ylims = ax.get_ylim()
            _ylim = max(abs(ylims[0]), abs(ylims[1]))
            ylim = np.ceil(_ylim)  
            ax.set_ylim(-ylim, ylim)
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("Dir. GWN\n[Mio. m³/Monat]")
            fig.tight_layout()
            fig.savefig(figures_dir / f"direct_recharge_{area}.pdf", dpi=300)
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors = ["orange" if x < 0 else "blue" for x in df_direct_recharge_anomalies_monthly_abs["anomaly"]]
            ax.bar(df_direct_recharge_anomalies_monthly_abs.index, df_direct_recharge_anomalies_monthly_abs["anomaly"], color=colors, width=20)
            ax.set_xticks(df_direct_recharge_anomalies_monthly_abs.index)
            ax.set_xlim(df_direct_recharge_anomalies_monthly_abs.index[0] - pd.Timedelta(days=15), df_direct_recharge_anomalies_monthly_abs.index[-1] + pd.Timedelta(days=15))  
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_direct_recharge_anomalies_monthly_abs.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ylims = ax.get_ylim()
            _ylim = max(abs(ylims[0]), abs(ylims[1]))
            ylim = np.ceil(_ylim)  
            ax.set_ylim(-ylim, ylim)
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("Dir. GWN-Anomalie\n[Mio. m³/Monat]")
            fig.tight_layout()
            fig.savefig(figures_dir / f"direct_recharge_anomalies_abs_monthly_{area}.pdf", dpi=300)
            plt.close(fig)
            
            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors = ["orange" if x < 0 else "blue" for x in df_direct_recharge_anomalies_monthly_percent["anomaly"]]
            ax.bar(df_direct_recharge_anomalies_monthly_percent.index, df_direct_recharge_anomalies_monthly_percent["anomaly"], color=colors, width=20)
            ax.set_xticks(df_direct_recharge_anomalies_monthly_percent.index)
            ax.set_xlim(df_direct_recharge_anomalies_monthly_percent.index[0] - pd.Timedelta(days=15), df_direct_recharge_anomalies_monthly_percent.index[-1] + pd.Timedelta(days=15))  
            ax.set_ylim(-100, 100)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_direct_recharge_anomalies_monthly_percent.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("Dir. GWN-Anomalie\n[%]")
            fig.tight_layout()
            fig.savefig(figures_dir / f"direct_recharge_anomalies_percent_monthly_{area}.pdf", dpi=300)
            plt.close(fig)

    return

if __name__ == "__main__":
    main()
