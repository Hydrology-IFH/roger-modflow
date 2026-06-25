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
    # base_path_output = base_path / "output"
    # base_path_output = Path("/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline-coupling/output")

    areas = ["wsg_hausen", "wsg_zartener_becken", "wsg_boetzingen", "wsg_breisach", "wsg_ebringen", "wsg_eichstetten", "wsg_gottenheim", "wsg_krozinger_berg", "wsg_march", "wsg_schlatt", "wsg_tuniberg", "wsg_umkirch"]

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
    click.echo("Loading topography...")
    path = base_path.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    xcoords = ds_params.x.values + 25
    ycoords = ds_params.y.values - 25

    for area in areas:
        click.echo(f"Processing {area}...")
        if area == "dmn":
            mask = ds_params["mask_porous_aquifer"].values
        else:
            file = base_path.parent / "input" / f"{area}_.tif"
            with rasterio.open(file) as src:
                mask = src.read(1)
                mask = np.where(mask == 1, True, False)

        for stress_test_scenario in stress_test_scenarios:
            click.echo(f"Processing scenario {stress_test_scenario}...")
            # load the indirect recharge
            click.echo("Loading indirect recharge...")
            ll_indirect_recharge = []
            for year in years:
                output_file = base_path / "output" / f"modflow_{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
                # output_file = base_path_output / f"{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
                ds_indirect_recharge = xr.open_dataset(output_file, engine="h5netcdf")
                indirect_recharge_year = ds_indirect_recharge["indirect_recharge"].values * 86400  # convert from m3/s to m3/day
                ds_indirect_recharge.close()
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

            # load direct recharge
            click.echo("Loading direct recharge...")
            ll_direct_recharge = []
            for year in years:
                _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"recharge_{_stress_test_scenario}_year{year}.nc"
                # _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
                # output_file = base_path_output / f"{stress_test_scenario}" / f"recharge_{_stress_test_scenario}_year{year}.nc"
                ds_direct_recharge = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _direct_recharge_year = ds_direct_recharge["recharge"].values
                ds_direct_recharge.close()
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

            # create xarray data array for total recharge
            da_recharge = da_indirect_recharge + da_direct_recharge
            # resample to monthly
            da_recharge_monthly = da_recharge.resample(time="ME").sum()
            # resample to annual
            da_recharge_annual = da_recharge.resample(time="YE").sum()

            df_recharge_monthly = pd.DataFrame(index=da_recharge_monthly.time.values, data=da_recharge_monthly.sum(dim=["y", "x"]).values, columns=["recharge"])
            df_recharge_annual = pd.DataFrame(index=da_recharge_annual.time.values, data=da_recharge_annual.sum(dim=["y", "x"]).values, columns=["recharge"])

            # load well extraction
            click.echo("Loading well extraction...")
            ll_well_extraction = []
            for year in years:
                output_file = base_path / "output" / f"modflow_{stress_test_scenario}" / f"well_extraction_run{model_run}_year{year}.nc"
                # output_file = base_path_output / f"{stress_test_scenario}" / f"well_extraction_run{model_run}_year{year}.nc"
                ds_well_extraction = xr.open_dataset(output_file, engine="h5netcdf")
                well_extraction_year = ds_well_extraction["well_extraction"].values
                ds_well_extraction.close()
                well_extraction_year = np.where(mask[np.newaxis, :, :], well_extraction_year, 0)
                ll_well_extraction.append(well_extraction_year)
            well_extraction = np.concatenate(ll_well_extraction, axis=0)
            # create xarray data array for well extraction
            da_well_extraction = xr.DataArray(
                data=well_extraction,
                dims=["time", "y", "x"],
                coords={
                    "time": date_time,
                    "y": ycoords,
                    "x": xcoords,
                },
            )
            # resample to monthly
            da_well_extraction_monthly = da_well_extraction.resample(time="ME").sum()
            # dataframe with monthly total sum of well extraction
            df_well_extraction_monthly = pd.DataFrame(index=da_well_extraction_monthly.time.values, data=da_well_extraction_monthly.sum(dim=["y", "x"]).values, columns=["well_extraction"])
            # resample to annual
            da_well_extraction_annual = da_well_extraction.resample(time="YE").sum()
            # dataframe with annual total sum of well extraction
            df_well_extraction_annual = pd.DataFrame(index=da_well_extraction_annual.time.values, data=da_well_extraction_annual.sum(dim=["y", "x"]).values, columns=["well_extraction"])
            # dataframe with long-term average annual well extraction
            df_well_extraction_long_term_average = pd.DataFrame(index=["long_term_average"], data=[df_well_extraction_annual["well_extraction"].mean()], columns=["well_extraction"])

            if "_irrigation_" in stress_test_scenario:
                # load irrigation
                click.echo("Loading irrigation...")
                ll_irrigation = []
                for year in years:
                    _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
                    base_path_roger = base_path.parent.parent.parent.parent / "roger"
                    output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"irrigation_{_stress_test_scenario}_year{year}.nc"
                    ds_irrigation = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                    _irrigation_year = ds_irrigation["irrigation"].values
                    _irrigation_year = np.where(_irrigation_year < 0, 0, _irrigation_year)  # set negative values to zero
                    ds_irrigation.close()
                    for i in range(_irrigation_year.shape[0]):
                        irrigation_day = aggregate_to_coarser_resolution(_irrigation_year[i, :, :], 25, 50, method="average")
                        irrigation_day = np.where(mask, irrigation_day, 0)
                        ll_irrigation.append(irrigation_day)
                irrigation = np.stack(ll_irrigation, axis=0)
                # convert from mm/day to m3/day
                # get the area of each grid cell in m2
                _area = 50 * 50  # 50 m x 50 m grid cells
                # multiply irrigation by area to get m3/day
                irrigation = irrigation * _area / 1000
                # create xarray data array for irrigation
                da_irrigation = xr.DataArray(
                    data=irrigation,
                    dims=["time", "y", "x"],
                    coords={
                        "time": date_time,
                        "y": ycoords,
                        "x": xcoords,
                    },
                )
                # resample to monthly
                da_irrigation_monthly = da_irrigation.resample(time="ME").sum()
                # dataframe with monthly total sum of irrigation
                df_irrigation_monthly = pd.DataFrame(index=da_irrigation_monthly.time.values, data=da_irrigation_monthly.sum(dim=["y", "x"]).values, columns=["irrigation"])

                # resample to annual
                da_irrigation_annual = da_irrigation.resample(time="YE").sum()
                # dataframe with annual total sum of irrigation
                df_irrigation_annual = pd.DataFrame(index=da_irrigation_annual.time.values, data=da_irrigation_annual.sum(dim=["y", "x"]).values, columns=["irrigation"])

            # calculate the monthly extraction balance
            df_extraction_balance_monthly = pd.DataFrame(index=df_well_extraction_monthly.index, columns=["sustainable_extraction", "actual_extraction", "extraction_balance"])
            df_extraction_balance_monthly["sustainable_extraction"] = df_recharge_monthly["recharge"].mean() * 0.3
            if "_irrigation" in stress_test_scenario:
                df_extraction_balance_monthly["actual_extraction"] = df_well_extraction_monthly["well_extraction"] + df_irrigation_monthly["irrigation"]
            else:
                df_extraction_balance_monthly["actual_extraction"] = df_well_extraction_monthly["well_extraction"]
            df_extraction_balance_monthly["extraction_balance"] = df_extraction_balance_monthly["sustainable_extraction"] - df_extraction_balance_monthly["actual_extraction"]

            # calculate the annual extraction balance
            df_extraction_balance_annual = pd.DataFrame(index=df_well_extraction_annual.index, columns=["sustainable_extraction", "actual_extraction", "extraction_balance"])
            df_extraction_balance_annual["sustainable_extraction"] = df_recharge_annual["recharge"].mean() * 0.3
            if "_irrigation" in stress_test_scenario:
                df_extraction_balance_annual["actual_extraction"] = df_well_extraction_annual["well_extraction"] + df_irrigation_annual["irrigation"]
            else:
                df_extraction_balance_annual["actual_extraction"] = df_well_extraction_annual["well_extraction"]
            df_extraction_balance_annual["extraction_balance"] = df_extraction_balance_annual["sustainable_extraction"] - df_extraction_balance_annual["actual_extraction"]

            # calculate the long-term sum extraction balance
            df_extraction_balance_long_term = pd.DataFrame(index=["long_term"], columns=["sustainable_extraction", "actual_extraction", "extraction_balance"])
            df_extraction_balance_long_term["sustainable_extraction"] = df_recharge_monthly["recharge"].sum() * 0.3
            if "_irrigation" in stress_test_scenario:
                df_extraction_balance_long_term["actual_extraction"] = df_well_extraction_monthly["well_extraction"].sum() + df_irrigation_monthly["irrigation"].sum()
            else:
                df_extraction_balance_long_term["actual_extraction"] = df_well_extraction_monthly["well_extraction"].sum()
            df_extraction_balance_long_term["extraction_balance"] = df_extraction_balance_long_term["sustainable_extraction"] - df_extraction_balance_long_term["actual_extraction"]

            # make figures directory if it does not exist
            figures_dir = base_path.parent / "figures" / "gw_extraction_balance" / stress_test_scenario
            figures_dir.mkdir(exist_ok=True)

            click.echo("Plotting the extraction balance...")

            # plot monthly drinking water well extraction
            fig, ax = plt.subplots(figsize=(6, 2.5))
            # convert to million m3/month
            ax.bar(df_well_extraction_monthly.index, df_well_extraction_monthly["well_extraction"] / 1e6, color="purple", width=15)
            # rotate xticklabels to vertical and show only the year and month
            xticklabels = df_well_extraction_monthly.index.strftime("%y-%m")
            ax.set_xticks(df_well_extraction_monthly.index)
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlim(df_well_extraction_monthly.index[0] - pd.DateOffset(months=1), df_well_extraction_monthly.index[-1] + pd.DateOffset(months=1))
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("GW-Entnahme\n[Mio. m³/Monat]")
            # set legend off
            ax.legend().set_visible(False)
            fig.tight_layout()
            fig.savefig(figures_dir / f"well_extraction_monthly_{area}.pdf", dpi=300)

            # plot annual drinking water well extraction
            fig, ax = plt.subplots(figsize=(6, 2))
            # convert to million m3/year
            ax.bar(df_well_extraction_annual.index.year, df_well_extraction_annual["well_extraction"] / 1e6, color="purple", width=0.8)
            # rotate xticklabels to vertical and show only the year
            ax.set_xlabel("Jahr")
            ax.set_ylabel("GW-Entnahme\n[Mio. m³/Jahr]")
            # set legend off        
            ax.legend().set_visible(False)
            fig.tight_layout()
            fig.savefig(figures_dir / f"well_extraction_annual_{area}.pdf", dpi=300)

            # plot long-term average annual drinking water well extraction
            fig, ax = plt.subplots(figsize=(2, 2))
            long_term_average = df_well_extraction_annual["well_extraction"].mean()
            df_long_term_average = pd.DataFrame(index=["long_term_average"], data=[long_term_average], columns=["well_extraction"])
            ax.bar(df_long_term_average.index, df_long_term_average["well_extraction"] / 1e6, color="purple")
            ax.set_xlabel("")
            ax.set_xticklabels([""], rotation=0)
            ax.set_ylabel("GW-Entnahme\n[Mio. m³/Jahr]")
            # set legend off
            ax.legend().set_visible(False)
            fig.tight_layout()
            fig.savefig(figures_dir / f"well_extraction_annual_long_term_average_{area}.pdf", dpi=300)
            click.echo(f"Long-term average annual drinking water well extraction: {long_term_average:.2f} million m3/year")

            if "_irrigation_" in stress_test_scenario:
                # plot monthly irrigation
                fig, ax = plt.subplots(figsize=(6, 2.5))
                # convert to million m3/month
                ax.bar(df_irrigation_monthly.index, df_irrigation_monthly["irrigation"] / 1e6, color="pink", width=15)
                # rotate xticklabels to vertical and show only the year and month
                xticklabels = df_irrigation_monthly.index.strftime("%y-%m")
                ax.set_xticks(df_irrigation_monthly.index)
                ax.set_xticklabels(xticklabels, rotation=90)
                ax.set_xlim(df_irrigation_monthly.index[0] - pd.DateOffset(months=1), df_irrigation_monthly.index[-1] + pd.DateOffset(months=1))
                ax.set_xlabel("Zeit [Jahr-Monat]")
                ax.set_ylabel("Bewässerung\n[Mio. m³/Monat]")
                # set legend off
                ax.legend().set_visible(False)
                fig.tight_layout()
                fig.savefig(figures_dir / f"irrigation_monthly_{area}.pdf", dpi=300)

                # plot annual irrigation
                fig, ax = plt.subplots(figsize=(6, 2))
                # convert to million m3/year
                ax.bar(df_irrigation_annual.index.year, df_irrigation_annual["irrigation"] / 1e6, color="pink", width=0.8)
                # rotate xticklabels to vertical and show only the year
                ax.set_xlabel("Jahr")
                ax.set_ylabel("Bewässerung\n[Mio. m³/Jahr]")
                # set legend off        
                ax.legend().set_visible(False)
                fig.tight_layout()
                fig.savefig(figures_dir / f"irrigation_annual_{area}.pdf", dpi=300)

                # plot long-term average annual irrigation
                fig, ax = plt.subplots(figsize=(2, 2))
                long_term_average = df_irrigation_annual["irrigation"].mean()
                df_long_term_average = pd.DataFrame(index=["long_term_average"], data=[long_term_average], columns=["irrigation"])
                ax.bar(df_long_term_average.index, df_long_term_average["irrigation"] / 1e6, color="pink")
                ax.set_xlabel("")
                ax.set_xticklabels([""], rotation=0)
                ax.set_ylabel("Bewässerung\n[Mio. m³/Jahr]")
                # set legend off
                ax.legend().set_visible(False)
                fig.tight_layout()
                fig.savefig(figures_dir / f"irrigation_annual_long_term_average_{area}.pdf", dpi=300)
                click.echo(f"Long-term average annual irrigation: {long_term_average:.2f} million m3/year")

                # plot monthly irrigation and well extraction using stacked bar plot use pink for irrigation and purple for well extraction
                fig, ax = plt.subplots(figsize=(6, 2.5))
                df_extraction_monthly_stacked = df_well_extraction_monthly.copy()
                df_extraction_monthly_stacked["irrigation"] = df_irrigation_monthly["irrigation"]/1e6  # convert to million m3/month
                df_extraction_monthly_stacked["well_extraction"] = df_extraction_monthly_stacked["well_extraction"]/1e6  # convert to million m3/month
                # make stacked bar plot
                ax.bar(df_extraction_monthly_stacked.index, df_extraction_monthly_stacked["well_extraction"], color="purple", label="Trinkwasser", width=15)
                ax.bar(df_extraction_monthly_stacked.index, df_extraction_monthly_stacked["irrigation"], bottom=df_extraction_monthly_stacked["well_extraction"], color="pink", label="Bewässerung", width=15)
                ax.set_xticks(df_extraction_monthly_stacked.index)
                # reformat xticklabels to show only the year and month and plot labels of every 4th month
                xticklabels = df_extraction_monthly_stacked.index.strftime("%y-%m")
                xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
                ax.set_xticklabels(xticklabels, rotation=90)
                ax.set_xlim(df_extraction_monthly_stacked.index[0] - pd.DateOffset(months=1), df_extraction_monthly_stacked.index[-1] + pd.DateOffset(months=1))
                ax.set_xlabel("Zeit [Jahr-Monat]")
                ax.set_ylabel("GW-Entnahme\n[Mio. m³/Monat]")
                # put legend outside of the plot on the top center
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
                fig.tight_layout()
                fig.savefig(figures_dir / f"extraction_monthly_stacked_{area}.pdf", dpi=300)

                # plot annual irrigation and well extraction using stacked bar plot use pink for irrigation and purple for well extraction
                fig, ax = plt.subplots(figsize=(6, 2))
                df_extraction_annual_stacked = df_well_extraction_annual.copy()
                df_extraction_annual_stacked["irrigation"] = df_irrigation_annual["irrigation"]/1e6  # convert to million m3/year
                df_extraction_annual_stacked["well_extraction"] = df_extraction_annual_stacked["well_extraction"]/1e6  # convert to million m3/year
                ax.bar(df_extraction_annual_stacked.index.year, df_extraction_annual_stacked["well_extraction"], color="purple", label="Trinkwasser", width=0.8)
                ax.bar(df_extraction_annual_stacked.index.year, df_extraction_annual_stacked["irrigation"], bottom=df_extraction_annual_stacked["well_extraction"], color="pink", label="Bewässerung", width=0.8)
                ax.set_xlabel("Jahr")
                ax.set_ylabel("GW-Entnahme\n[Mio. m³/Jahr]")
                # put legend outside of the plot on the top center
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
                fig.tight_layout()
                fig.savefig(figures_dir / f"extraction_annual_stacked_{area}.pdf", dpi=300)

                # plot monthly irrigation share
                fig, ax = plt.subplots(figsize=(6, 2.5))
                # convert to million m3/month
                monthly_total_extraction = df_well_extraction_monthly["well_extraction"] + df_irrigation_monthly["irrigation"]
                monthly_irrigation_share = (df_irrigation_monthly["irrigation"] / monthly_total_extraction) * 100
                ax.bar(df_irrigation_monthly.index, monthly_irrigation_share, color="pink", width=15)
                # rotate xticklabels to vertical and show only the year and month
                xticklabels = df_irrigation_monthly.index.strftime("%y-%m")
                ax.set_xticks(df_irrigation_monthly.index)
                ax.set_xticklabels(xticklabels, rotation=90)
                ax.set_xlim(df_irrigation_monthly.index[0] - pd.DateOffset(months=1), df_irrigation_monthly.index[-1] + pd.DateOffset(months=1))
                ax.set_xlabel("Zeit [Jahr-Monat]")
                ax.set_ylabel("Anteil Bewässerung\nan GW-Entnahme [%]")
                # set legend off
                ax.legend().set_visible(False)
                fig.tight_layout()
                fig.savefig(figures_dir / f"irrigation_monthly_{area}.pdf", dpi=300)

            # plot monthly direct and indirect recharge using stacked bar plot use blue for direct recharge and purple for indirect recharge 
            fig, ax = plt.subplots(figsize=(6, 2.5))
            df_recharge_monthly_stacked = df_direct_recharge_monthly.copy()
            df_recharge_monthly_stacked["indirect_recharge"] = df_indirect_recharge_monthly["indirect_recharge"]/1e6  # convert to million m3/month
            df_recharge_monthly_stacked["direct_recharge"] = df_recharge_monthly_stacked["direct_recharge"]/1e6  # convert to million m3/month
            # make stacked bar plot
            ax.bar(df_recharge_monthly_stacked.index, df_recharge_monthly_stacked["direct_recharge"], color="blue", label="Direkte GWN", width=15)
            ax.bar(df_recharge_monthly_stacked.index, df_recharge_monthly_stacked["indirect_recharge"], bottom=df_recharge_monthly_stacked["direct_recharge"], color="lightblue", label="Indirekte GWN", width=15)
            ax.set_xticks(df_recharge_monthly_stacked.index)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            xticklabels = df_recharge_monthly_stacked.index.strftime("%y-%m")
            xticklabels = [label if i % 4 == 0 else "" for i, label in enumerate(xticklabels)]
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlim(df_recharge_monthly_stacked.index[0] - pd.DateOffset(months=1), df_recharge_monthly_stacked.index[-1] + pd.DateOffset(months=1))
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("GWN\n[Mio. m³/Monat]")
            # put legend outside of the plot on the top center
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
            fig.tight_layout()
            fig.savefig(figures_dir / f"recharge_monthly_stacked_{area}.pdf", dpi=300)
            plt.close(fig)

            # plot annual direct and indirect recharge using stacked bar plot use blue for direct recharge and purple for indirect recharge
            fig, ax = plt.subplots(figsize=(6, 2))
            df_recharge_annual_stacked = df_direct_recharge_annual.copy()
            df_recharge_annual_stacked["indirect_recharge"] = df_indirect_recharge_annual["indirect_recharge"]/1e6  # convert to million m3/year
            df_recharge_annual_stacked["direct_recharge"] = df_recharge_annual_stacked["direct_recharge"]/1e6  # convert to million m3/year
            ax.bar(df_recharge_annual_stacked.index.year, df_recharge_annual_stacked["direct_recharge"], color="blue", label="Direkte GWN", width=0.8)
            ax.bar(df_recharge_annual_stacked.index.year, df_recharge_annual_stacked["indirect_recharge"], bottom=df_recharge_annual_stacked["direct_recharge"], color="lightblue", label="Indirekte GWN", width=0.8)
            ax.set_xlabel("Jahr")
            ax.set_ylabel("GWN\n[Mio. m³/Jahr]")
            # put legend outside of the plot on the top center
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2)
            fig.tight_layout()
            fig.savefig(figures_dir / f"recharge_annual_stacked_{area}.pdf", dpi=300)
            plt.close(fig)

            # plot long-term annual average direct and indirect recharge using stacked bar plot use blue for direct recharge and purple for indirect recharge
            fig, ax = plt.subplots(figsize=(2, 2))
            df_recharge_long_term_stacked = pd.DataFrame(index=["long_term_average"], columns=["direct_recharge", "indirect_recharge"])
            df_recharge_long_term_stacked["direct_recharge"] = df_direct_recharge_monthly["direct_recharge"].mean()/1e6  # convert to million m3/year    
            df_recharge_long_term_stacked["indirect_recharge"] = df_indirect_recharge_monthly["indirect_recharge"].mean()/1e6  # convert to million m3/year
            ax.bar(df_recharge_long_term_stacked.index, df_recharge_long_term_stacked["direct_recharge"], color="blue", label="Direkte GWN")
            ax.bar(df_recharge_long_term_stacked.index, df_recharge_long_term_stacked["indirect_recharge"], bottom=df_recharge_long_term_stacked["direct_recharge"], color="lightblue", label="Indirekte GWN")
            ax.set_xlabel("")
            ax.set_xticklabels([""], rotation=0)
            ax.set_ylabel("GWN\n[Mio. m³/Monat]")
            # set legend off    
            ax.legend().set_visible(False)
            fig.tight_layout()
            fig.savefig(figures_dir / f"recharge_long_term_stacked_{area}.pdf", dpi=300)
            plt.close(fig)
            click.echo(f"Long-term average recharge ({area}): {df_recharge_long_term_stacked['direct_recharge'].values[0] + df_recharge_long_term_stacked['indirect_recharge'].values[0]:.2f} million m3/year")
            click.echo(f"Long-term average direct recharge ({area}): {df_recharge_long_term_stacked['direct_recharge'].values[0]:.2f} million m3/year")
            click.echo(f"Long-term average indirect recharge ({area}): {df_recharge_long_term_stacked['indirect_recharge'].values[0]:.2f} million m3/year")
            # print the percentage of indirect recharge in the total recharge
            percentage_indirect_recharge = (df_recharge_long_term_stacked["indirect_recharge"].values[0] / (df_recharge_long_term_stacked["direct_recharge"].values[0] + df_recharge_long_term_stacked["indirect_recharge"].values[0])) * 100
            click.echo(f"Percentage of indirect recharge in total recharge ({area}): {percentage_indirect_recharge:.2f}%")
            # print the percentage of direct recharge in the total recharge
            percentage_direct_recharge = (df_recharge_long_term_stacked["direct_recharge"].values[0] / (df_recharge_long_term_stacked["direct_recharge"].values[0] + df_recharge_long_term_stacked["indirect_recharge"].values[0])) * 100
            click.echo(f"Percentage of direct recharge in total recharge ({area}): {percentage_direct_recharge:.2f}%")

            # plot the monthly extraction balance using bar plot, make bars with negative values orange and bars with positive values blue
            fig, ax = plt.subplots(figsize=(6, 2.5))
            colors = ["purple" if x < 0 else "blue" for x in df_extraction_balance_monthly["extraction_balance"]]
            ax.bar(df_extraction_balance_monthly.index, df_extraction_balance_monthly["extraction_balance"]/1e6, color=colors, width=15)
            # reformat xticklabels to show only the year and month and plot labels of every 4th month
            ax.set_xlim(df_extraction_balance_monthly.index[0], df_extraction_balance_monthly.index[-1])
            ax.set_xticks(df_extraction_balance_monthly.index)
            xticklabels = df_extraction_balance_monthly.index.strftime("%y-%m")
            xticklabels = [label if i % 6 == 0 else "" for i, label in enumerate(xticklabels)]
            ax.set_xticklabels(xticklabels, rotation=90)
            ax.set_xlabel("Zeit [Jahr-Monat]")
            ax.set_ylabel("GW-Entnahmebilanz\n[Mio. m³/Monat]")
            fig.tight_layout()
            fig.savefig(figures_dir / f"extraction_balance_monthly_{area}.pdf", dpi=300)
            plt.close(fig)

            # plot the annual extraction balance using bar plot, make bars with negative values orange and bars with positive values blue
            fig, ax = plt.subplots(figsize=(6, 2))
            colors = ["purple" if x < 0 else "blue" for x in df_extraction_balance_annual["extraction_balance"]]
            ax.bar(df_extraction_balance_annual.index.year, df_extraction_balance_annual["extraction_balance"]/1e6, color=colors, width=0.8)
            ax.set_xlabel("Jahr")
            ax.set_ylabel("GW-Entnahmebilanz\n[Mio. m³/Jahr]")
            fig.tight_layout()
            fig.savefig(figures_dir / f"extraction_balance_annual_{area}.pdf", dpi=300)
            plt.close(fig)

            # plot the long-term balance of the extraction balance using a bar plot, make bars with negative values orange and bars with positive values blue
            long_term_balance = df_extraction_balance_annual["extraction_balance"].sum()
            fig, ax = plt.subplots(figsize=(2, 2))
            color = "purple" if long_term_balance < 0 else "blue"
            ax.bar([""], [long_term_balance/1e6], color=color)
            ax.set_ylabel("GW-Entnahmebilanz\n[Mio. m³]")
            fig.tight_layout()
            fig.savefig(figures_dir / f"extraction_balance_long_term_{area}.pdf", dpi=300)
            plt.close(fig)

            # remove data arrays to free up memory
            del da_indirect_recharge, da_indirect_recharge_monthly, da_indirect_recharge_annual
            del da_direct_recharge, da_direct_recharge_monthly, da_direct_recharge_annual
            del da_recharge, da_recharge_monthly, da_recharge_annual
            del da_well_extraction, da_well_extraction_monthly, da_well_extraction_annual
            # remove list of arrays to free up memory
            del ll_indirect_recharge, ll_direct_recharge, ll_well_extraction

    return

if __name__ == "__main__":
    main()
