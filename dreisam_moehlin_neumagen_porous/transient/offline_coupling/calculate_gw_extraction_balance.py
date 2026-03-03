from pathlib import Path
import numpy as np
import xarray as xr
import xesmf as xe
import pandas as pd
import geopandas as gpd
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

    date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
    years = np.unique(date_time.year.values)
    timesteps = np.arange(len(date_time))

    # load modflow parameters to get the coordinates of the grid
    click.echo("Loading topography...")
    path = base_path.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    xcoords = ds_params["x"].values
    ycoords = ds_params["y"].values

    # load the indirect recharge
    # click.echo("Loading indirect recharge...")
    # ll_indirect_recharge = []
    # for year in years:
    #     output_file = base_path / "output" / "modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction" / f"indirect_recharge_run{model_run}_year{year}.nc"
    #     ds_indirect_recharge = xr.open_dataset(output_file, engine="h5netcdf")
    #     indirect_recharge_year = ds_indirect_recharge["indirect_recharge"].values
    #     ll_indirect_recharge.append(indirect_recharge_year)
    # indirect_recharge = np.concatenate(ll_indirect_recharge, axis=0)
    # # create xarray data array for indirect recharge
    # da_indirect_recharge = xr.DataArray(
    #     data=indirect_recharge,
    #     dims=["time", "y", "x"],
    #     coords={
    #         "time": date_time,
    #         "y": ds_indirect_recharge["y"].values,
    #         "x": ds_indirect_recharge["x"].values,
    #     },
    # )
    # # resample to monthly
    # da_indirect_recharge_monthly = da_indirect_recharge.resample(time="ME").sum()
    # # dataframe with monthly total sum of indirect recharge
    # df_indirect_recharge_monthly = pd.DataFrame(index=da_indirect_recharge_monthly.time.values, data=da_indirect_recharge_monthly.sum(dim=["y", "x"]).values, columns=["indirect_recharge"])
    # # resample to annual
    # da_indirect_recharge_annual = da_indirect_recharge.resample(time="YE").sum()
    # # dataframe with annual total sum of indirect recharge
    # df_indirect_recharge_annual = pd.DataFrame(index=da_indirect_recharge_annual.time.values, data=da_indirect_recharge_annual.sum(dim=["y", "x"]).values, columns=["indirect_recharge"])

    # load direct recharge
    click.echo("Loading direct recharge...")
    ll_direct_recharge = []
    for year in years:
        base_path_roger = base_path.parent.parent.parent.parent / "roger"
        output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc"
        # output_file = base_path.parent / "input" / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc"
        ds_direct_recharge = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
        _direct_recharge_year = ds_direct_recharge["recharge"].values
        _direct_recharge_year[_direct_recharge_year < 0] = 0  # set negative values to zero
        _direct_recharge_year[_direct_recharge_year > 100] = 100  # set values above 100 mm/day to 100 mm/day
        for i in range(_direct_recharge_year.shape[0]):
            direct_recharge_day = aggregate_to_coarser_resolution(_direct_recharge_year[i, :, :], 25, 50, method="average")
            ll_direct_recharge.append(direct_recharge_day)
    direct_recharge = np.stack(ll_direct_recharge, axis=0)
    click.echo(direct_recharge.shape)
    # convert from mm/day to m3/day
    # get the area of each grid cell in m2
    area = 50 * 50  # 50 m x 50 m grid cells
    # multiply direct recharge by area to get m3/day
    direct_recharge = direct_recharge * area / 1000
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

    # # create xarray data array for total recharge
    # da_recharge = da_indirect_recharge + da_direct_recharge
    # # resample to monthly
    # da_recharge_monthly = da_recharge.resample(time="ME").sum()
    # # resample to annual
    # da_recharge_annual = da_recharge.resample(time="YE").sum()
    # df_recharge_monthly = df_indirect_recharge_monthly + df_direct_recharge_monthly
    # df_recharge_monthly.columns = ["recharge"]
    # df_recharge_annual = df_indirect_recharge_annual + df_direct_recharge_annual
    # df_recharge_annual.columns = ["recharge"]

    # load well extraction
    click.echo("Loading well extraction...")
    # save daily drinking water well extraction to csv
    file = base_path.parent / "input" / "drinking_water_well_extraction_daily.csv"
    df_drinking_water_well_extraction_daily = pd.read_csv(file, sep=";", index_col=0)
    df_drinking_water_well_extraction_daily.index = pd.to_datetime(df_drinking_water_well_extraction_daily.index, format="%Y-%m-%d")
    # resample to monthly
    df_drinking_water_well_extraction_monthly = df_drinking_water_well_extraction_daily.resample("ME").sum()
    # resample to annual
    df_drinking_water_well_extraction_annual = df_drinking_water_well_extraction_daily.resample("YE").sum()

    # calculate the extraction balance as the difference between the drinking water well extraction and the indirect recharge multiplied by 0.3 (assuming that 30% of the indirect recharge is available for extraction)
    df_extraction_balance_monthly = pd.DataFrame(index=df_drinking_water_well_extraction_monthly.index, columns=["extraction_balance"])
    df_extraction_balance_monthly["extraction_balance"] = df_drinking_water_well_extraction_monthly["well_extraction"] - (df_direct_recharge_monthly["direct_recharge"] * 0.3)
    df_extraction_balance_annual = pd.DataFrame(index=df_drinking_water_well_extraction_annual.index, columns=["extraction_balance"])
    df_extraction_balance_annual["extraction_balance"] = df_drinking_water_well_extraction_annual["well_extraction"] - (df_direct_recharge_annual["direct_recharge"] * 0.3)

    # make figures directory if it does not exist
    figures_dir = base_path.parent / "figures" / "gw_extraction_balance"
    figures_dir.mkdir(exist_ok=True)

    # # plot monthly direct and indirect recharge using stacked bar plot use blue for direct recharge and purple for indirect recharge 
    # fig, ax = plt.subplots(figsize=(6, 2))
    # df_recharge_monthly_stacked = df_direct_recharge_monthly.copy()
    # df_recharge_monthly_stacked["indirect_recharge"] = df_indirect_recharge_monthly["indirect_recharge"]
    # df_recharge_monthly_stacked["direct_recharge"] = df_recharge_monthly_stacked["direct_recharge"]/1e6  # convert to million m3/month
    # df_recharge_monthly_stacked["indirect_recharge"] = df_recharge_monthly_stacked["indirect_recharge"]/1e6  # convert to million m3/month
    # df_recharge_monthly_stacked.plot(kind="bar", stacked=True, ax=ax, color=["blue", "purple"])
    # ax.set_xlabel("Jahr")
    # ax.set_ylabel("Recharge [Mio. m³/Monat]")
    # fig.tight_layout()
    # fig.savefig(figures_dir / "recharge_monthly_stacked.png", dpi=300)

    # # plot annual direct and indirect recharge using stacked bar plot use blue for direct recharge and purple for indirect recharge
    # fig, ax = plt.subplots(figsize=(6, 2))
    # df_recharge_annual_stacked = df_direct_recharge_annual.copy()
    # df_recharge_annual_stacked["indirect_recharge"] = df_indirect_recharge_annual["indirect_recharge"]
    # df_recharge_annual_stacked["direct_recharge"] = df_recharge_annual_stacked["direct_recharge"]/1e6  # convert to million m3/year
    # df_recharge_annual_stacked["indirect_recharge"] = df_recharge_annual_stacked["indirect_recharge"]/1e6  # convert to million m3/year
    # df_recharge_annual_stacked.plot(kind="bar", stacked=True, ax=ax, color=["blue", "purple"])
    # ax.set_xlabel("Jahr")
    # ax.set_ylabel("Recharge [Mio. m³/Jahr]")
    # fig.tight_layout()
    # fig.savefig(figures_dir / "recharge_annual_stacked.png", dpi=300)

    # plot the monthly extraction balance using bar plot, make bars with negative values orange and bars with positive values blue
    fig, ax = plt.subplots(figsize=(6, 2))
    colors = ["orange" if x < 0 else "blue" for x in df_extraction_balance_monthly["extraction_balance"]]
    ax.bar(df_extraction_balance_monthly.index, df_extraction_balance_monthly["extraction_balance"]/1e6, color=colors, width=20)
    # rotate xticklabels by 45 degrees
    plt.xticks(rotation=25)
    ax.set_xlabel("Zeit")
    ax.set_ylabel("GW-Entnahmebilanz\n[Mio. m³/Monat]")
    fig.tight_layout()
    fig.savefig(figures_dir / "extraction_balance_monthly.png", dpi=300)

    # plot the annual extraction balance using bar plot, make bars with negative values orange and bars with positive values blue
    fig, ax = plt.subplots(figsize=(6, 2))
    colors = ["orange" if x < 0 else "blue" for x in df_extraction_balance_annual["extraction_balance"]]
    ax.bar(df_extraction_balance_annual.index.year, df_extraction_balance_annual["extraction_balance"]/1e6, color=colors, width=20)
    # reformat xticklabels to show only the year
    ax.set_xticklabels(df_extraction_balance_annual.index.year, rotation=25)
    ax.set_xlabel("Jahr")
    ax.set_ylabel("GW-Entnahmebilanz\n[m³/Jahr]")
    fig.tight_layout()
    fig.savefig(figures_dir / "extraction_balance_annual.png", dpi=300)

    # plot the long-term balance of the extraction balance using a bar plot, make bars with negative values orange and bars with positive values blue
    long_term_balance = df_extraction_balance_annual["extraction_balance"].sum()
    fig, ax = plt.subplots(figsize=(2, 2))
    color = "orange" if long_term_balance < 0 else "blue"
    ax.bar([""], [long_term_balance/1e6], color=color)
    ax.set_ylabel("GW-Entnahmebilanz\n[Mio. m³]")
    fig.tight_layout()
    fig.savefig(figures_dir / "extraction_balance_long_term.png", dpi=300)

    return

if __name__ == "__main__":
    main()
