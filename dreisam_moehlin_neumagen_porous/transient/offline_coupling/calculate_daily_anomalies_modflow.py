from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
import gc
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
@click.option("-a", "--area", type=click.Choice(["dmn", "wsg_hausen", "wsg_zartener_becken", "wsg_boetzingen", "wsg_breisach", "wsg_ebringen", "wsg_eichstetten", "wsg_gottenheim", "wsg_krozinger_berg", "wsg_march", "wsg_schlatt", "wsg_tuniberg", "wsg_umkirch"]), default="dmn", help="Area to process")
@click.command("main", short_help="Calculate daily anomalies for MODFLOW output")
def main(model_run, area):
    base_path = Path(__file__).parent
    # base_path_output = base_path / "output"
    # base_path_output = Path("/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline-coupling/output")

    base = "base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"
    
    stress_test_scenarios = ["base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction",
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

    df_metrics = pd.DataFrame(columns=["scenario", "area", "time", "variable", "unit", "metric", "value"])
    df_anomaly_metrics_abs = pd.DataFrame(columns=["scenario", "area", "time", "variable", "unit", "metric", "value"])
    df_anomaly_metrics_rel = pd.DataFrame(columns=["scenario", "area", "time", "variable", "unit", "metric", "value"])

    if area == "dmn":
        mask = ds_params["mask_porous_aquifer"].values
        file = base_path.parent / "input" / f"dmn_25m.tif"
        with rasterio.open(file) as src:
            mask25 = src.read(1)
            mask25 = np.where(mask25 == 1, True, False)
    else:
        file = base_path.parent / "input" / f"{area}_.tif"
        with rasterio.open(file) as src:
            mask = src.read(1)
            mask = np.where(mask == 1, True, False)

        file = base_path.parent / "input" / f"{area}_25m.tif"
        with rasterio.open(file) as src:
            mask25 = src.read(1)
            mask25 = np.where(mask25 == 1, True, False)

    click.echo(f"Processing area {area}...")
    click.echo(f"Processing scenario {base}...")
    # load the groundwater depths for the base scenario
    click.echo("Loading groundwater depths (base)...")
    ll_gw_depths = []
    for year in years:
        output_file = base_path / "output" / f"modflow_{base}" / f"gw_depth_run{model_run}_year{year}.nc"
        # output_file = base_path_output / f"{base}" / f"gw_depth_run{model_run}_year{year}.nc"
        ds_gw_depths = xr.open_dataset(output_file, engine="h5netcdf")
        gw_depths_year = ds_gw_depths["depth"].values[:, 1, :, :]
        gw_depths_year = np.where(mask[np.newaxis, :, :], gw_depths_year, np.nan)
        ll_gw_depths.append(gw_depths_year)
    gw_depths = np.concatenate(ll_gw_depths, axis=0)
    # create xarray data array for groundwater depths
    da_gw_depths_base = xr.DataArray(
        data=gw_depths,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_gw_depths["lat"].values,
            "x": ds_gw_depths["lon"].values,
        },
    )
    del gw_depths, ll_gw_depths, gw_depths_year, ds_gw_depths
    value = np.nanmean(da_gw_depths_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "average", "value": value}
    value = np.nanpercentile(da_gw_depths_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "5th_percentile", "value": value}
    value = np.nanpercentile(da_gw_depths_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "25th_percentile", "value": value}
    value = np.nanpercentile(da_gw_depths_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "median", "value": value}
    value = np.nanpercentile(da_gw_depths_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "75th_percentile", "value": value}
    value = np.nanpercentile(da_gw_depths_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "95th_percentile", "value": value}

    for year in years:
        click.echo(f"Processing year {year}...")
        gw_depths_year = da_gw_depths_base.sel(time=str(year)).values

        value = np.nanmean(gw_depths_year.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "average", "value": value}
        value = np.nanpercentile(gw_depths_year.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(gw_depths_year.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(gw_depths_year.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "median", "value": value}
        value = np.nanpercentile(gw_depths_year.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(gw_depths_year.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "95th_percentile", "value": value}

    del gw_depths_year

    # load the indirect recharge
    click.echo("Loading indirect recharge...")
    ll_indirect_recharge = []
    for year in years:
        output_file = base_path / "output" / f"modflow_{base}" / f"indirect_recharge_run{model_run}_year{year}.nc"
        # output_file = base_path_output / f"{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
        ds_indirect_recharge = xr.open_dataset(output_file, engine="h5netcdf")
        indirect_recharge_year = ds_indirect_recharge["indirect_recharge"].values * 86400  # convert from m3/s to m3/day
        ds_indirect_recharge.close()
        indirect_recharge_year[indirect_recharge_year >= 0] = np.nan  # set positive values to zero
        indirect_recharge_year = np.abs(indirect_recharge_year)
        indirect_recharge_year = np.where(mask[np.newaxis, :, :], indirect_recharge_year, np.nan)
        ll_indirect_recharge.append(indirect_recharge_year)
    indirect_recharge = np.concatenate(ll_indirect_recharge, axis=0)
    # create xarray data array for indirect recharge
    da_indirect_recharge_base = xr.DataArray(
        data=indirect_recharge,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_indirect_recharge["lat"].values,
            "x": ds_indirect_recharge["lon"].values,
        },
    )
    del indirect_recharge, ll_indirect_recharge, indirect_recharge_year, ds_indirect_recharge
    value = np.nanmean(da_indirect_recharge_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "average", "value": value}
    value = np.nanpercentile(da_indirect_recharge_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "5th_percentile", "value": value}
    value = np.nanpercentile(da_indirect_recharge_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "25th_percentile", "value": value}
    value = np.nanpercentile(da_indirect_recharge_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "median", "value": value}
    value = np.nanpercentile(da_indirect_recharge_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "75th_percentile", "value": value}
    value = np.nanpercentile(da_indirect_recharge_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": value}

    for year in years:
        click.echo(f"Processing year {year}...")
        indirect_recharge_year = da_indirect_recharge_base.sel(time=str(year)).values

        value = np.nanmean(indirect_recharge_year.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "average", "value": value}
        value = np.nanpercentile(indirect_recharge_year.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(indirect_recharge_year.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(indirect_recharge_year.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "median", "value": value}
        value = np.nanpercentile(indirect_recharge_year.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(indirect_recharge_year.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": value}

    del indirect_recharge_year

    # load direct recharge
    click.echo("Loading direct recharge...")
    ll_direct_recharge = []
    for year in years:
        base_path_roger = base_path.parent.parent.parent.parent / "roger"
        output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"recharge_{base}_year{year}.nc"
        ds_direct_recharge = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
        _direct_recharge_year = ds_direct_recharge["recharge"].values
        ds_direct_recharge.close()
        _direct_recharge_year[_direct_recharge_year < 0] = 0  # set negative values to zero
        _direct_recharge_year[_direct_recharge_year > 100] = 100  # set values above 100 mm/day to 100 mm/day
        ll_direct_recharge.append(_direct_recharge_year)
    direct_recharge = np.concatenate(ll_direct_recharge, axis=0)
    # convert from mm/day to m3/day
    # get the area of each grid cell in m2
    _area = 25 * 25  # 25 m x 25 m grid cells
    # multiply direct recharge by area to get m3/day
    direct_recharge = direct_recharge * _area / 1000
    # create xarray data array for direct recharge
    da_direct_recharge_base = xr.DataArray(
        data=direct_recharge,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_direct_recharge["y"].values,
            "x": ds_direct_recharge["x"].values,
        },
    )
    del direct_recharge, ll_direct_recharge, _direct_recharge_year, ds_direct_recharge
    value = np.nanmean(da_direct_recharge_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "average", "value": value}
    value = np.nanpercentile(da_direct_recharge_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time":	"overall ", "variable":	"direct_recharge ", "unit": "m3/day ", "metric": "5th_percentile ", "value": value}
    value = np.nanpercentile(da_direct_recharge_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time":	"overall ", "variable":	"direct_recharge ", "unit": "m3/day ", "metric": "25th_percentile ", "value": value}
    value = np.nanpercentile(da_direct_recharge_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time":	"overall ", "variable":	"direct_recharge ", "unit": "m3/day ", "metric": "50th_percentile ", "value": value}
    value = np.nanpercentile(da_direct_recharge_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time":	"overall ", "variable":	"direct_recharge ", "unit": "m3/day ", "metric": "75th_percentile ", "value": value}
    value = np.nanpercentile(da_direct_recharge_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": value}

    for year in years:
        direct_recharge = da_direct_recharge_base.sel(time=str(year)).values
        value = np.nanmean(direct_recharge.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "average", "value": value}
        value = np.nanpercentile(direct_recharge.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario ": base, ("area"): area, "time":	f"{year}", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"5th_percentile ", ("value"): 	value}
        value = np.nanpercentile(direct_recharge.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario ": base, ("area"): area, "time":	f"{year}", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"25th_percentile ", ("value"): 	value}
        value = np.nanpercentile(direct_recharge.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario ": base, ("area"): area, "time":	f"{year}", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"50th_percentile ", ("value"): 	value}
        value = np.nanpercentile(direct_recharge.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario ": base, ("area"): area, "time":	f"{year}", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"75th_percentile ", ("value"): 	value}
        value = np.nanpercentile(direct_recharge.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": value}

    del direct_recharge

    # load well extraction
    click.echo("Loading well extraction...")
    ll_well_extraction = []
    for year in years:
        output_file = base_path / "output" / f"modflow_{base}" / f"well_extraction_run{model_run}_year{year}.nc"
        ds_well_extraction = xr.open_dataset(output_file, engine="h5netcdf")
        well_extraction_year = ds_well_extraction["well_extraction"].values
        ds_well_extraction.close()
        well_extraction_year = np.where(mask[np.newaxis, :, :], well_extraction_year, 0)
        ll_well_extraction.append(well_extraction_year)
    well_extraction = np.concatenate(ll_well_extraction, axis=0)
    well_extraction = np.where(well_extraction <= 0, np.nan, well_extraction)  # set negative values to nan
    # create xarray data array for well extraction
    da_well_extraction_base = xr.DataArray(
        data=well_extraction,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_well_extraction["lat"].values,
            "x": ds_well_extraction["lon"].values,
        },
    )
    del well_extraction, ll_well_extraction, well_extraction_year, ds_well_extraction
    value = np.nanmean(da_well_extraction_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "average", "value": value}
    value = np.nanpercentile(da_well_extraction_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "5th_percentile", "value": value}
    value = np.nanpercentile(da_well_extraction_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "25th_percentile", "value": value}
    value = np.nanpercentile(da_well_extraction_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "median", "value": value}
    value = np.nanpercentile(da_well_extraction_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "75th_percentile", "value": value}
    value = np.nanpercentile(da_well_extraction_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "95th_percentile", "value": value}

    for year in years:
        click.echo(f"Processing year {year}...")
        output_file = base_path / "output" / f"modflow_{base}" / f"well_extraction_run{model_run}_year{year}.nc"
        ds_well_extraction = xr.open_dataset(output_file, engine="h5netcdf")
        well_extraction_year = ds_well_extraction["well_extraction"].values
        ds_well_extraction.close()
        well_extraction_year = np.where(mask[np.newaxis, :, :], well_extraction_year, 0)

        value = np.nanmean(well_extraction_year.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/day", "metric": "average", "value": value}
        value = np.nanpercentile(well_extraction_year.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/day", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(well_extraction_year.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/day", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(well_extraction_year.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/day", "metric": "median", "value": value}
        value = np.nanpercentile(well_extraction_year.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/day", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(well_extraction_year.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/day", "metric": "95th_percentile", "value": value}


    # save the metrics to csv
    output_file = base_path / "output" / f"daily_values_run{model_run}_{area}_modflow.csv"
    df_metrics.to_csv(output_file, index=False, sep=";")

    for stress_test_scenario in stress_test_scenarios:
        click.echo(f"Processing scenario {stress_test_scenario}...")
        # load the groundwater depths for the stress test scenario
        click.echo("Loading groundwater depths (stress test)...")
        ll_gw_depths = []
        for year in years:
            click.echo(f"Processing year {year}...")
            output_file = base_path / "output" / f"modflow_{stress_test_scenario}" / f"gw_depth_run{model_run}_year{year}.nc"
            # output_file = base_path_output / f"{stress_test_scenario}" / f"gw_depth_run{model_run}_year{year}.nc"
            ds_gw_depths = xr.open_dataset(output_file, engine="h5netcdf")
            gw_depths_year = ds_gw_depths["depth"].values[:, 1, :, :]
            gw_depths_year = np.where(mask[np.newaxis, :, :], gw_depths_year, np.nan)
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
        del gw_depths, ll_gw_depths, gw_depths_year, ds_gw_depths
        click.echo("Calculating groundwater anomalies...")
        value = np.nanmean(da_gw_depths.values.flatten())
        value_base = np.nanmean(da_gw_depths_base.values.flatten())
        anomaly_abs = (value - value_base) * (-1)
        anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_gw_depths.values.flatten(), 5)
        value_base = np.nanpercentile(da_gw_depths_base.values.flatten(), 5)
        anomaly_abs = (value - value_base) * (-1)
        anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "5th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_gw_depths.values.flatten(), 25)
        value_base = np.nanpercentile(da_gw_depths_base.values.flatten(), 25)
        anomaly_abs = (value - value_base) * (-1)
        anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_gw_depths.values.flatten(), 50)
        value_base = np.nanpercentile(da_gw_depths_base.values.flatten(), 50)
        anomaly_abs = (value - value_base) * (-1)
        anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "median", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_gw_depths.values.flatten(), 75)
        value_base = np.nanpercentile(da_gw_depths_base.values.flatten(), 75)
        anomaly_abs = (value - value_base) * (-1)
        anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "75th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "75th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_gw_depths.values.flatten(), 95)
        value_base = np.nanpercentile(da_gw_depths_base.values.flatten(), 95)
        anomaly_abs = (value - value_base) * (-1)
        anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "m", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}

        for year in years:
            click.echo(f"Processing year {year}...")
            gw_depths_base_year = da_gw_depths_base.sel(time=str(year)).values
            gw_depths_year = da_gw_depths.sel(time=str(year)).values

            value = np.nanmean(gw_depths_year.flatten())
            value_base = np.nanmean(gw_depths_base_year.flatten())
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(gw_depths_year.flatten(), 5)
            value_base = np.nanpercentile(gw_depths_base_year.flatten(), 5)
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "5th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(gw_depths_year.flatten(), 25)
            value_base = np.nanpercentile(gw_depths_base_year.flatten(), 25)
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(gw_depths_year.flatten(), 50)
            value_base = np.nanpercentile(gw_depths_base_year.flatten(), 50)
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "median", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(gw_depths_year.flatten(), 75)
            value_base = np.nanpercentile(gw_depths_base_year.flatten(), 75)
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(gw_depths_year.flatten(), 95)
            value_base = np.nanpercentile(gw_depths_base_year.flatten(), 95)
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "m", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "gw_depth", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}

        del da_gw_depths, gw_depths_year

        # load the indirect recharge
        click.echo("Loading indirect recharge...")
        ll_indirect_recharge = []
        for year in years:
            output_file = base_path / "output" / f"modflow_{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
            # output_file = base_path_output / f"{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
            ds_indirect_recharge = xr.open_dataset(output_file, engine="h5netcdf")
            indirect_recharge_year = ds_indirect_recharge["indirect_recharge"].values * 86400  # convert from m3/s to m3/day
            ds_indirect_recharge.close()
            indirect_recharge_year[indirect_recharge_year >= 0] = np.nan  # set positive values to zero
            indirect_recharge_year = np.abs(indirect_recharge_year)
            indirect_recharge_year = np.where(mask[np.newaxis, :, :], indirect_recharge_year, np.nan)
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
        del indirect_recharge, ll_indirect_recharge, indirect_recharge_year, ds_indirect_recharge
        click.echo("Calculating indirect recharge anomalies...")
        value = np.nanmean(da_indirect_recharge.values.flatten())
        value_base = np.nanmean(da_indirect_recharge_base.values.flatten())
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_indirect_recharge.values.flatten(), 5)
        value_base = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values.flatten()), 5)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "5th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_indirect_recharge.values.flatten(), 25)
        value_base = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values.flatten()), 25)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(np.nanmean(da_indirect_recharge.values.flatten()), 50)
        value_base = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values.flatten()), 50)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "median", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_indirect_recharge.values.flatten(), 75)
        value_base = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values.flatten()), 75)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "75th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "75th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(np.nanmean(da_indirect_recharge.values.flatten()), 95)
        value_base = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values.flatten()), 95)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}

        for year in years:
            click.echo(f"Processing year {year}...")
            indirect_recharge_base_year = da_indirect_recharge_base.sel(time=str(year)).values
            indirect_recharge_year = da_indirect_recharge.sel(time=str(year)).values

            value = np.nanmean(indirect_recharge_year.flatten())
            value_base = np.nanmean(indirect_recharge_base_year.flatten())
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(indirect_recharge_year.flatten(), 5)
            value_base = np.nanpercentile(indirect_recharge_base_year.flatten(), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "5th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(indirect_recharge_year.flatten(), 25)
            value_base = np.nanpercentile(indirect_recharge_base_year.flatten(), 25)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(np.nanmean(indirect_recharge_year.flatten()), 50)
            value_base = np.nanpercentile(np.nanmean(indirect_recharge_base_year.flatten()), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "median", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(np.nanmean(indirect_recharge_year.flatten()), 75)
            value_base = np.nanpercentile(np.nanmean(indirect_recharge_base_year.flatten()), 75)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(np.nanmean(indirect_recharge_year.flatten()), 95)
            value_base = np.nanpercentile(np.nanmean(indirect_recharge_base_year.flatten()), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "indirect_recharge", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}

        del da_indirect_recharge, indirect_recharge_year

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
            ll_direct_recharge.append(_direct_recharge_year)
        direct_recharge = np.concatenate(ll_direct_recharge, axis=0)
        # convert from mm/day to m3/day
        # get the area of each grid cell in m2
        _area = 25 * 25  # 25 m x 25 m grid cells
        # multiply direct recharge by area to get m3/day
        direct_recharge = direct_recharge * _area / 1000
        # create xarray data array for direct recharge
        da_direct_recharge = xr.DataArray(
            data=direct_recharge,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_direct_recharge["y"].values,
                "x": ds_direct_recharge["x"].values,
            },
        )
        del direct_recharge, ll_direct_recharge, _direct_recharge_year, ds_direct_recharge
        click.echo("Calculating direct recharge anomalies...")
        value = np.nanmean(da_direct_recharge.values.flatten())
        value_base = np.nanmean(da_direct_recharge_base.values.flatten())
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_direct_recharge.values.flatten(), 5)
        value_base = np.nanpercentile(da_direct_recharge_base.values.flatten(), 5)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "5th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_direct_recharge.values.flatten(), 25)
        value_base = np.nanpercentile(da_direct_recharge_base.values.flatten(), 25)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_direct_recharge.values.flatten(), 50)
        value_base = np.nanpercentile(da_direct_recharge_base.values.flatten(), 50)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "median", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_direct_recharge.values.flatten(), 95)
        value_base = np.nanpercentile(da_direct_recharge_base.values.flatten(), 95)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}

        for year in years:
            click.echo(f"Processing year {year}...")
            direct_recharge_base = da_direct_recharge_base.sel(time=f"{year}").values
            direct_recharge = da_direct_recharge.sel(time=f"{year}").values

            value = np.nanmean(direct_recharge.flatten())
            value_base = np.nanmean(direct_recharge_base.flatten())
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(direct_recharge.flatten(), 5)
            value_base = np.nanpercentile(direct_recharge_base.flatten(), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "5th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(direct_recharge.flatten(), 25)
            value_base = np.nanpercentile(direct_recharge_base.flatten(), 25)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(direct_recharge.flatten(), 50)
            value_base = np.nanpercentile(direct_recharge_base.flatten(), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "median", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(direct_recharge.flatten(), 75)
            value_base = np.nanpercentile(direct_recharge_base.flatten(), 75)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(direct_recharge.flatten(), 95)
            value_base = np.nanpercentile(direct_recharge_base.flatten(), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}

        del da_direct_recharge, direct_recharge
        
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
        well_extraction = np.where(well_extraction <= 0, np.nan, well_extraction)  # set negative values to nan
        # create xarray data array for well extraction
        da_well_extraction = xr.DataArray(
            data=well_extraction,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_well_extraction["lat"].values,
                "x": ds_well_extraction["lon"].values,
            },
        )
        del well_extraction, ll_well_extraction, well_extraction_year, ds_well_extraction
        click.echo("Calculating well extraction anomalies...")
        value = np.nanmean(da_well_extraction.values.flatten())
        value_base = np.nanmean(da_well_extraction_base.values.flatten())
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_well_extraction.values.flatten(), 5)
        value_base = np.nanpercentile(da_well_extraction_base.values.flatten(), 5)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area,("variable"): 'well_extraction', ("unit"): 'm3/s', ("metric"): '5th_percentile', ("value"): anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "%",("metric"): '5th_percentile', ("value"): anomaly_rel}
        value = np.nanpercentile(da_well_extraction.values.flatten(), 25)
        value_base = np.nanpercentile(da_well_extraction_base.values.flatten(), 25)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_well_extraction.values.flatten(), 50)
        value_base = np.nanpercentile(da_well_extraction_base.values.flatten(), 50)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "median", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_well_extraction.values.flatten(), 75)
        value_base = np.nanpercentile(da_well_extraction_base.values.flatten(), 75)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "75th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "75th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_well_extraction.values.flatten(), 95)
        value_base = np.nanpercentile(da_well_extraction_base.values.flatten(), 95)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/s", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "well_extraction", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}

        for year in years:
            click.echo(f"Processing year {year}...")
            well_extraction_base_year = da_well_extraction_base.sel(time=f"{year}").values
            well_extraction_year = da_well_extraction.sel(time=f"{year}").values

            value = np.nanmean(well_extraction_year.flatten())
            value_base = np.nanmean(well_extraction_base_year.flatten())
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(well_extraction_year.flatten(), 5)
            value_base = np.nanpercentile(well_extraction_base_year.flatten(), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": 'well_extraction', ("unit"): 'm3/s', ("metric"): '5th_percentile', ("value"): anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "%",("metric"): '5th_percentile', ("value"): anomaly_rel}
            value = np.nanpercentile(well_extraction_year.flatten(), 25)
            value_base = np.nanpercentile(well_extraction_base_year.flatten(), 25)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(well_extraction_year.flatten(), 50)
            value_base = np.nanpercentile(well_extraction_base_year.flatten(), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "median", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(well_extraction_year.flatten(), 75)
            value_base = np.nanpercentile(well_extraction_base_year.flatten(), 75)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(well_extraction_year.flatten(), 95)
            value_base = np.nanpercentile(well_extraction_base_year.flatten(), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "m3/s", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "well_extraction", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}

        del da_well_extraction
        gc.collect()
        df_metrics = df_metrics.copy()
        df_anomaly_metrics_abs = df_anomaly_metrics_abs.copy()
        df_anomaly_metrics_rel = df_anomaly_metrics_rel.copy()

        # save the metrics to csv
        output_file = base_path / "output" / f"daily_values_run{model_run}_{area}_modflow.csv"
        df_metrics.to_csv(output_file, index=False, sep=";")
        output_file = base_path / "output" / f"daily_anomalies_abs_run{model_run}_{area}_modflow.csv"
        df_anomaly_metrics_abs.to_csv(output_file, index=False, sep=";")
        output_file = base_path / "output" / f"daily_anomalies_rel_run{model_run}_{area}_modflow.csv"
        df_anomaly_metrics_rel.to_csv(output_file, index=False, sep=";")

    return

if __name__ == "__main__":
    main()
