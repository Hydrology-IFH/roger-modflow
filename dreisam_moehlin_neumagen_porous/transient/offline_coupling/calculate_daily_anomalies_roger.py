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
@click.command("main", short_help="Evaluate the transient simulation")
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

    df_metrics = pd.DataFrame(columns=["scenario", "area", "time", "variable", "unit", "metric", "value"])
    df_anomaly_metrics_abs = pd.DataFrame(columns=["scenario", "area", "time", "variable", "unit", "metric", "value"])
    df_anomaly_metrics_rel = pd.DataFrame(columns=["scenario", "area", "time", "variable", "unit", "metric", "value"])

    if area == "dmn":
        with rasterio.open(file) as src:
            mask25 = src.read(1)
            mask25 = np.where(mask25 == 1, True, False)
    else:
        file = base_path.parent / "input" / f"{area}_25m.tif"
        with rasterio.open(file) as src:
            mask25 = src.read(1)
            mask25 = np.where(mask25 == 1, True, False)

    click.echo(f"Processing area {area}...")
    click.echo(f"Processing scenario {base}...")
    # load potential evapotranspiration
    click.echo("Loading potential evapotranspiration...")
    ll_potential_evapotranspiration = []
    for year in years:
        base_path_roger = base_path.parent.parent.parent.parent / "roger"
        output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"potential_evapotranspiration_{base}_year{year}.nc"
        ds_potential_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
        _potential_evapotranspiration_year = ds_potential_evapotranspiration["potential_evapotranspiration"].values
        ds_potential_evapotranspiration.close()
        _potential_evapotranspiration_year[_potential_evapotranspiration_year < 0] = 0  # set negative values to zero
        ll_potential_evapotranspiration.append(_potential_evapotranspiration_year)
    potential_evapotranspiration = np.concatenate(ll_potential_evapotranspiration, axis=0)
    # create xarray data array for potential evapotranspiration
    da_potential_evapotranspiration_base = xr.DataArray(
        data=potential_evapotranspiration,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_potential_evapotranspiration["y"].values,
            "x": ds_potential_evapotranspiration["x"].values,
        },
    )
    del potential_evapotranspiration, ll_potential_evapotranspiration, _potential_evapotranspiration_year, ds_potential_evapotranspiration
    value = np.nanmean(da_potential_evapotranspiration_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "average", "value": value}
    value = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": value}
    value = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": value}
    value = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "median", "value": value}
    value = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "75th_percentile", "value": value}
    value = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": value}

    for year in years:
        click.echo(f"Processing year {year}...")
        potential_evapotranspiration = da_potential_evapotranspiration_base.sel(time=str(year)).values
        value = np.nanmean(potential_evapotranspiration.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "average", "value": value}
        value = np.nanpercentile(potential_evapotranspiration.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(potential_evapotranspiration.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(potential_evapotranspiration.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "median", "value": value}
        value = np.nanpercentile(potential_evapotranspiration.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(potential_evapotranspiration.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": value}

    # load actual evapotranspiration
    click.echo("Loading actual evapotranspiration...")
    ll_actual_evapotranspiration = []
    for year in years:
        base_path_roger = base_path.parent.parent.parent.parent / "roger"
        output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"actual_evapotranspiration_{base}_year{year}.nc"
        ds_actual_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
        _actual_evapotranspiration_year = ds_actual_evapotranspiration["actual_evapotranspiration"].values
        ds_actual_evapotranspiration.close()
        _actual_evapotranspiration_year[_actual_evapotranspiration_year < 0] = 0  # set negative values to zero
        ll_actual_evapotranspiration.append(_actual_evapotranspiration_year)
    actual_evapotranspiration = np.concatenate(ll_actual_evapotranspiration, axis=0)
    # create xarray data array for actual evapotranspiration
    da_actual_evapotranspiration_base = xr.DataArray(
        data=actual_evapotranspiration,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_actual_evapotranspiration["y"].values,
            "x": ds_actual_evapotranspiration["x"].values,
        },
    )
    del actual_evapotranspiration, ll_actual_evapotranspiration, _actual_evapotranspiration_year, ds_actual_evapotranspiration
    value = np.nanmean(da_actual_evapotranspiration_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "average", "value": value}
    value = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": value}
    value = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": value}
    value = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "median", "value": value}
    value = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "75th_percentile", "value": value}
    value = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": value}

    for year in years:
        click.echo(f"Processing year {year}...")
        actual_evapotranspiration = da_actual_evapotranspiration_base.sel(time=str(year)).values

        value = np.nanmean(actual_evapotranspiration.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "average", "value": value}
        value = np.nanpercentile(actual_evapotranspiration.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(actual_evapotranspiration.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(actual_evapotranspiration.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "median", "value": value}
        value = np.nanpercentile(actual_evapotranspiration.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(actual_evapotranspiration.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": value}

    # load precipitation
    click.echo("Loading precipitation...")
    ll_precipitation = []
    for year in years:
        base_path_roger = base_path.parent.parent.parent.parent / "roger"
        output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"precipitation_{base}_year{year}.nc"
        ds_precipitation = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
        _precipitation_year = ds_precipitation["precipitation"].values
        ds_precipitation.close()
        _precipitation_year[_precipitation_year < 0] = 0  # set negative values to zero
        ll_precipitation.append(_precipitation_year)
    precipitation = np.concatenate(ll_precipitation, axis=0)
    # create xarray data array for precipitation
    da_precipitation_base = xr.DataArray(
        data=precipitation,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_precipitation["y"].values,
            "x": ds_precipitation["x"].values,
        },
    )
    del precipitation, ll_precipitation, _precipitation_year, ds_precipitation
    value = np.nanmean(da_precipitation_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "average", "value": value}
    value = np.nanpercentile(da_precipitation_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "5th_percentile", "value": value}
    value = np.nanpercentile(da_precipitation_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "25th_percentile", "value": value}
    value = np.nanpercentile(da_precipitation_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "median", "value": value}
    value = np.nanpercentile(da_precipitation_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "75th_percentile", "value": value}
    value = np.nanpercentile(da_precipitation_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm", "metric": "95th_percentile", "value": value}

    for year in years:
        precipitation = da_precipitation_base.sel(time=str(year)).values

        value = np.nanmean(precipitation.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "average", "value": value}
        value = np.nanpercentile(precipitation.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(precipitation.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(precipitation.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "median", "value": value}
        value = np.nanpercentile(precipitation.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(precipitation.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "95th_percentile", "value": value}

    # load air temperature
    click.echo("Loading air temperature...")
    ll_air_temperature = []
    for year in years:
        click.echo(f"Processing year {year}...")
        base_path_roger = base_path.parent.parent.parent.parent / "roger"
        output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"ta_{base}_year{year}.nc"
        # _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
        # output_file = base_path_output / f"{stress_test_scenario}" / f"recharge_{_stress_test_scenario}_year{year}.nc"
        ds_air_temperature = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
        _air_temperature_year = ds_air_temperature["ta"].values
        _air_temperature_year = np.where(_air_temperature_year < -50, np.nan, _air_temperature_year)
        ds_air_temperature.close()
        ll_air_temperature.append(_air_temperature_year)
    air_temperature = np.concatenate(ll_air_temperature, axis=0)
    # create xarray data array for air temperature
    da_air_temperature_base = xr.DataArray(
        data=air_temperature,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_air_temperature["y"].values,
            "x": ds_air_temperature["x"].values,
        },
    )
    del air_temperature, ll_air_temperature, _air_temperature_year, ds_air_temperature
    value = np.nanmean(da_air_temperature_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "average", "value": value}
    value = np.nanpercentile(da_air_temperature_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "5th_percentile", "value": value}
    value = np.nanpercentile(da_air_temperature_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "25th_percentile", "value": value}
    value = np.nanpercentile(da_air_temperature_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "median", "value": value}
    value = np.nanpercentile(da_air_temperature_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "75th_percentile", "value": value}
    value = np.nanpercentile(da_air_temperature_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "95th_percentile", "value": value}

    for year in years:
        click.echo(f"Processing year {year}...")
        air_temperature = da_air_temperature_base.sel(time=str(year)).values

        value = np.nanmean(air_temperature.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "average", "value": value}
        value = np.nanpercentile(air_temperature.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(air_temperature.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(air_temperature.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "median", "value": value}
        value = np.nanpercentile(air_temperature.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(air_temperature.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "95th_percentile", "value": value}

    # save the metrics to csv
    output_file = base_path / "output" / f"daily_values_run{model_run}_{area}_roger.csv"
    df_metrics.to_csv(output_file, index=False, sep=";")

    for stress_test_scenario in stress_test_scenarios:
        click.echo(f"Processing scenario {stress_test_scenario}...")
        # load potential evapotranspiration
        click.echo("Loading potential evapotranspiration...")
        ll_potential_evapotranspiration = []
        for year in years:
            _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"potential_evapotranspiration_{_stress_test_scenario}_year{year}.nc"
            ds_potential_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _potential_evapotranspiration_year = ds_potential_evapotranspiration["potential_evapotranspiration"].values
            # set negative values to zero
            _potential_evapotranspiration_year[_potential_evapotranspiration_year < 0] = 0
            ds_potential_evapotranspiration.close()
            ll_potential_evapotranspiration.append(_potential_evapotranspiration_year)
        potential_evapotranspiration = np.concatenate(ll_potential_evapotranspiration, axis=0)
        # create xarray data array for potential evapotranspiration
        da_potential_evapotranspiration = xr.DataArray(
            data=potential_evapotranspiration,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_potential_evapotranspiration["y"].values,
                "x": ds_potential_evapotranspiration["x"].values,
            },
        )
        del potential_evapotranspiration, ll_potential_evapotranspiration, _potential_evapotranspiration_year, ds_potential_evapotranspiration
        click.echo("Calculating potential evapotranspiration anomalies...")
        value = np.nanmean(da_potential_evapotranspiration.values.flatten())
        value_base = np.nanmean(da_potential_evapotranspiration_base.values.flatten())
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_potential_evapotranspiration.values.flatten(), 5)
        value_base = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 5)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_potential_evapotranspiration.values.flatten(), 25)
        value_base = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 25)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_potential_evapotranspiration.values.flatten(), 50)
        value_base = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 50)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "median", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_potential_evapotranspiration.values.flatten(), 75)
        value_base = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 75)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "75th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "75th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_potential_evapotranspiration.values.flatten(), 95)
        value_base = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 95)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        for year in years:
            click.echo(f"Processing year {year}...")
            potential_evapotranspiration_base = da_potential_evapotranspiration_base.sel(time=f"{year}").values
            potential_evapotranspiration = da_potential_evapotranspiration.sel(time=f"{year}").values

            value = np.nanmean(potential_evapotranspiration.flatten())
            value_base = np.nanmean(potential_evapotranspiration_base.flatten())
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(potential_evapotranspiration.flatten(), 5)
            value_base = np.nanpercentile(potential_evapotranspiration_base.flatten(), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(potential_evapotranspiration.flatten(), 25)
            value_base = np.nanpercentile(potential_evapotranspiration_base.flatten(), 25)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(potential_evapotranspiration.flatten(), 50)
            value_base = np.nanpercentile(potential_evapotranspiration_base.flatten(), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "median", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(potential_evapotranspiration.flatten(), 75)
            value_base = np.nanpercentile(potential_evapotranspiration_base.flatten(), 75)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(potential_evapotranspiration.flatten(), 95)
            value_base = np.nanpercentile(potential_evapotranspiration_base.flatten(), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        del da_potential_evapotranspiration, potential_evapotranspiration

        # load actual evapotranspiration
        click.echo("Loading actual evapotranspiration...")
        ll_actual_evapotranspiration = []
        for year in years:
            _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"actual_evapotranspiration_{_stress_test_scenario}_year{year}.nc"
            ds_actual_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _actual_evapotranspiration_year = ds_actual_evapotranspiration["actual_evapotranspiration"].values
            # set negative values to zero
            _actual_evapotranspiration_year[_actual_evapotranspiration_year < 0] = 0
            ds_actual_evapotranspiration.close()
            ll_actual_evapotranspiration.append(_actual_evapotranspiration_year)
        actual_evapotranspiration = np.concatenate(ll_actual_evapotranspiration, axis=0)
        # create xarray data array for actual evapotranspiration
        da_actual_evapotranspiration = xr.DataArray(
            data=actual_evapotranspiration,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_actual_evapotranspiration["y"].values,
                "x": ds_actual_evapotranspiration["x"].values,
            },
        )
        del actual_evapotranspiration, ll_actual_evapotranspiration, _actual_evapotranspiration_year, ds_actual_evapotranspiration
        click.echo("Calculating actual evapotranspiration anomalies...")
        value = np.nanmean(da_actual_evapotranspiration.values.flatten())
        value_base = np.nanmean(da_actual_evapotranspiration_base.values.flatten())
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_actual_evapotranspiration.values.flatten(), 5)
        value_base = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 5)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_actual_evapotranspiration.values.flatten(), 25)
        value_base = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 25)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_actual_evapotranspiration.values.flatten(), 50)
        value_base = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 50)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "median", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_actual_evapotranspiration.values.flatten(), 95)
        value_base = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 95)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        for year in years:
            click.echo(f"Processing year {year}...")
            actual_evapotranspiration_base = da_actual_evapotranspiration_base.sel(time=f"{year}").values
            actual_evapotranspiration = da_actual_evapotranspiration.sel(time=f"{year}").values

            value = np.nanmean(actual_evapotranspiration.flatten())
            value_base = np.nanmean(actual_evapotranspiration_base.flatten())
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(actual_evapotranspiration.flatten(), 5)
            value_base = np.nanpercentile(actual_evapotranspiration_base.flatten(), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "5th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(actual_evapotranspiration.flatten(), 25)
            value_base = np.nanpercentile(actual_evapotranspiration_base.flatten(), 25)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(actual_evapotranspiration.flatten(), 50)
            value_base = np.nanpercentile(actual_evapotranspiration_base.flatten(), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "median", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(actual_evapotranspiration.flatten(), 75)
            value_base = np.nanpercentile(actual_evapotranspiration_base.flatten(), 75)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(actual_evapotranspiration.flatten(), 95)
            value_base = np.nanpercentile(actual_evapotranspiration_base.flatten(), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        del da_actual_evapotranspiration, actual_evapotranspiration

        # load precipitation
        click.echo("Loading precipitation...")
        ll_precipitation = []
        for year in years:
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"precipitation_{_stress_test_scenario}_year{year}.nc"
            ds_precipitation = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _precipitation_year = ds_precipitation["precipitation"].values
            ds_precipitation.close()
            _precipitation_year[_precipitation_year < 0] = 0  # set negative values to zero
            ll_precipitation.append(_precipitation_year)
        precipitation = np.concatenate(ll_precipitation, axis=0)
        # create xarray data array for precipitation
        da_precipitation = xr.DataArray(
            data=precipitation,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_precipitation["y"].values,
                "x": ds_precipitation["x"].values,
            },
        )
        del precipitation, ll_precipitation, _precipitation_year, ds_precipitation
        click.echo("Calculating precipitation anomalies...")
        value = np.nanmean(da_precipitation.values.flatten())
        value_base = np.nanmean(da_precipitation_base.values.flatten())
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_precipitation.values.flatten(), 5)
        value_base = np.nanpercentile(da_precipitation_base.values.flatten(), 5)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "5th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_precipitation.values.flatten(), 25)
        value_base = np.nanpercentile(da_precipitation_base.values.flatten(), 25)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_precipitation.values.flatten(), 50)
        value_base = np.nanpercentile(da_precipitation_base.values.flatten(), 50)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "median", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_precipitation.values.flatten(), 75)
        value_base = np.nanpercentile(da_precipitation_base.values.flatten(), 75)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "75th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "75th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_precipitation.values.flatten(), 95)
        value_base = np.nanpercentile(da_precipitation_base.values.flatten(), 95)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/day", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        for year in years:
            click.echo(f"Processing year {year}...")
            precipitation_base = da_precipitation_base.sel(time=f"{year}").values
            precipitation = da_precipitation.sel(time=f"{year}").values

            value = np.nanmean(precipitation.flatten())
            value_base = np.nanmean(precipitation_base.flatten())
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(precipitation.flatten(), 5)
            value_base = np.nanpercentile(precipitation_base.flatten(), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "5th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area,"time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(precipitation.flatten(), 25)
            value_base = np.nanpercentile(precipitation_base.flatten(), 25)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(precipitation.flatten(), 50)
            value_base = np.nanpercentile(precipitation_base.flatten(), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "median", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(precipitation.flatten(), 75)
            value_base = np.nanpercentile(precipitation_base.flatten(), 75)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(precipitation.flatten(), 95)
            value_base = np.nanpercentile(precipitation_base.flatten(), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/day", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        del da_precipitation, precipitation

        # load air temperature
        click.echo("Loading air temperature...")
        ll_air_temperature = []
        for year in years:
            _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"ta_{_stress_test_scenario}_year{year}.nc"
            # _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
            # output_file = base_path_output / f"{stress_test_scenario}" / f"air_temperature_{_stress_test_scenario}_year{year}.nc"
            ds_air_temperature = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _air_temperature_year = ds_air_temperature["ta"].values
            _air_temperature_year = np.where(_air_temperature_year < -50, np.nan, _air_temperature_year)
            ds_air_temperature.close()
            ll_air_temperature.append(_air_temperature_year)
        air_temperature = np.concatenate(ll_air_temperature, axis=0)
        # create xarray data array for air temperature
        da_air_temperature = xr.DataArray(
            data=air_temperature,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_air_temperature["y"].values,
                "x": ds_air_temperature["x"].values,
            },
        )
        del air_temperature, ll_air_temperature, _air_temperature_year, ds_air_temperature
        click.echo("Calculating air temperature anomalies...")
        value = np.nanmean(da_air_temperature.values.flatten())
        value_base = np.nanmean(da_air_temperature_base.values.flatten())
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_air_temperature.values.flatten(), 5)
        value_base = np.nanpercentile(da_air_temperature_base.values.flatten(), 5)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "5th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_air_temperature.values.flatten(), 25)
        value_base = np.nanpercentile(da_air_temperature_base.values.flatten(), 25)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_air_temperature.values.flatten(), 50)
        value_base = np.nanpercentile(da_air_temperature_base.values.flatten(), 50)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "median", "value": anomaly_abs}  
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_air_temperature.values.flatten(), 75)
        value_base = np.nanpercentile(da_air_temperature_base.values.flatten(), 75)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "75th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "75th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_air_temperature.values.flatten(), 95)
        value_base = np.nanpercentile(da_air_temperature_base.values.flatten(), 95)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "degC", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "air_temperature", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}

        for year in years:
            click.echo(f"Processing year {year}...")
            air_temperature_base = da_air_temperature_base.sel(time=f"{year}").values
            air_temperature = da_air_temperature.sel(time=f"{year}").values

            value = np.nanmean(air_temperature.flatten())
            value_base = np.nanmean(air_temperature_base.flatten())
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(air_temperature.flatten(), 5)
            value_base = np.nanpercentile(air_temperature_base.flatten(), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "5th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(air_temperature.flatten(), 25)
            value_base = np.nanpercentile(air_temperature_base.flatten(), 25)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(air_temperature.flatten(), 50)
            value_base = np.nanpercentile(air_temperature_base.flatten(), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "median", "value": anomaly_abs}  
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(air_temperature.flatten(), 75)
            value_base = np.nanpercentile(air_temperature_base.flatten(), 75)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(air_temperature.flatten(), 95)
            value_base = np.nanpercentile(air_temperature_base.flatten(), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "degC", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "air_temperature", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}

        del da_air_temperature, air_temperature

        if "_irrigation" in stress_test_scenario:
            # load irrigation
            click.echo("Loading irrigation...")
            ll_irrigation = []
            for year in years:
                _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"irrigation_{_stress_test_scenario}_year{year}.nc"
                ds_irrigation = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                ds_irrigation.close()
                _irrigation_year = np.sum(ds_irrigation["irrigation"].values, axis=0)
                ll_irrigation.append(_irrigation_year)
            irrigation = np.concatenate(ll_irrigation, axis=0)
            irrigation = np.where(irrigation <= 0, np.nan, irrigation)  # set negative values to nan
            # create xarray data array for irrigation
            da_irrigation = xr.DataArray(
                data=irrigation,
                dims=["time", "y", "x"],
                coords={
                    "time": years,
                    "y": ds_irrigation["y"].values,
                    "x": ds_irrigation["x"].values,
                },
            )
            del irrigation, ll_irrigation, _irrigation_year, ds_irrigation
            click.echo("Calculating irrigation metrics...")
            value = np.nanmean(da_irrigation.values.flatten())
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "irrigation", "unit": "m3/year", "metric": "average", "value": value}
            value = np.nanpercentile(da_irrigation.values.flatten(), 5)
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "irrigation", "unit": "m3/year", "metric": "5th_percentile", "value": value}
            value = np.nanpercentile(da_irrigation.values.flatten(), 25)
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "irrigation", "unit": "m3/year", "metric": "25th_percentile", "value": value}
            value = np.nanpercentile(da_irrigation.values.flatten(), 50)
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "irrigation", "unit": "m3/year", "metric": "median", "value": value}
            value = np.nanpercentile(da_irrigation.values.flatten(), 75)
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "irrigation", "unit": "m3/year", "metric": "75th_percentile", "value": value}
            value = np.nanpercentile(da_irrigation.values.flatten(), 95)
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "irrigation", "unit": "m3/year", "metric": "95th_percentile", "value": value}

            for year in years:
                click.echo(f"Processing year {year}...")
                _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"irrigation_{_stress_test_scenario}_year{year}.nc"
                ds_irrigation = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                ds_irrigation.close()
                _irrigation_year = np.sum(ds_irrigation["irrigation"].values, axis=0)
                irrigation = np.where(_irrigation_year <= 0, np.nan, _irrigation_year)  # set negative values to nan
                value = np.nanmean(irrigation.flatten())
                df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "irrigation", "unit": "m3/year", "metric": "average", "value": value}
                value = np.nanpercentile(irrigation.flatten(), 5)
                df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "irrigation", "unit": "m3/year", "metric": "5th_percentile", "value": value}
                value = np.nanpercentile(irrigation.flatten(), 25)
                df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "irrigation", "unit": "m3/year", "metric": "25th_percentile", "value": value}
                value = np.nanpercentile(irrigation.flatten(), 50)
                df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "irrigation", "unit": "m3/year", "metric": "median", "value": value}
                value = np.nanpercentile(irrigation.flatten(), 75)
                df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "irrigation", "unit": "m3/year", "metric": "75th_percentile", "value": value}
                value = np.nanpercentile(irrigation.flatten(), 95)
                df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "irrigation", "unit": "m3/year", "metric": "95th_percentile", "value": value}
        
            del da_irrigation, irrigation
        
    return

if __name__ == "__main__":
    main()
