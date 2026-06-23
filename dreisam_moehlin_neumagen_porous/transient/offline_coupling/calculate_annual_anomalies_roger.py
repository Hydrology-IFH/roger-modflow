from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
import click
import gc

@click.option("-mr", "--model-run", type=int, default=1806)
@click.option("-a", "--area", type=click.Choice(["dmn", "wsg_hausen", "wsg_zartener_becken", "wsg_boetzingen", "wsg_breisach", "wsg_ebringen", "wsg_eichstetten", "wsg_gottenheim", "wsg_krozinger_berg", "wsg_march", "wsg_schlatt", "wsg_tuniberg", "wsg_umkirch"]), default="dmn", help="Area to process")
@click.command("main", short_help="Calculate annual anomalies for Roger output")
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
        file = base_path.parent / "input" / "dmn_25m.tif"
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
        _potential_evapotranspiration_year = np.where(mask25[np.newaxis, :, :], _potential_evapotranspiration_year, np.nan)
        ll_potential_evapotranspiration.append(_potential_evapotranspiration_year)
    potential_evapotranspiration = np.concatenate(ll_potential_evapotranspiration, axis=0)
    # create xarray data array for potential evapotranspiration
    _da_potential_evapotranspiration_base = xr.DataArray(
        data=potential_evapotranspiration,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_potential_evapotranspiration["y"].values,
            "x": ds_potential_evapotranspiration["x"].values,
        },
    )
    da_potential_evapotranspiration_base = _da_potential_evapotranspiration_base.resample(time="YE").sum(dim="time")
    del potential_evapotranspiration, ll_potential_evapotranspiration, _da_potential_evapotranspiration_base, _potential_evapotranspiration_year, ds_potential_evapotranspiration
    value = np.nanmean(da_potential_evapotranspiration_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "average", "value": value}
    value = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": value}
    value = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": value}
    value = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "median", "value": value}
    value = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "75th_percentile", "value": value}
    value = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": value}

    for i, year in enumerate(years):
        click.echo(f"Processing year {year}...")
        potential_evapotranspiration = da_potential_evapotranspiration_base.values[i, :, :]  # sum over time to get annual potential evapotranspiration
        value = np.nanmean(potential_evapotranspiration.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "average", "value": value}
        value = np.nanpercentile(potential_evapotranspiration.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(potential_evapotranspiration.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(potential_evapotranspiration.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "median", "value": value}
        value = np.nanpercentile(potential_evapotranspiration.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(potential_evapotranspiration.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": value}

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
        _actual_evapotranspiration_year = np.where(mask25[np.newaxis, :, :], _actual_evapotranspiration_year, np.nan)
        ll_actual_evapotranspiration.append(_actual_evapotranspiration_year)
    actual_evapotranspiration = np.concatenate(ll_actual_evapotranspiration, axis=0)
    # create xarray data array for actual evapotranspiration
    _da_actual_evapotranspiration_base = xr.DataArray(
        data=actual_evapotranspiration,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_actual_evapotranspiration["y"].values,
            "x": ds_actual_evapotranspiration["x"].values,
        },
    )
    da_actual_evapotranspiration_base = _da_actual_evapotranspiration_base.resample(time="YE").sum(dim="time")
    del actual_evapotranspiration, ll_actual_evapotranspiration, _da_actual_evapotranspiration_base, _actual_evapotranspiration_year, ds_actual_evapotranspiration
    value = np.nanmean(da_actual_evapotranspiration_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "average", "value": value}
    value = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": value}
    value = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": value}
    value = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "median", "value": value}
    value = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "75th_percentile", "value": value}
    value = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": value}

    for i, year in enumerate(years):
        click.echo(f"Processing year {year}...")
        actual_evapotranspiration = da_actual_evapotranspiration_base.values[i, :, :]  # sum over time to get annual actual evapotranspiration

        value = np.nanmean(actual_evapotranspiration.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "average", "value": value}
        value = np.nanpercentile(actual_evapotranspiration.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(actual_evapotranspiration.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(actual_evapotranspiration.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "median", "value": value}
        value = np.nanpercentile(actual_evapotranspiration.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(actual_evapotranspiration.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": value}

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
        _precipitation_year = np.where(mask25[np.newaxis, :, :], _precipitation_year, np.nan)
        ll_precipitation.append(_precipitation_year)
    precipitation = np.concatenate(ll_precipitation, axis=0)
    # create xarray data array for precipitation
    _da_precipitation_base = xr.DataArray(
        data=precipitation,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_precipitation["y"].values,
            "x": ds_precipitation["x"].values,
        },
    )
    da_precipitation_base = _da_precipitation_base.resample(time="YE").sum(dim="time")
    del precipitation, ll_precipitation, _da_precipitation_base, _precipitation_year, ds_precipitation

    value = np.nanmean(da_precipitation_base.values.flatten())
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "average", "value": value}
    value = np.nanpercentile(da_precipitation_base.values.flatten(), 5)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "5th_percentile", "value": value}
    value = np.nanpercentile(da_precipitation_base.values.flatten(), 25)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "25th_percentile", "value": value}
    value = np.nanpercentile(da_precipitation_base.values.flatten(), 50)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "median", "value": value}
    value = np.nanpercentile(da_precipitation_base.values.flatten(), 75)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "75th_percentile", "value": value}
    value = np.nanpercentile(da_precipitation_base.values.flatten(), 95)
    df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm", "metric": "95th_percentile", "value": value}

    for i, year in enumerate(years):
        click.echo(f"Processing year {year}...")
        precipitation = da_precipitation_base.values[i, :, :]  # sum over time to get annual precipitation

        value = np.nanmean(precipitation.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "average", "value": value}
        value = np.nanpercentile(precipitation.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(precipitation.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(precipitation.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "median", "value": value}
        value = np.nanpercentile(precipitation.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(precipitation.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "95th_percentile", "value": value}

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
    _da_air_temperature_base = xr.DataArray(
        data=air_temperature,
        dims=["time", "y", "x"],
        coords={
            "time": date_time,
            "y": ds_air_temperature["y"].values,
            "x": ds_air_temperature["x"].values,
        },
    )
    da_air_temperature_base = _da_air_temperature_base.resample(time="YE").mean(dim="time")
    del air_temperature, ll_air_temperature, _da_air_temperature_base, _air_temperature_year, ds_air_temperature

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

    for i, year in enumerate(years):
        click.echo(f"Processing year {year}...")
        air_temperature = da_air_temperature_base.values[i, :, :]  # mean over time to get annual air temperature

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
    output_file = base_path / "output" / f"annual_values_run{model_run}_{area}_roger.csv"
    df_metrics.to_csv(output_file, index=False, sep=";")

    for stress_test_scenario in stress_test_scenarios:
        # load potential evapotranspiration
        click.echo("Loading potential evapotranspiration...")
        ll_potential_evapotranspiration = []
        for year in years:
            _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"potential_evapotranspiration_{_stress_test_scenario}_year{year}.nc"
            ds_potential_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _potential_evapotranspiration_year = ds_potential_evapotranspiration["potential_evapotranspiration"].values
            ds_potential_evapotranspiration.close()
            # set negative values to zero
            _potential_evapotranspiration_year[_potential_evapotranspiration_year < 0] = 0
            _potential_evapotranspiration_year = np.where(mask25[np.newaxis, :, :], _potential_evapotranspiration_year, np.nan)
            ll_potential_evapotranspiration.append(_potential_evapotranspiration_year)
        potential_evapotranspiration = np.concatenate(ll_potential_evapotranspiration, axis=0)
        # create xarray data array for potential evapotranspiration
        _da_potential_evapotranspiration = xr.DataArray(
            data=potential_evapotranspiration,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_potential_evapotranspiration["y"].values,
                "x": ds_potential_evapotranspiration["x"].values,
            },
        )
        da_potential_evapotranspiration = _da_potential_evapotranspiration.resample(time="YE").sum(dim="time")
        del potential_evapotranspiration, ll_potential_evapotranspiration, _da_potential_evapotranspiration, _potential_evapotranspiration_year, ds_potential_evapotranspiration
        click.echo("Calculating potential evapotranspiration anomalies...")
        value = np.nanmean(da_potential_evapotranspiration.values.flatten())
        value_base = np.nanmean(da_potential_evapotranspiration_base.values.flatten())
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_potential_evapotranspiration.values.flatten(), 5)
        value_base = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 5)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_potential_evapotranspiration.values.flatten(), 25)
        value_base = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 25)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_potential_evapotranspiration.values.flatten(), 50)
        value_base = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 50)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "median", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_potential_evapotranspiration.values.flatten(), 75)
        value_base = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 75)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "75th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "75th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_potential_evapotranspiration.values.flatten(), 95)
        value_base = np.nanpercentile(da_potential_evapotranspiration_base.values.flatten(), 95)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "potential_evapotranspiration", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        for i, year in enumerate(years):
            click.echo(f"Processing year {year}...")
            potential_evapotranspiration_base = da_potential_evapotranspiration_base.values[i, :, :]
            potential_evapotranspiration = da_potential_evapotranspiration.values[i, :, :]

            value = np.nanmean(potential_evapotranspiration.flatten())
            value_base = np.nanmean(potential_evapotranspiration_base.flatten())
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(potential_evapotranspiration.flatten(), 5)
            value_base = np.nanpercentile(potential_evapotranspiration_base.flatten(), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(potential_evapotranspiration.flatten(), 25)
            value_base = np.nanpercentile(potential_evapotranspiration_base.flatten(), 25)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(potential_evapotranspiration.flatten(), 50)
            value_base = np.nanpercentile(potential_evapotranspiration_base.flatten(), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "median", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(potential_evapotranspiration.flatten(), 75)
            value_base = np.nanpercentile(potential_evapotranspiration_base.flatten(), 75)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(potential_evapotranspiration.flatten(), 95)
            value_base = np.nanpercentile(potential_evapotranspiration_base.flatten(), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "potential_evapotranspiration", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        del da_potential_evapotranspiration

        # load actual evapotranspiration
        click.echo("Loading actual evapotranspiration...")
        ll_actual_evapotranspiration = []
        for year in years:
            _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"actual_evapotranspiration_{_stress_test_scenario}_year{year}.nc"
            ds_actual_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _actual_evapotranspiration_year = ds_actual_evapotranspiration["actual_evapotranspiration"].values
            ds_actual_evapotranspiration.close()
            # set negative values to zero
            _actual_evapotranspiration_year[_actual_evapotranspiration_year < 0] = 0
            _actual_evapotranspiration_year = np.where(mask25[np.newaxis, :, :], _actual_evapotranspiration_year, np.nan)
            ll_actual_evapotranspiration.append(_actual_evapotranspiration_year)
        actual_evapotranspiration = np.concatenate(ll_actual_evapotranspiration, axis=0)
        # create xarray data array for actual evapotranspiration
        _da_actual_evapotranspiration = xr.DataArray(
            data=actual_evapotranspiration,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_actual_evapotranspiration["y"].values,
                "x": ds_actual_evapotranspiration["x"].values,
            },
        )
        da_actual_evapotranspiration = _da_actual_evapotranspiration.resample(time="YE").sum(dim="time")
        del actual_evapotranspiration, ll_actual_evapotranspiration, _da_actual_evapotranspiration, _actual_evapotranspiration_year, ds_actual_evapotranspiration
        click.echo("Calculating actual evapotranspiration anomalies...")
        value = np.nanmean(da_actual_evapotranspiration.values.flatten())
        value_base = np.nanmean(da_actual_evapotranspiration_base.values.flatten())
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_actual_evapotranspiration.values.flatten(), 5)
        value_base = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 5)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_actual_evapotranspiration.values.flatten(), 25)
        value_base = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 25)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_actual_evapotranspiration.values.flatten(), 50)
        value_base = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 50)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "median", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_actual_evapotranspiration.values.flatten(), 95)
        value_base = np.nanpercentile(da_actual_evapotranspiration_base.values.flatten(), 95)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        for i, year in enumerate(years):
            actual_evapotranspiration_base = da_actual_evapotranspiration_base.values[i, :, :]
            actual_evapotranspiration = da_actual_evapotranspiration.values[i, :, :]

            value = np.nanmean(actual_evapotranspiration.flatten())
            value_base = np.nanmean(actual_evapotranspiration_base.flatten())
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(actual_evapotranspiration.flatten(), 5)
            value_base = np.nanpercentile(actual_evapotranspiration_base.flatten(), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "5th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(actual_evapotranspiration.flatten(), 25)
            value_base = np.nanpercentile(actual_evapotranspiration_base.flatten(), 25)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(actual_evapotranspiration.flatten(), 50)
            value_base = np.nanpercentile(actual_evapotranspiration_base.flatten(), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "median", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(actual_evapotranspiration.flatten(), 75)
            value_base = np.nanpercentile(actual_evapotranspiration_base.flatten(), 75)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(actual_evapotranspiration.flatten(), 95)
            value_base = np.nanpercentile(actual_evapotranspiration_base.flatten(), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "mm/year", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "actual_evapotranspiration", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        del da_actual_evapotranspiration

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
            _precipitation_year = np.where(mask25[np.newaxis, :, :], _precipitation_year, np.nan)
            ll_precipitation.append(_precipitation_year)
        precipitation = np.concatenate(ll_precipitation, axis=0)
        # create xarray data array for precipitation
        _da_precipitation = xr.DataArray(
            data=precipitation,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_precipitation["y"].values,
                "x": ds_precipitation["x"].values,
            },
        )
        da_precipitation = _da_precipitation.resample(time="YE").sum(dim="time")
        del precipitation, ll_precipitation, _da_precipitation, _precipitation_year, ds_precipitation
        click.echo("Calculating precipitation anomalies...")
        value = np.nanmean(da_precipitation.values.flatten())
        value_base = np.nanmean(da_precipitation_base.values.flatten())
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "average", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "average", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "average", "value": anomaly_rel}
        value = np.nanpercentile(da_precipitation.values.flatten(), 5)
        value_base = np.nanpercentile(da_precipitation_base.values.flatten(), 5)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "5th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "5th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_precipitation.values.flatten(), 25)
        value_base = np.nanpercentile(da_precipitation_base.values.flatten(), 25)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "25th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "25th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_precipitation.values.flatten(), 50)
        value_base = np.nanpercentile(da_precipitation_base.values.flatten(), 50)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "median", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "median", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "median", "value": anomaly_rel}
        value = np.nanpercentile(da_precipitation.values.flatten(), 75)
        value_base = np.nanpercentile(da_precipitation_base.values.flatten(), 75)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "75th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "75th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
        value = np.nanpercentile(da_precipitation.values.flatten(), 95)
        value_base = np.nanpercentile(da_precipitation_base.values.flatten(), 95)
        anomaly_abs = value - value_base
        anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
        df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "95th_percentile", "value": value}
        df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm/year", "metric": "95th_percentile", "value": anomaly_abs}
        df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "precipitation", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        for i, year in enumerate(years):
            click.echo(f"Processing year {year}...")
            precipitation_base = da_precipitation_base.values[i, :, :]
            precipitation = da_precipitation.values[i, :, :]

            value = np.nanmean(precipitation.flatten())
            value_base = np.nanmean(precipitation_base.flatten())
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "average", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "average", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "average", "value": anomaly_rel}
            value = np.nanpercentile(precipitation.flatten(), 5)
            value_base = np.nanpercentile(precipitation_base.flatten(), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "5th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "5th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area,"time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(precipitation.flatten(), 25)
            value_base = np.nanpercentile(precipitation_base.flatten(), 25)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "25th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "25th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "25th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(precipitation.flatten(), 50)
            value_base = np.nanpercentile(precipitation_base.flatten(), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "median", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "median", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "median", "value": anomaly_rel}
            value = np.nanpercentile(precipitation.flatten(), 75)
            value_base = np.nanpercentile(precipitation_base.flatten(), 75)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "75th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "75th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "75th_percentile", "value": anomaly_rel}
            value = np.nanpercentile(precipitation.flatten(), 95)
            value_base = np.nanpercentile(precipitation_base.flatten(), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm/year", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    

        del da_precipitation

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
            ds_air_temperature.close()
            _air_temperature_year = np.where(_air_temperature_year < -50, np.nan, _air_temperature_year)
            _air_temperature_year = np.where(mask25[np.newaxis, :, :], _air_temperature_year, np.nan)
            ll_air_temperature.append(_air_temperature_year)
        air_temperature = np.concatenate(ll_air_temperature, axis=0)
        # create xarray data array for air temperature
        _da_air_temperature = xr.DataArray(
            data=air_temperature,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_air_temperature["y"].values,
                "x": ds_air_temperature["x"].values,
            },
        )
        da_air_temperature = _da_air_temperature.resample(time="YE").mean(dim="time")
        del air_temperature, ll_air_temperature, _da_air_temperature, _air_temperature_year, ds_air_temperature
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

        for i, year in enumerate(years):
            click.echo(f"Processing year {year}...")
            air_temperature_base = da_air_temperature_base.values[i, :, :]
            air_temperature = da_air_temperature.values[i, :, :]

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

        del da_air_temperature

        if "_irrigation" in stress_test_scenario:
            # load irrigation
            click.echo("Loading irrigation...")
            ll_irrigation = []
            for year in years:
                _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"irrigation_{_stress_test_scenario}_year{year}.nc"
                ds_irrigation = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _irrigation_year = ds_irrigation["irrigation"].values
                ds_irrigation.close()
                _irrigation_year = np.where(_irrigation_year < 0, np.nan, _irrigation_year)
                _irrigation_year = np.where(mask25, _irrigation_year, np.nan)
                ll_irrigation.append(_irrigation_year)
            irrigation = np.concatenate(ll_irrigation, axis=0)
            irrigation = np.where(irrigation <= 0, np.nan, irrigation)  # set negative values to nan
            # create xarray data array for irrigation
            _da_irrigation = xr.DataArray(
                data=irrigation,
                dims=["time", "y", "x"],
                coords={
                    "time": date_time,
                    "y": ds_irrigation["y"].values,
                    "x": ds_irrigation["x"].values,
                },
            )
            da_irrigation = _da_irrigation.resample(time="YE").sum(dim="time")
            del irrigation, ll_irrigation, _da_irrigation, _irrigation_year, ds_irrigation
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

            for i, year in enumerate(years):
                irrigation = da_irrigation.values[i, :, :]

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

            del da_irrigation

        gc.collect()

        df_metrics = df_metrics.copy()
        df_anomaly_metrics_abs = df_anomaly_metrics_abs.copy()
        df_anomaly_metrics_rel = df_anomaly_metrics_rel.copy()

        # save the metrics to csv
        output_file = base_path / "output" / f"annual_values_run{model_run}_{area}_roger.csv"
        df_metrics.to_csv(output_file, index=False, sep=";")
        output_file = base_path / "output" / f"annual_anomalies_abs_run{model_run}_{area}_roger.csv"
        df_anomaly_metrics_abs.to_csv(output_file, index=False, sep=";")
        output_file = base_path / "output" / f"annual_anomalies_rel_run{model_run}_{area}_roger.csv"
        df_anomaly_metrics_rel.to_csv(output_file, index=False, sep=";")

    return

if __name__ == "__main__":
    main()
