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

    areas = ["dmn", "wsg_hausen", "wsg_zartener_becken", "wsg_boetzingen", "wsg_breisach", "wsg_ebringen", "wsg_eichstetten", "wsg_gottenheim", "wsg_krozinger_berg", "wsg_march", "wsg_schlatt", "wsg_tuniberg", "wsg_umkirch"]

    base = "base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"
    
    stress_test_scenarios = ["summer-drought-magnitude0-duration3_no-irrigation_no-yellow-mustard_soil-compaction",
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

    df_metrics = pd.DataFrame(columns=["scenario", "area", "time", "variable", "unit", "metric", "value"])
    df_anomaly_metrics_abs = pd.DataFrame(columns=["scenario", "area", "time", "variable", "unit", "metric", "value"])
    df_anomaly_metrics_rel = pd.DataFrame(columns=["scenario", "area", "time", "variable", "unit", "metric", "value"])

    for area in areas:
        if area == "dmn":
            mask = ds_params["mask_porous_aquifer"].values
        else:
            file = base_path.parent / "input" / f"{area}_.tif"
            with rasterio.open(file) as src:
                mask = src.read(1)
                mask = np.where(mask == 1, True, False)

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
            output_file = base_path / "output" / f"modflow_{base}" / f"gw_depth_run{model_run}_year{year}.nc"
            # output_file = base_path_output / f"{base}" / f"gw_depth_run{model_run}_year{year}.nc"
            ds_gw_depths = xr.open_dataset(output_file, engine="h5netcdf")
            gw_depths_year = ds_gw_depths["depth"].values[:, 1, :, :]
            gw_depths_year = np.where(mask[np.newaxis, :, :], gw_depths_year, np.nan)

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
            output_file = base_path / "output" / f"modflow_{base}" / f"indirect_recharge_run{model_run}_year{year}.nc"
            # output_file = base_path_output / f"{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
            ds_indirect_recharge = xr.open_dataset(output_file, engine="h5netcdf")
            indirect_recharge_year = ds_indirect_recharge["indirect_recharge"].values * 86400  # convert from m3/s to m3/day
            ds_indirect_recharge.close()
            indirect_recharge_year[indirect_recharge_year >= 0] = np.nan  # set positive values to zero
            indirect_recharge_year = np.abs(indirect_recharge_year)
            indirect_recharge_year = np.where(mask[np.newaxis, :, :], indirect_recharge_year, np.nan)

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
            for i in range(_direct_recharge_year.shape[0]):
                direct_recharge_day = aggregate_to_coarser_resolution(_direct_recharge_year[i, :, :], 25, 50, method="sum")
                direct_recharge_day = np.where(mask, direct_recharge_day, np.nan)
                ll_direct_recharge.append(direct_recharge_day)
        direct_recharge = np.stack(ll_direct_recharge, axis=0)
        # convert from mm/day to m3/day
        # get the area of each grid cell in m2
        _area = 50 * 50  # 50 m x 50 m grid cells
        # multiply direct recharge by area to get m3/day
        direct_recharge = direct_recharge * _area / 1000
        # create xarray data array for direct recharge
        da_direct_recharge_base = xr.DataArray(
            data=direct_recharge,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ycoords,
                "x": xcoords,
            },
        )
        value = np.nanmean(da_direct_recharge_base.values.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "average", "value": value}
        value = np.nanpercentile(da_direct_recharge_base.values.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario ": base,("area"): area, ("time"):	"overall ", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"5th_percentile ", ("value"): 	value}
        value = np.nanpercentile(da_direct_recharge_base.values.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario ": base,("area"): area, ("time"):	"overall ", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"25th_percentile ", ("value"): 	value}
        value = np.nanpercentile(da_direct_recharge_base.values.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario ": base,("area"): area, ("time"):	"overall ", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"50th_percentile ", ("value"): 	value}
        value = np.nanpercentile(da_direct_recharge_base.values.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario ": base,("area"): area, ("time"):	"overall ", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"75th_percentile ", ("value"): 	value}
        value = np.nanpercentile(da_direct_recharge_base.values.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "direct_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": value}

        for year in years:
            click.echo(f"Processing year {year}...")
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"recharge_{base}_year{year}.nc"
            ds_direct_recharge = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _direct_recharge_year = ds_direct_recharge["recharge"].values
            ds_direct_recharge.close()
            _direct_recharge_year[_direct_recharge_year < 0] = 0  # set negative values to zero
            _direct_recharge_year[_direct_recharge_year > 100] = 100  # set values above 100 mm/day to 100 mm/day
            ll_direct_recharge = []
            for i in range(_direct_recharge_year.shape[0]):
                direct_recharge_day = aggregate_to_coarser_resolution(_direct_recharge_year[i, :, :], 25, 50, method="sum")
                direct_recharge_day = np.where(mask, direct_recharge_day, np.nan)
                ll_direct_recharge.append(direct_recharge_day)
            direct_recharge = np.stack(ll_direct_recharge, axis=0)
            # convert from mm/day to m3/day
            # get the area of each grid cell in m2
            _area = 50 * 50  # 50 m x 50 m grid cells
            # multiply direct recharge by area to get m3/day
            direct_recharge = direct_recharge * _area / 1000
            value = np.nanmean(direct_recharge.flatten())
            df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "average", "value": value}
            value = np.nanpercentile(direct_recharge.flatten(), 5)
            df_metrics.loc[len(df_metrics)] = {"scenario ": base,("area"): area, ("time"):	f"{year}", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"5th_percentile ", ("value"): 	value}
            value = np.nanpercentile(direct_recharge.flatten(), 25)
            df_metrics.loc[len(df_metrics)] = {"scenario ": base,("area"): area, ("time"):	f"{year}", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"25th_percentile ", ("value"): 	value}
            value = np.nanpercentile(direct_recharge.flatten(), 50)
            df_metrics.loc[len(df_metrics)] = {"scenario ": base,("area"): area, ("time"):	f"{year}", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"50th_percentile ", ("value"): 	value}
            value = np.nanpercentile(direct_recharge.flatten(), 75)
            df_metrics.loc[len(df_metrics)] = {"scenario ": base,("area"): area, ("time"):	f"{year}", ("variable"):	"direct_recharge ", ("unit"): 	"m3/day ", ("metric"): 	"75th_percentile ", ("value"): 	value}
            value = np.nanpercentile(direct_recharge.flatten(), 95)
            df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "direct_recharge", "unit": "m3/day", "metric": "95th_percentile", "value": value}

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
            for i in range(_potential_evapotranspiration_year.shape[0]):
                potential_evapotranspiration_day = aggregate_to_coarser_resolution(_potential_evapotranspiration_year[i, :, :], 25, 50, method="sum")
                potential_evapotranspiration_day = np.where(mask, potential_evapotranspiration_day, np.nan)
                ll_potential_evapotranspiration.append(potential_evapotranspiration_day)
        potential_evapotranspiration = np.stack(ll_potential_evapotranspiration, axis=0)
        # create xarray data array for potential evapotranspiration
        da_potential_evapotranspiration_base = xr.DataArray(
            data=potential_evapotranspiration,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ycoords,
                "x": xcoords,
            },
        )
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
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"potential_evapotranspiration_{base}_year{year}.nc"
            ds_potential_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _potential_evapotranspiration_year = ds_potential_evapotranspiration["potential_evapotranspiration"].values
            ds_potential_evapotranspiration.close()
            _potential_evapotranspiration_year[_potential_evapotranspiration_year < 0] = 0  # set negative values to zero
            ll_potential_evapotranspiration = []
            for i in range(_potential_evapotranspiration_year.shape[0]):
                potential_evapotranspiration_day = aggregate_to_coarser_resolution(_potential_evapotranspiration_year[i, :, :], 25, 50, method="sum")
                potential_evapotranspiration_day = np.where(mask, potential_evapotranspiration_day, np.nan)
                ll_potential_evapotranspiration.append(potential_evapotranspiration_day)
            potential_evapotranspiration = np.stack(ll_potential_evapotranspiration, axis=0)
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
            for i in range(_actual_evapotranspiration_year.shape[0]):
                actual_evapotranspiration_day = aggregate_to_coarser_resolution(_actual_evapotranspiration_year[i, :, :], 25, 50, method="sum")
                actual_evapotranspiration_day = np.where(mask, actual_evapotranspiration_day, np.nan)
                ll_actual_evapotranspiration.append(actual_evapotranspiration_day)
        actual_evapotranspiration = np.stack(ll_actual_evapotranspiration, axis=0)
        # create xarray data array for actual evapotranspiration
        da_actual_evapotranspiration_base = xr.DataArray(
            data=actual_evapotranspiration,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ycoords,
                "x": xcoords,
            },
        )
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
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"actual_evapotranspiration_{base}_year{year}.nc"
            ds_actual_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _actual_evapotranspiration_year = ds_actual_evapotranspiration["actual_evapotranspiration"].values
            ds_actual_evapotranspiration.close()
            _actual_evapotranspiration_year[_actual_evapotranspiration_year < 0] = 0  # set negative values to zero
            ll_actual_evapotranspiration = []
            for i in range(_actual_evapotranspiration_year.shape[0]):
                actual_evapotranspiration_day = aggregate_to_coarser_resolution(_actual_evapotranspiration_year[i, :, :], 25, 50, method="sum")
                actual_evapotranspiration_day = np.where(mask, actual_evapotranspiration_day, np.nan)
                ll_actual_evapotranspiration.append(actual_evapotranspiration_day)
            actual_evapotranspiration = np.stack(ll_actual_evapotranspiration, axis=0)

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
            for i in range(_precipitation_year.shape[0]):
                precipitation_day = aggregate_to_coarser_resolution(_precipitation_year[i, :, :], 25, 50, method="sum")
                precipitation_day = np.where(mask, precipitation_day, np.nan)
                ll_precipitation.append(precipitation_day)
        precipitation = np.stack(ll_precipitation, axis=0)
        # create xarray data array for precipitation
        da_precipitation_base = xr.DataArray(
            data=precipitation,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ycoords,
                "x": xcoords,
            },
        )

        value = np.nanmean(da_precipitation_base.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm", "metric": "average", "value": value}
        value = np.nanpercentile(da_precipitation_base.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(da_precipitation_base.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(da_precipitation_base.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm", "metric": "median", "value": value}
        value = np.nanpercentile(da_precipitation_base.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(da_precipitation_base.flatten(), 95)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "precipitation", "unit": "mm", "metric": "95th_percentile", "value": value}

        for year in years:
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"precipitation_{base}_year{year}.nc"
            ds_precipitation = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _precipitation_year = ds_precipitation["precipitation"].values
            ds_precipitation.close()
            _precipitation_year[_precipitation_year < 0] = 0  # set negative values to zero
            ll_precipitation = []
            for i in range(_precipitation_year.shape[0]):
                precipitation_day = aggregate_to_coarser_resolution(_precipitation_year[i, :, :], 25, 50, method="sum")
                precipitation_day = np.where(mask, precipitation_day, np.nan)
                ll_precipitation.append(precipitation_day)
            precipitation = np.stack(ll_precipitation, axis=0)

            value = np.nanmean(precipitation.flatten())
            df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm", "metric": "average", "value": value}
            value = np.nanpercentile(precipitation.flatten(), 5)
            df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm", "metric": "5th_percentile", "value": value}
            value = np.nanpercentile(precipitation.flatten(), 25)
            df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm", "metric": "25th_percentile", "value": value}
            value = np.nanpercentile(precipitation.flatten(), 50)
            df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm", "metric": "median", "value": value}
            value = np.nanpercentile(precipitation.flatten(), 75)
            df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm", "metric": "75th_percentile", "value": value}
            value = np.nanpercentile(precipitation.flatten(), 95)
            df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": f"{year}", "variable": "precipitation", "unit": "mm", "metric": "95th_percentile", "value": value}

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
            for i in range(_air_temperature_year.shape[0]):
                air_temperature_day = aggregate_to_coarser_resolution(_air_temperature_year[i, :, :], 25, 50, method="average")
                air_temperature_day = np.where(mask, air_temperature_day, np.nan)
                ll_air_temperature.append(air_temperature_day)
        air_temperature = np.stack(ll_air_temperature, axis=0)
        # create xarray data array for air temperature
        da_air_temperature_base = xr.DataArray(
            data=air_temperature,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ycoords,
                "x": xcoords,
            },
        )

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
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"ta_{base}_year{year}.nc"
            # _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
            # output_file = base_path_output / f"{stress_test_scenario}" / f"recharge_{_stress_test_scenario}_year{year}.nc"
            ds_air_temperature = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
            _air_temperature_year = ds_air_temperature["ta"].values
            _air_temperature_year = np.where(_air_temperature_year < -50, np.nan, _air_temperature_year)
            ds_air_temperature.close()
            ll_air_temperature = []
            for i in range(_air_temperature_year.shape[0]):
                air_temperature_day = aggregate_to_coarser_resolution(_air_temperature_year[i, :, :], 25, 50, method="average")
                air_temperature_day = np.where(mask, air_temperature_day, np.nan)
                ll_air_temperature.append(air_temperature_day)
            air_temperature = np.stack(ll_air_temperature, axis=0)

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
                "y": ycoords,
                "x": xcoords,
            },
        )

        value = np.nanmean(da_well_extraction_base.flatten())
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "average", "value": value}
        value = np.nanpercentile(da_well_extraction_base.flatten(), 5)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "5th_percentile", "value": value}
        value = np.nanpercentile(da_well_extraction_base.flatten(), 25)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "25th_percentile", "value": value}
        value = np.nanpercentile(da_well_extraction_base.flatten(), 50)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "median", "value": value}
        value = np.nanpercentile(da_well_extraction_base.flatten(), 75)
        df_metrics.loc[len(df_metrics)] = {"scenario": base, "area": area, "time": "overall", "variable": "well_extraction", "unit": "m3/day", "metric": "75th_percentile", "value": value}
        value = np.nanpercentile(da_well_extraction_base.flatten(), 95)
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
        output_file = base_path / "output" / f"metrics_run{model_run}.csv"
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
                output_file = base_path / "output" / f"modflow_{base}" / f"gw_depth_run{model_run}_year{year}.nc"
                # output_file = base_path_output / f"{stress_test_scenario}" / f"gw_depth_run{model_run}_year{year}.nc"
                ds_gw_depths = xr.open_dataset(output_file, engine="h5netcdf")
                gw_depths_base_year = ds_gw_depths["depth"].values[:, 1, :, :]
                gw_depths_base_year = np.where(mask[np.newaxis, :, :], gw_depths_base_year, np.nan)

                output_file = base_path / "output" / f"modflow_{stress_test_scenario}" / f"gw_depth_run{model_run}_year{year}.nc"
                # output_file = base_path_output / f"{stress_test_scenario}" / f"gw_depth_run{model_run}_year{year}.nc"
                ds_gw_depths = xr.open_dataset(output_file, engine="h5netcdf")
                gw_depths_year = ds_gw_depths["depth"].values[:, 1, :, :]
                gw_depths_year = np.where(mask[np.newaxis, :, :], gw_depths_year, np.nan)

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
                output_file = base_path / "output" / f"modflow_{base}" / f"indirect_recharge_run{model_run}_year{year}.nc"
                # output_file = base_path_output / f"{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
                ds_indirect_recharge = xr.open_dataset(output_file, engine="h5netcdf")
                indirect_recharge_year = ds_indirect_recharge["indirect_recharge"].values * 86400  # convert from m3/s to m3/day
                ds_indirect_recharge.close()
                indirect_recharge_year[indirect_recharge_year >= 0] = np.nan  # set positive values to zero
                indirect_recharge_year = np.abs(indirect_recharge_year)
                indirect_recharge_base_year = np.where(mask[np.newaxis, :, :], indirect_recharge_year, np.nan)

                output_file = base_path / "output" / f"modflow_{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
                # output_file = base_path_output / f"{stress_test_scenario}" / f"indirect_recharge_run{model_run}_year{year}.nc"
                ds_indirect_recharge = xr.open_dataset(output_file, engine="h5netcdf")
                indirect_recharge_year = ds_indirect_recharge["indirect_recharge"].values * 86400  # convert from m3/s to m3/day
                ds_indirect_recharge.close()
                indirect_recharge_year[indirect_recharge_year >= 0] = np.nan  # set positive values to zero
                indirect_recharge_year = np.abs(indirect_recharge_year)
                indirect_recharge_year = np.where(mask[np.newaxis, :, :], indirect_recharge_year, np.nan)

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
                    direct_recharge_day = aggregate_to_coarser_resolution(_direct_recharge_year[i, :, :], 25, 50, method="sum")
                    direct_recharge_day = np.where(mask, direct_recharge_day, np.nan)
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
                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"recharge_{base}_year{year}.nc"
                ds_direct_recharge = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _direct_recharge_year = ds_direct_recharge["recharge"].values
                ds_direct_recharge.close()
                _direct_recharge_year[_direct_recharge_year < 0] = 0  # set negative values to zero
                _direct_recharge_year[_direct_recharge_year > 100] = 100  # set values above 100 mm/day to 100 mm/day
                ll_direct_recharge = []
                for i in range(_direct_recharge_year.shape[0]):
                    direct_recharge_day = aggregate_to_coarser_resolution(_direct_recharge_year[i, :, :], 25, 50, method="sum")
                    direct_recharge_day = np.where(mask, direct_recharge_day, np.nan)
                    ll_direct_recharge.append(direct_recharge_day)
                direct_recharge = np.stack(ll_direct_recharge, axis=0)
                # convert from mm/day to m3/day
                # get the area of each grid cell in m2
                _area = 50 * 50  # 50 m x 50 m grid cells
                # multiply direct recharge by area to get m3/day
                direct_recharge_base = direct_recharge * _area / 1000

                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"recharge_{_stress_test_scenario}_year{year}.nc"
                ds_direct_recharge = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _direct_recharge_year = ds_direct_recharge["recharge"].values
                ds_direct_recharge.close()
                _direct_recharge_year[_direct_recharge_year < 0] = 0  # set negative values to zero
                _direct_recharge_year[_direct_recharge_year > 100] = 100  # set values above 100 mm/day to 100 mm/day
                ll_direct_recharge = []
                for i in range(_direct_recharge_year.shape[0]):
                    direct_recharge_day = aggregate_to_coarser_resolution(_direct_recharge_year[i, :, :], 25, 50, method="sum")
                    direct_recharge_day = np.where(mask, direct_recharge_day, np.nan)
                    ll_direct_recharge.append(direct_recharge_day)
                direct_recharge = np.stack(ll_direct_recharge, axis=0)
                # convert from mm/day to m3/day
                # get the area of each grid cell in m2
                _area = 50 * 50  # 50 m x 50 m grid cells
                # multiply direct recharge by area to get m3/day
                direct_recharge = direct_recharge * _area / 1000

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
                for i in range(_potential_evapotranspiration_year.shape[0]):
                    potential_evapotranspiration_day = aggregate_to_coarser_resolution(_potential_evapotranspiration_year[i, :, :], 25, 50, method="sum")
                    potential_evapotranspiration_day = np.where(mask, potential_evapotranspiration_day, np.nan)
                    ll_potential_evapotranspiration.append(potential_evapotranspiration_day)
            potential_evapotranspiration = np.stack(ll_potential_evapotranspiration, axis=0)
            # create xarray data array for potential evapotranspiration
            da_potential_evapotranspiration = xr.DataArray(
                data=potential_evapotranspiration,
                dims=["time", "y", "x"],
                coords={
                    "time": date_time,
                    "y": ycoords,
                    "x": xcoords,
                },
            )
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
                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"potential_evapotranspiration_{base}_year{year}.nc"
                ds_potential_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _potential_evapotranspiration_year = ds_potential_evapotranspiration["potential_evapotranspiration"].values
                ds_potential_evapotranspiration.close()
                _potential_evapotranspiration_year[_potential_evapotranspiration_year < 0] = 0  # set negative values to zero
                ll_potential_evapotranspiration = []
                for i in range(_potential_evapotranspiration_year.shape[0]):
                    potential_evapotranspiration_day = aggregate_to_coarser_resolution(_potential_evapotranspiration_year[i, :, :], 25, 50, method="sum")
                    potential_evapotranspiration_day = np.where(mask, potential_evapotranspiration_day, np.nan)
                    ll_potential_evapotranspiration.append(potential_evapotranspiration_day)
                potential_evapotranspiration_base = np.stack(ll_potential_evapotranspiration, axis=0)

                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"potential_evapotranspiration_{_stress_test_scenario}_year{year}.nc"
                ds_potential_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _potential_evapotranspiration_year = ds_potential_evapotranspiration["potential_evapotranspiration"].values
                ds_potential_evapotranspiration.close()
                _potential_evapotranspiration_year[_potential_evapotranspiration_year < 0] = 0  # set negative values to zero
                ll_potential_evapotranspiration = []
                for i in range(_potential_evapotranspiration_year.shape[0]):
                    potential_evapotranspiration_day = aggregate_to_coarser_resolution(_potential_evapotranspiration_year[i, :, :], 25, 50, method="sum")
                    potential_evapotranspiration_day = np.where(mask, potential_evapotranspiration_day, np.nan)
                    ll_potential_evapotranspiration.append(potential_evapotranspiration_day)
                potential_evapotranspiration = np.stack(ll_potential_evapotranspiration, axis=0)

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
                for i in range(_actual_evapotranspiration_year.shape[0]):
                    actual_evapotranspiration_day = aggregate_to_coarser_resolution(_actual_evapotranspiration_year[i, :, :], 25, 50, method="sum")
                    actual_evapotranspiration_day = np.where(mask, actual_evapotranspiration_day, np.nan)
                    ll_actual_evapotranspiration.append(actual_evapotranspiration_day)
            actual_evapotranspiration = np.stack(ll_actual_evapotranspiration, axis=0)
            # create xarray data array for actual evapotranspiration
            da_actual_evapotranspiration = xr.DataArray(
                data=actual_evapotranspiration,
                dims=["time", "y", "x"],
                coords={
                    "time": date_time,
                    "y": ycoords,
                    "x": xcoords,
                },
            )
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
                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"actual_evapotranspiration_{base}_year{year}.nc"
                ds_actual_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _actual_evapotranspiration_year = ds_actual_evapotranspiration["actual_evapotranspiration"].values
                ds_actual_evapotranspiration.close()
                _actual_evapotranspiration_year[_actual_evapotranspiration_year < 0] = 0  # set negative values to zero
                ll_actual_evapotranspiration = []
                for i in range(_actual_evapotranspiration_year.shape[0]):
                    actual_evapotranspiration_day = aggregate_to_coarser_resolution(_actual_evapotranspiration_year[i, :, :], 25, 50, method="sum")
                    actual_evapotranspiration_day = np.where(mask, actual_evapotranspiration_day, np.nan)
                    ll_actual_evapotranspiration.append(actual_evapotranspiration_day)
                actual_evapotranspiration_base = np.stack(ll_actual_evapotranspiration, axis=0)

                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"actual_evapotranspiration_{_stress_test_scenario}_year{year}.nc"
                ds_actual_evapotranspiration = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _actual_evapotranspiration_year = ds_actual_evapotranspiration["actual_evapotranspiration"].values
                ds_actual_evapotranspiration.close()
                _actual_evapotranspiration_year[_actual_evapotranspiration_year < 0] = 0  # set negative values to zero
                ll_actual_evapotranspiration = []
                for i in range(_actual_evapotranspiration_year.shape[0]):
                    actual_evapotranspiration_day = aggregate_to_coarser_resolution(_actual_evapotranspiration_year[i, :, :], 25, 50, method="sum")
                    actual_evapotranspiration_day = np.where(mask, actual_evapotranspiration_day, np.nan)
                    ll_actual_evapotranspiration.append(actual_evapotranspiration_day)
                actual_evapotranspiration = np.stack(ll_actual_evapotranspiration, axis=0)

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
                for i in range(_precipitation_year.shape[0]):
                    precipitation_day = aggregate_to_coarser_resolution(_precipitation_year[i, :, :], 25, 50, method="sum")
                    precipitation_day = np.where(mask, precipitation_day, np.nan)
                    ll_precipitation.append(precipitation_day)
            precipitation = np.stack(ll_precipitation, axis=0)
            # create xarray data array for precipitation
            da_precipitation = xr.DataArray(
                data=precipitation,
                dims=["time", "y", "x"],
                coords={
                    "time": date_time,
                    "y": ycoords,
                    "x": xcoords,
                },
            )

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
                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"precipitation_{base}_year{year}.nc"
                ds_precipitation = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _precipitation_year = ds_precipitation["precipitation"].values
                ds_precipitation.close()
                _precipitation_year[_precipitation_year < 0] = 0  # set negative values to zero
                ll_precipitation = []
                for i in range(_precipitation_year.shape[0]):
                    precipitation_day = aggregate_to_coarser_resolution(_precipitation_year[i, :, :], 25, 50, method="sum")
                    precipitation_day = np.where(mask, precipitation_day, np.nan)
                    ll_precipitation.append(precipitation_day)
                precipitation_base = np.stack(ll_precipitation, axis=0)

                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"precipitation_{_stress_test_scenario}_year{year}.nc"
                ds_precipitation = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _precipitation_year = ds_precipitation["precipitation"].values
                ds_precipitation.close()
                _precipitation_year[_precipitation_year < 0] = 0  # set negative values to zero
                ll_precipitation = []
                for i in range(_precipitation_year.shape[0]):
                    precipitation_day = aggregate_to_coarser_resolution(_precipitation_year[i, :, :], 25, 50, method="sum")
                    precipitation_day = np.where(mask, precipitation_day, np.nan)
                    ll_precipitation.append(precipitation_day)
                precipitation = np.stack(ll_precipitation, axis=0)

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
                df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area,("time"): f"{year}", "variable": "precipitation", "unit": "%", "metric": "5th_percentile", "value": anomaly_rel}
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

            # load air temperature
            click.echo("Loading air temperature...")
            ll_air_temperature = []
            for year in years:
                _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
            df_metrics.loc[len(df_metrics)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": value}
            df_anomaly_metrics_abs.loc[len(df_anomaly_metrics_abs)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "mm/day", "metric": "95th_percentile", "value": anomaly_abs}
            df_anomaly_metrics_rel.loc[len(df_anomaly_metrics_rel)] = {"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "actual_evapotranspiration", "unit": "%", "metric": "95th_percentile", "value": anomaly_rel}    


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
                for i in range(_air_temperature_year.shape[0]):
                    air_temperature_day = aggregate_to_coarser_resolution(_air_temperature_year[i, :, :], 25, 50, method="average")
                    air_temperature_day = np.where(mask, air_temperature_day, np.nan)
                    ll_air_temperature.append(air_temperature_day)
            air_temperature = np.stack(ll_air_temperature, axis=0)
            # create xarray data array for air temperature
            da_air_temperature = xr.DataArray(
                data=air_temperature,
                dims=["time", "y", "x"],
                coords={
                    "time": date_time,
                    "y": ycoords,
                    "x": xcoords,
                },
            )
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
                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"ta_{base}_year{year}.nc"
                # _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
                # output_file = base_path_output / f"{stress_test_scenario}" / f"recharge_{_stress_test_scenario}_year{year}.nc"
                ds_air_temperature = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _air_temperature_year = ds_air_temperature["ta"].values
                _air_temperature_year = np.where(_air_temperature_year < -50, np.nan, _air_temperature_year)
                ds_air_temperature.close()
                ll_air_temperature = []
                for i in range(_air_temperature_year.shape[0]):
                    air_temperature_day = aggregate_to_coarser_resolution(_air_temperature_year[i, :, :], 25, 50, method="average")
                    air_temperature_day = np.where(mask, air_temperature_day, np.nan)
                    ll_air_temperature.append(air_temperature_day)
                air_temperature_base = np.stack(ll_air_temperature, axis=0)

                base_path_roger = base_path.parent.parent.parent.parent / "roger"
                output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"ta_{_stress_test_scenario}_year{year}.nc"
                # _stress_test_scenario = stress_test_scenario.replace("_well-extraction-stress", "")
                # output_file = base_path_output / f"{stress_test_scenario}" / f"recharge_{_stress_test_scenario}_year{year}.nc"
                ds_air_temperature = xr.open_dataset(output_file, engine="h5netcdf", decode_timedelta=False)
                _air_temperature_year = ds_air_temperature["ta"].values
                _air_temperature_year = np.where(_air_temperature_year < -50, np.nan, _air_temperature_year)
                ds_air_temperature.close()
                ll_air_temperature = []
                for i in range(_air_temperature_year.shape[0]):
                    air_temperature_day = aggregate_to_coarser_resolution(_air_temperature_year[i, :, :], 25, 50, method="average")
                    air_temperature_day = np.where(mask, air_temperature_day, np.nan)
                    ll_air_temperature.append(air_temperature_day)
                air_temperature = np.stack(ll_air_temperature, axis=0)

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
                    _irrigation_year = aggregate_to_coarser_resolution(_irrigation_year, 25, 50, method="average")
                    ll_irrigation.append(_irrigation_year)
                irrigation = np.stack(ll_irrigation, axis=0)
                irrigation = np.where(irrigation <= 0, np.nan, irrigation)  # set negative values to nan
                # create xarray data array for irrigation
                da_irrigation = xr.DataArray(
                    data=irrigation,
                    dims=["time", "y", "x"],
                    coords={
                        "time": years,
                        "y": ycoords,
                        "x": xcoords,
                    },
                )
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
                    _irrigation_year = aggregate_to_coarser_resolution(_irrigation_year, 25, 50, method="average")
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
                    "y": ycoords,
                    "x": xcoords,
                },
            )

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
                output_file = base_path / "output" / f"modflow_{base}" / f"well_extraction_run{model_run}_year{year}.nc"
                ds_well_extraction = xr.open_dataset(output_file, engine="h5netcdf")
                well_extraction_year = ds_well_extraction["well_extraction"].values
                ds_well_extraction.close()
                well_extraction_base_year = np.where(mask[np.newaxis, :, :], well_extraction_year, 0)
                well_extraction_base_year = np.where(well_extraction_base_year <= 0, np.nan, well_extraction_base_year)  # set negative values to nan

                output_file = base_path / "output" / f"modflow_{stress_test_scenario}" / f"well_extraction_run{model_run}_year{year}.nc"
                ds_well_extraction = xr.open_dataset(output_file, engine="h5netcdf")
                well_extraction_year = ds_well_extraction["well_extraction"].values
                ds_well_extraction.close()
                well_extraction_year = np.where(mask[np.newaxis, :, :], well_extraction_year, 0)
                well_extraction_year = np.where(well_extraction_year <= 0, np.nan, well_extraction_year) 

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

            # remove data arrays to free up memory
            del da_gw_depths, gw_depths
            del da_indirect_recharge, indirect_recharge
            del da_direct_recharge, direct_recharge
            del da_potential_evapotranspiration, potential_evapotranspiration
            del da_air_temperature, air_temperature
            # del da_irrigation, irrigation
            del da_well_extraction, well_extraction
            # remove list of arrays to free up memory
            del ll_gw_depths, ll_indirect_recharge, ll_direct_recharge, ll_well_extraction

            df_metrics = df_metrics.copy()
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.copy()
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.copy()

            # save the metrics to csv
            output_file = base_path / "output" / f"metrics_run{model_run}.csv"
            df_metrics.to_csv(output_file, index=False, sep=";")
            output_file = base_path / "output" / f"anomaly_metrics_abs_run{model_run}.csv"
            df_anomaly_metrics_abs.to_csv(output_file, index=False, sep=";")
            output_file = base_path / "output" / f"anomaly_metrics_rel_run{model_run}.csv"
            df_anomaly_metrics_rel.to_csv(output_file, index=False, sep=";")

    return

if __name__ == "__main__":
    main()
