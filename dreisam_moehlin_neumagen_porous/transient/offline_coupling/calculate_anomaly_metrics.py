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

    areas = ["dmn", "wsg_hausen", "wsg_zartener_becken"]

    base = "base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"

    stress_test_scenarios = ["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude2-duration3_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress"]

    
    date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
    years = np.unique(date_time.year.values)

    # load modflow parameters to get the coordinates of the grid
    click.echo("Loading topography...")
    path = base_path.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    xcoords = ds_params["x"].values
    ycoords = ds_params["y"].values

    df_metrics = pd.DataFrame(columns=["scenario", "area", "time", "variable", "metric", "value"])
    df_anomaly_metrics_abs = pd.DataFrame(columns=["scenario", "area", "time", "variable", "metric", "value"])
    df_anomaly_metrics_rel = pd.DataFrame(columns=["scenario", "area", "time", "variable", "metric", "value"])

    for area in areas:
        if area == "dmn":
            mask = ds_params["mask_porous_aquifer"].values
        elif area == "wsg_hausen":
            file = base_path.parent / "input" / "wsg_hausen_.tif"
            with rasterio.open(file) as src:
                mask = src.read(1)
                mask = np.where(mask == 1, True, False)

        elif area == "wsg_zartener_becken":
            file = base_path.parent / "input" / "wsg_zartener_becken_.tif"
            with rasterio.open(file) as src:
                mask = src.read(1)
                mask = np.where(mask == 1, True, False)

        click.echo(f"Processing scenario {base}...")
        # load the indirect recharge
        click.echo("Loading groundwater depths (base)...")
        ll_gw_depths = []
        for year in years:
            output_file = base_path / "output" / f"modflow_{base}" / f"gw_depth_run{model_run}_year{year}.nc"
            # output_file = base_path_output / f"{base}" / f"gw_depth_run{model_run}_year{year}.nc"
            ds_gw_depths = xr.open_dataset(output_file, engine="h5netcdf")
            gw_depths_year = ds_gw_depths["depth"].values[:, 1, :, :]
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
        value = np.nanmean(np.nanmean(da_gw_depths_base.values, axis=0))
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "metric": "average", "value": value}, ignore_index=True)
        value = np.nanmin(np.nanmean(da_gw_depths_base.values, axis=0))
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "metric": "minimum", "value": value}, ignore_index=True)
        value = np.nanmax(np.nanmean(da_gw_depths_base.values, axis=0))
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "metric": "maximum", "value": value}, ignore_index=True)
        value = np.nanpercentile(np.nanmean(da_gw_depths_base.values, axis=0), 5)
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "metric": "5th_percentile", "value": value}, ignore_index=True)
        value = np.nanpercentile(np.nanmean(da_gw_depths_base.values, axis=0), 50)
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "metric": "median", "value": value}, ignore_index=True)
        value = np.nanpercentile(np.nanmean(da_gw_depths_base.values, axis=0), 95)
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "gw_depth", "metric": "95th_percentile", "value": value}, ignore_index=True)

        click.echo(f"Processing scenario {base}...")
        # load the indirect recharge
        click.echo("Loading indirect recharge...")
        ll_indirect_recharge = []
        for year in years:
            output_file = base_path / "output" / f"modflow_{base}" / f"indirect_recharge_run{model_run}_year{year}.nc"
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
        da_indirect_recharge_base = xr.DataArray(
            data=indirect_recharge,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ds_indirect_recharge["lat"].values,
                "x": ds_indirect_recharge["lon"].values,
            },
        )
        value = np.nanmean(np.nanmean(da_indirect_recharge_base.values, axis=0))
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "average", "value": value}, ignore_index=True)
        value = np.nanmin(np.nanmean(da_indirect_recharge_base.values, axis=0))
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "minimum", "value": value}, ignore_index=True)
        value = np.nanmax(np.nanmean(da_indirect_recharge_base.values, axis=0))
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "maximum", "value": value}, ignore_index=True)
        value = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values, axis=0), 5)
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "5th_percentile", "value": value}, ignore_index=True)
        value = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values, axis=0), 50)
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "median", "value": value}, ignore_index=True)
        value = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values, axis=0), 95)
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "95th_percentile", "value": value}, ignore_index=True)

        # load direct recharge
        click.echo("Loading direct recharge...")
        ll_direct_recharge = []
        for year in years:
            base_path_roger = base_path.parent.parent.parent.parent / "roger"
            output_file = base_path_roger / "examples" / "catchment_scale" / "dreisam_moehlin_neumagen" / "oneD_crop_distributed" / "output" / f"recharge_{base}_year{year}.nc"
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
        da_direct_recharge_base = xr.DataArray(
            data=direct_recharge,
            dims=["time", "y", "x"],
            coords={
                "time": date_time,
                "y": ycoords,
                "x": xcoords,
            },
        )
        value = np.nanmean(np.nanmean(da_direct_recharge_base.values, axis=0))
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "average", "value": value}, ignore_index=True)
        value = np.nanmin(np.nanmean(da_direct_recharge_base.values, axis=0))
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "minimum", "value": value}, ignore_index=True)
        value = np.nanmax(np.nanmean(da_direct_recharge_base.values, axis=0))
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "maximum", "value": value}, ignore_index=True)
        value = np.nanpercentile(np.nanmean(da_direct_recharge_base.values, axis=0), 5)
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "5th_percentile", "value": value}, ignore_index=True)
        value = np.nanpercentile(np.nanmean(da_direct_recharge_base.values, axis=0), 50)
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "50th_percentile", "value": value}, ignore_index=True)  
        value = np.nanpercentile(np.nanmean(da_direct_recharge_base.values, axis=0), 95)
        df_metrics = df_metrics.append({"scenario": base, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "95th_percentile", "value": value}, ignore_index=True)  
        
        # load well extraction
        click.echo("Loading well extraction...")
        ll_well_extraction = []
        for year in years:
            output_file = base_path / "output" / f"modflow_{base}" / f"well_extraction_run{model_run}_year{year}.nc"
            # output_file = base_path_output / f"{stress_test_scenario}" / f"well_extraction_run{model_run}_year{year}.nc"
            ds_well_extraction = xr.open_dataset(output_file, engine="h5netcdf")
            well_extraction_year = ds_well_extraction["well_extraction"].values
            ds_well_extraction.close()
            well_extraction_year = np.where(mask[np.newaxis, :, :], well_extraction_year, 0)
            ll_well_extraction.append(well_extraction_year)
        well_extraction = np.concatenate(ll_well_extraction, axis=0)
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
            value = np.nanmean(np.nanmean(da_gw_depths.values, axis=0))
            value_base = np.nanmean(np.nanmean(da_gw_depths_base.values, axis=0))
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "average", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "average", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "average", "value": anomaly_rel}, ignore_index=True)
            value = np.nanmin(np.nanmean(da_gw_depths.values, axis=0))
            value_base = np.nanmin(np.nanmean(da_gw_depths_base.values, axis=0))
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "minimum", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "minimum", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "minimum", "value": anomaly_rel}, ignore_index=True)
            value = np.nanmax(np.nanmean(da_gw_depths.values, axis=0))
            value_base = np.nanmax(np.nanmean(da_gw_depths_base.values, axis=0))
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "maximum", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "maximum", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "maximum", "value": anomaly_rel}, ignore_index=True)
            value = np.nanpercentile(np.nanmean(da_gw_depths.values, axis=0), 5)
            value_base = np.nanpercentile(np.nanmean(da_gw_depths_base.values, axis=0), 5)
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "5th_percentile", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "5th_percentile", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "5th_percentile", "value": anomaly_rel}, ignore_index=True)    
            value = np.nanpercentile(np.nanmean(da_gw_depths.values, axis=0), 50)
            value_base = np.nanpercentile(np.nanmean(da_gw_depths_base.values, axis=0), 50)
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "median", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "median", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "median", "value": anomaly_rel}, ignore_index=True)
            value = np.nanpercentile(np.nanmean(da_gw_depths.values, axis=0), 95)
            value_base = np.nanpercentile(np.nanmean(da_gw_depths_base.values, axis=0), 95)
            anomaly_abs = (value - value_base) * (-1)
            anomaly_rel = ((value - value_base) * (-1)) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "95th_percentile", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "95th_percentile", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "gw_depth", "metric": "95th_percentile", "value": anomaly_rel}, ignore_index=True)

        
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

            value = np.nanmean(np.nanmean(da_indirect_recharge.values, axis=0))
            value_base = np.nanmean(np.nanmean(da_indirect_recharge_base.values, axis=0))
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "average", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "average", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "average", "value": anomaly_rel}, ignore_index=True)
            value = np.nanmin(np.nanmean(da_indirect_recharge.values, axis=0))
            value_base = np.nanmin(np.nanmean(da_indirect_recharge_base.values, axis=0))
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "minimum", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "minimum", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "minimum", "value": anomaly_rel}, ignore_index=True)
            value = np.nanmax(np.nanmean(da_indirect_recharge.values, axis=0))
            value_base = np.nanmax(np.nanmean(da_indirect_recharge_base.values, axis=0))
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "maximum", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "maximum", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "maximum", "value": anomaly_rel}, ignore_index=True)
            value = np.nanpercentile(np.nanmean(da_indirect_recharge.values, axis=0), 5)
            value_base = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values, axis=0), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "5th_percentile", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "5th_percentile", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "5th_percentile", "value": anomaly_rel}, ignore_index=True)    
            value = np.nanpercentile(np.nanmean(da_indirect_recharge.values, axis=0), 50)
            value_base = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values, axis=0), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "median", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "median", "value": anomaly_abs}, ignore_index=True)   
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "median", "value": anomaly_rel}, ignore_index=True)
            value = np.nanpercentile(np.nanmean(da_indirect_recharge.values, axis=0), 95)
            value_base = np.nanpercentile(np.nanmean(da_indirect_recharge_base.values, axis=0), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "95th_percentile", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "95th_percentile", "value": anomaly_abs}, ignore_index=True)   
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "indirect_recharge", "metric": "95th_percentile", "value": anomaly_rel}, ignore_index=True)

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
            value = np.nanmean(np.nanmean(da_direct_recharge.values, axis=0))
            value_base = np.nanmean(np.nanmean(da_direct_recharge_base.values, axis=0))
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "average", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "average", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "average", "value": anomaly_rel}, ignore_index=True)
            value = np.nanmin(np.nanmean(da_direct_recharge.values, axis=0))
            value_base = np.nanmin(np.nanmean(da_direct_recharge_base.values, axis=0))
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "minimum", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "minimum", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "minimum", "value": anomaly_rel}, ignore_index=True)
            value = np.nanmax(np.nanmean(da_direct_recharge.values, axis=0))
            value_base = np.nanmax(np.nanmean(da_direct_recharge_base.values, axis=0))
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "maximum", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "maximum", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "maximum", "value": anomaly_rel}, ignore_index=True)
            value = np.nanpercentile(np.nanmean(da_direct_recharge.values, axis=0), 5)
            value_base = np.nanpercentile(np.nanmean(da_direct_recharge_base.values, axis=0), 5)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "5th_percentile", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "5th_percentile", "value": anomaly_abs}, ignore_index=True)
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "5th_percentile", "value": anomaly_rel}, ignore_index=True)    
            value = np.nanpercentile(np.nanmean(da_direct_recharge.values, axis=0), 50)
            value_base = np.nanpercentile(np.nanmean(da_direct_recharge_base.values, axis=0), 50)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "median", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "median", "value": anomaly_abs}, ignore_index=True)   
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "median", "value": anomaly_rel}, ignore_index=True)
            value = np.nanpercentile(np.nanmean(da_direct_recharge.values, axis=0), 95)
            value_base = np.nanpercentile(np.nanmean(da_direct_recharge_base.values, axis=0), 95)
            anomaly_abs = value - value_base
            anomaly_rel = (value - value_base) / value_base * 100 if value_base != 0 else np.nan
            df_metrics = df_metrics.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "95th_percentile", "value": value}, ignore_index=True)
            df_anomaly_metrics_abs = df_anomaly_metrics_abs.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "95th_percentile", "value": anomaly_abs}, ignore_index=True)   
            df_anomaly_metrics_rel = df_anomaly_metrics_rel.append({"scenario": stress_test_scenario, "area": area, "time": "overall", "variable": "direct_recharge", "metric": "95th_percentile", "value": anomaly_rel}, ignore_index=True)

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

            # remove data arrays to free up memory
            del da_gw_depths
            del da_indirect_recharge
            del da_direct_recharge
            del da_well_extraction
            # remove list of arrays to free up memory
            del ll_gw_depths, ll_indirect_recharge, ll_direct_recharge, ll_well_extraction

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
