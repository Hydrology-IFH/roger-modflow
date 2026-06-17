import sys
from pathlib import Path
import flopy
import xarray as xr
import geoxarray
import numpy as np
import pandas as pd
import datetime
import shutil
import tarfile
import click
import os

@click.option("-stm", "--stress-test-meteo", type=click.Choice(["base", "base_2000-2024", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]), default="base", help="Type of meteorological stress test")
@click.option("-stmm", "--stress-test-meteo-magnitude", type=click.Choice([0, 1, 2]), default=0, help="Magnitude of meteorological stress test")
@click.option("-stmd", "--stress-test-meteo-duration", type=click.Choice([0, 2, 3]), default=0, help="Duration of meteorological stress test in consecutive years")
@click.option("-irr", "--irrigation", type=click.Choice(["no-irrigation", "irrigation"]), default="no-irrigation", help="Enable irrigation")
@click.option("-ym", "--yellow-mustard", type=click.Choice(["no-yellow-mustard", "yellow-mustard"]), default="no-yellow-mustard", help="Enable catch crop using yellow mustard")
@click.option("-sc", "--soil-compaction", type=click.Choice(["no-soil-compaction", "soil-compaction"]), default="no-soil-compaction", help="Enable soil compaction")
@click.option("-gco", "--grain-corn-only", type=click.Choice(["no-grain-corn-only", "grain-corn-only"]), default="no-grain-corn-only", help="Enable grain corn monoculture (no crop rotation)")
@click.option("-stwe", "--stress-test-well-extraction", type=click.Choice(["no-stress", "stress", "ta-dependent-20", "ta-dependent-40"]), default="no-stress", help="Enable stress test for well extraction")
@click.option("-mr", "--model-run", type=int, default=944)
@click.command("main")
def main(stress_test_meteo, stress_test_meteo_magnitude, stress_test_meteo_duration, irrigation, yellow_mustard, soil_compaction, grain_corn_only, stress_test_well_extraction, model_run):
    click.echo(sys.version)
    click.echo(f"flopy version: {flopy.__version__}")

    base_path = Path(__file__).parent
    base_path_project = Path("/pfs/10/project/bw22g004/fr_rs1092/workspace-1773831854/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output")

    if grain_corn_only == "no-grain-corn-only":
        _grain_corn_only = ""
    else:
        _grain_corn_only = "_grain-corn-only"

    if stress_test_well_extraction == "no-stress":
        _stress_test_well_extraction = ""
    else:
        _stress_test_well_extraction = "_well-extraction-stress"

    if stress_test_meteo == "base_2000-2024":
        date_time = pd.date_range(start="2000-01-01", end="2024-12-31", freq="D")
        years = np.unique(date_time.year.values)
    else:
        date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
        years = np.unique(date_time.year.values)

    stress_test_name = f"modflow_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}{_stress_test_well_extraction}"

    click.echo(f"Loading MODFLOW6 simulation for model run {model_run}...")
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=base_path / "output" / stress_test_name,
        exe_name="mf6",
        version="mf6",
        verbosity_level=0,
    )

    ml = sim.get_model(f"dmn_run_{model_run}")
    nlayers = np.arange(ml.modelgrid.nlay)

    # load spatial reference and coordinates
    click.echo("Loading spatial reference and coordinates from parameters_modflow.nc...")
    with xr.open_dataset(base_path.parent / "input" / "parameters_modflow.nc") as ds:
        topography = ds['elevations'].isel(z=0).values
        elevation_bottom_layer1 = ds['elevations'].isel(z=1).values
        elevation_bottom_layer2 = ds['elevations'].isel(z=2).values
        elevation_bottom_layer3 = ds['elevations'].isel(z=3).values
        elevation_bottom_layer4 = ds['elevations'].isel(z=4).values
        spatial_ref = ds.spatial_ref
        xcoords = ds.x.values + 25
        ycoords = ds.y.values - 25

    fhead = base_path / "output" / stress_test_name / f"dmn_run_{model_run}.hds"
    click.echo(f"Reading head file {fhead}...")
    hds = flopy.utils.HeadFile(fhead)
    # remove first and last time step (steady-state and final time step)
    heads = hds.get_alldata()[1:, :, :, :]
    ntimesteps = heads.shape[0]
    timesteps = np.arange(ntimesteps)

    fbudget = base_path / "output" / stress_test_name / f"dmn_run_{model_run}.cbc"
    click.echo(f"Reading cell budget file {fbudget}...")
    cbb = flopy.utils.CellBudgetFile(fbudget)

    files_to_compress = []
    for year in years:
        file = base_path / "output" / stress_test_name / f"gw_head_run{model_run}_year{year}.nc"
        if not os.path.exists(file):
            click.echo(f"Processing year {year}...")
            cond_year = (date_time.year == year)
            date_time_year = date_time[cond_year]
            timesteps_year = timesteps[cond_year]  # convert to days since start of the year
            # create xarray dataset
            attrs = dict(
                    date_created=datetime.datetime.today().isoformat(),
                    title="MODFLOW6 transient simulations of the Dreisam-Moehlin-Neumagen catchment (offline coupling with RoGeR)",
                    institution="University of Freiburg, Chair of Hydrology",
                    flopy_version=f"{flopy.__version__}",
                    modflow_version=f"{ml.version}",
                )
            coords = {
                    "lon": ("lon", xcoords),  # x
                    "lat": ("lat", ycoords),  # y
                    "layer": ("layer", nlayers),
                    "Time": ("Time", date_time_year.dayofyear.values - 1, {"units": f"days since {year}-01-01", "calendar": "gregorian"}),
                }
            click.echo("Extracting data for heads,...")
            heads_year = np.where(heads[cond_year, :, :, :] > 10000, np.nan, heads[cond_year, :, :, :])
            thickness_layer1 = np.where(elevation_bottom_layer1 >= topography, 0, topography - elevation_bottom_layer1)
            thickness_layer2 = np.where(elevation_bottom_layer2 >= elevation_bottom_layer1, 0, elevation_bottom_layer1 - elevation_bottom_layer2)
            thickness_layer3 = np.where(elevation_bottom_layer3 >= elevation_bottom_layer2, 0, elevation_bottom_layer2 - elevation_bottom_layer3)
            thickness_layer4 = np.where(elevation_bottom_layer4 >= elevation_bottom_layer3, 0, elevation_bottom_layer3 - elevation_bottom_layer4)
            gw_thickness_layer1 = np.where(heads_year[:, 0, :, :] <= elevation_bottom_layer1, 0, heads_year[:, 0, :, :] - elevation_bottom_layer1)
            gw_thickness_layer2 = np.where(heads_year[:, 1, :, :] <= elevation_bottom_layer2, 0, heads_year[:, 1, :, :] - elevation_bottom_layer2)
            gw_thickness_layer3 = np.where(heads_year[:, 2, :, :] <= elevation_bottom_layer3, 0, heads_year[:, 2, :, :] - elevation_bottom_layer3)
            gw_thickness_layer4 = np.where(heads_year[:, 3, :, :] <= elevation_bottom_layer4, 0, heads_year[:, 3, :, :] - elevation_bottom_layer4)
            gw_thickness_year = np.empty_like(heads_year)
            gw_thickness_year[:, 0, :, :] = gw_thickness_layer1
            gw_thickness_year[:, 1, :, :] = gw_thickness_layer2
            gw_thickness_year[:, 2, :, :] = gw_thickness_layer3
            gw_thickness_year[:, 3, :, :] = gw_thickness_layer4
            gw_thickness_rel_layer1 = gw_thickness_layer1 / thickness_layer1
            gw_thickness_rel_layer2 = gw_thickness_layer2 / thickness_layer2
            gw_thickness_rel_layer3 = gw_thickness_layer3 / thickness_layer3
            gw_thickness_rel_layer4 = gw_thickness_layer4 / thickness_layer4
            gw_thickness_rel_year = np.empty_like(heads_year)
            gw_thickness_rel_year[:, 0, :, :] = gw_thickness_rel_layer1
            gw_thickness_rel_year[:, 1, :, :] = gw_thickness_rel_layer2
            gw_thickness_rel_year[:, 2, :, :] = gw_thickness_rel_layer3
            gw_thickness_rel_year[:, 3, :, :] = gw_thickness_rel_layer4
            click.echo("..., depths,...")
            depths_year = np.where(heads_year > 10000, np.nan, np.where(topography[np.newaxis, np.newaxis, :, :] - heads_year > 0, topography[np.newaxis, np.newaxis, :, :] - heads_year, 0))
            click.echo("... and groundwater-surface water flux...")
            gw_sw_year = np.zeros((len(timesteps_year), len(ycoords), len(xcoords)))
            for i, _timestep_year in enumerate(timesteps_year):
                timestep_year = int(_timestep_year) + 1  # get time step index from timesteps_year
                click.echo(f"Processing time step {timestep_year} for year {year}... (GW-SW flux)")
                try:
                    gw_sw_year[i, :, :] = np.nansum(cbb.get_data(text="SFR", kstpkper=(timestep_year, 1), full3D=True)[0].filled(fill_value=np.nan), axis=0) * (-1)
                except IndexError:
                    click.echo(f"Error occurred while processing time step {timestep_year} for year {year}... (GW-SW flux)")

            well_extraction_year = np.zeros((len(timesteps_year), len(ycoords), len(xcoords)))
            for i, _timestep_year in enumerate(timesteps_year):
                timestep_year = int(_timestep_year) + 1  # get time step index from timesteps_year
                click.echo(f"Processing time step {timestep_year} for year {year}... (well extraction)")
                try:
                    well_extraction_year[i, :, :] = np.nansum(cbb.get_data(text="WEL", kstpkper=(timestep_year, 1), full3D=True)[0].filled(fill_value=np.nan), axis=0) * (-1)
                except IndexError:
                    click.echo(f"Error occurred while processing time step {timestep_year} for year {year}... (well extraction)")

            data_vars=dict(
                    head=(["Time", "layer", "lat", "lon"], heads_year),
                )

            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            ds["head"].attrs["units"] = "m a.s.l."
            ds["head"].attrs["long_name"] = "Groundwater head"
            # create spatial reference
            ds = ds.geo.write_crs("EPSG:25832")
            ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
            file = base_path / "output" / stress_test_name / f"gw_head_run{model_run}_year{year}.nc"
            click.echo(f"Writing {file}...")
            comp = dict(zlib=True, complevel=1)  # compress data to save storage
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
            ds.close()
            files_to_compress.append(file)

            data_vars=dict(
                    thickness=(["Time", "layer", "lat", "lon"], gw_thickness_year),
                )

            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            ds["thickness"].attrs["units"] = "m"
            ds["thickness"].attrs["long_name"] = "Groundwater thickness"
            # create spatial reference
            ds = ds.geo.write_crs("EPSG:25832")
            ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
            file = base_path / "output" / stress_test_name / f"gw_thickness_run{model_run}_year{year}.nc"
            click.echo(f"Writing {file}...")
            comp = dict(zlib=True, complevel=1)  # compress data to save storage
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
            ds.close()
            files_to_compress.append(file)

            data_vars=dict(
                    thickness_rel=(["Time", "layer", "lat", "lon"], gw_thickness_year),
                )

            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            ds["thickness_rel"].attrs["units"] = "-"
            ds["thickness_rel"].attrs["long_name"] = "Relative groundwater thickness"
            # create spatial reference
            ds = ds.geo.write_crs("EPSG:25832")
            ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
            file = base_path / "output" / stress_test_name / f"gw_thickness_rel_run{model_run}_year{year}.nc"
            click.echo(f"Writing {file}...")
            comp = dict(zlib=True, complevel=1)  # compress data to save storage
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
            ds.close()
            files_to_compress.append(file)



            data_vars=dict(
                    depth=(["Time", "layer", "lat", "lon"], depths_year),
                )

            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            ds["depth"].attrs["units"] = "m"
            ds["depth"].attrs["long_name"] = "Groundwater depth"
            # create spatial reference
            ds = ds.geo.write_crs("EPSG:25832")
            ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
            file = base_path / "output" / stress_test_name / f"gw_depth_run{model_run}_year{year}.nc"
            click.echo(f"Writing {file}...")
            comp = dict(zlib=True, complevel=1)  # compress data to save storage
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
            ds.close()
            files_to_compress.append(file)

            data_vars=dict(
                    indirect_recharge=(["Time", "lat", "lon"], gw_sw_year/86400.0),
                )

            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            ds["indirect_recharge"].attrs["units"] = "m3/s"
            ds["indirect_recharge"].attrs["long_name"] = "Recharge from surface water. Negative values indicate surface water leakage into the groundwater."
            # create spatial reference
            ds = ds.geo.write_crs("EPSG:25832")
            ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
            file = base_path / "output" / stress_test_name / f"indirect_recharge_run{model_run}_year{year}.nc"
            click.echo(f"Writing {file}...")
            comp = dict(zlib=True, complevel=1)  # compress data to save storage
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
            ds.close()
            files_to_compress.append(file)

            data_vars=dict(
                    well_extraction=(["Time", "lat", "lon"], well_extraction_year),
                )
            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            ds["well_extraction"].attrs["units"] = "m3/day"
            ds["well_extraction"].attrs["long_name"] = "Groundwater extraction by wells"
            # create spatial reference
            ds = ds.geo.write_crs("EPSG:25832")
            ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
            file = base_path / "output" / stress_test_name / f"well_extraction_run{model_run}_year{year}.nc"
            click.echo(f"Writing {file}...")
            comp = dict(zlib=True, complevel=1)  # compress data to save storage
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
            ds.close()
            files_to_compress.append(file)

    # compress files into a single archive
    if files_to_compress:
        output_file = base_path / "output" / stress_test_name / f"{stress_test_name}_run_{model_run}.tar.gz"
        with tarfile.open(output_file, "w:gz") as tar:
            for f in files_to_compress:
                tar.add(f, arcname=f.name)

    # copy archive to project directory
    output_project_dir = base_path_project / f"{stress_test_name}"
    if not output_project_dir.exists():
        output_project_dir.mkdir(parents=True, exist_ok=True)
    output_file_project = output_project_dir / f"{stress_test_name}_run_{model_run}.tar.gz"
    click.echo(f"Copying {output_file} to {output_file_project}...")
    shutil.copy(output_file, output_file_project)
    # remove archive from work directory after copying
    os.remove(output_file)

    # remove individual files after compressing
    for f in files_to_compress:
        os.remove(f)


    # list all modflow output files
    modflow_files = list((base_path / "output" / stress_test_name).glob(f"dmn_run_{model_run}.*"))
    # copy all modflow files that to save storage
    for file in modflow_files:
        shutil.copy(file, output_project_dir)

    # remove all files from work directory after copying
    for file in modflow_files:
        os.remove(file)

    return


if __name__ == "__main__":
    main()