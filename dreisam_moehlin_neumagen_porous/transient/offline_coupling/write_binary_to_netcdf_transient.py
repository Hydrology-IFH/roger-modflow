import sys
from pathlib import Path
import flopy
import xarray as xr
import geoxarray
import numpy as np
import pandas as pd
import datetime
import zlib
import struct
import click
import os

def compress_files(file_list, output_file, compression_level=6):
    """
    Compress multiple files into a single archive using zlib.
    
    Args:
        file_list: List of file paths to compress
        output_file: Output archive file path
        compression_level: Compression level (0-9, where 9 is maximum compression)
    """
    output_dir = Path(file_list[0]).parent
    zip_file = str(output_dir / output_file)
    with open(zip_file, 'wb') as out:
        # Write number of files
        out.write(struct.pack('I', len(file_list)))
        
        for file_path in file_list:
            file_path = Path(file_path)
            
            if not file_path.exists():
                click.echo(f"Warning: {file_path} not found, skipping...")
                continue
            
            # Read file content
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Compress the data
            compressed_data = zlib.compress(data, compression_level)
            
            # Get file metadata
            file_name = str(file_path.name)
            file_name_bytes = file_name.encode('utf-8')
            original_size = len(data)
            compressed_size = len(compressed_data)
            
            # Write metadata and compressed data
            # Format: [name_length][name][original_size][compressed_size][compressed_data]
            out.write(struct.pack('I', len(file_name_bytes)))  # Name length
            out.write(file_name_bytes)                          # File name
            out.write(struct.pack('Q', original_size))          # Original size
            out.write(struct.pack('Q', compressed_size))        # Compressed size
            out.write(compressed_data)                          # Compressed data
            
            # click.echo compression stats
            ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            click.echo(f"Compressed: {file_name}")
            click.echo(f"  Original: {original_size:,} bytes")
            click.echo(f"  Compressed: {compressed_size:,} bytes")
            click.echo(f"  Ratio: {ratio:.2f}%\n")


@click.option("-stm", "--stress-test-meteo", type=click.Choice(["base", "base_2000-2024", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]), default="base", help="Type of meteorological stress test")
@click.option("-stmm", "--stress-test-meteo-magnitude", type=click.Choice([0, 1, 2]), default=0, help="Magnitude of meteorological stress test")
@click.option("-stmd", "--stress-test-meteo-duration", type=click.Choice([0, 2, 3]), default=0, help="Duration of meteorological stress test in consecutive years")
@click.option("-irr", "--irrigation", type=click.Choice(["no-irrigation", "irrigation"]), default="no-irrigation", help="Enable irrigation")
@click.option("-ym", "--yellow-mustard", type=click.Choice(["no-yellow-mustard", "yellow-mustard"]), default="no-yellow-mustard", help="Enable catch crop using yellow mustard")
@click.option("-sc", "--soil-compaction", type=click.Choice(["no-soil-compaction", "soil-compaction"]), default="no-soil-compaction", help="Enable soil compaction")
@click.option("-gco", "--grain-corn-only", type=click.Choice(["no-grain-corn-only", "grain-corn-only"]), default="no-grain-corn-only", help="Enable grain corn monoculture (no crop rotation)")
@click.option("-stwe", "--stress-test-well-extraction", type=click.Choice(["no-stress", "stress", "ta-dependent-20", "ta-dependent-40"]), default="no-stress", help="Enable stress test for well extraction")
@click.option("-mr", "--model-run", type=int, default=1806)
@click.command("main")
def main(stress_test_meteo, stress_test_meteo_magnitude, stress_test_meteo_duration, irrigation, yellow_mustard, soil_compaction, grain_corn_only, stress_test_well_extraction, model_run):
    click.echo(sys.version)
    click.echo(f"flopy version: {flopy.__version__}")

    base_path = Path(__file__).parent

    if grain_corn_only == "no-grain-corn-only":
        _grain_corn_only = ""
    else:
        _grain_corn_only = "_grain-corn-only"

    if stress_test_well_extraction == "no-stress":
        _stress_test_well_extraction = ""
    else:
        _stress_test_well_extraction = "_well-extraction-stress"

    if stress_test_meteo == "base_2000-2024":
        date_time = pd.date_range(start="2000-01-01", end="2024-12-31", freq="7D")
        years = np.unique(date_time.year.values)
        timesteps = np.arange(len(date_time)) * 7
    else:
        date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="7D")
        years = np.unique(date_time.year.values)
        timesteps = np.arange(len(date_time)) * 7

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
        spatial_ref = ds.spatial_ref
        xcoords = ds.x.values
        ycoords = ds.y.values

    fhead = base_path / "output" / stress_test_name / f"dmn_run_{model_run}.hds"
    click.echo(f"Reading head file {fhead}...")
    hds = flopy.utils.HeadFile(fhead)
    # remove first and last time step (steady-state and final time step)
    heads = hds.get_alldata()[1:-1, :, :, :]
    ntimesteps = heads.shape[0]
    timesteps = np.arange(ntimesteps)

    fbudget = base_path / "output" / stress_test_name / f"dmn_run_{model_run}.cbc"
    click.echo(f"Reading cell budget file {fbudget}...")
    cbb = flopy.utils.CellBudgetFile(fbudget)

    files_to_compress = []
    for year in years:
        file = base_path / "output" / stress_test_name / f"dmn_run_{model_run}_year{year}.nc"
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
                    "Time": ("Time", timesteps_year, {"units": f"days since {year}-01-01", "calendar": "gregorian"}),
                }
            click.echo("Extracting data for heads,...")
            heads_year = np.where(heads[cond_year, :, :, :] > 10000, np.nan, heads[cond_year, :, :, :])
            click.echo("..., depths,...")
            depths_year = np.where(heads_year > 10000, np.nan, np.where(topography[np.newaxis, np.newaxis, :, :] - heads_year > 0, topography[np.newaxis, np.newaxis, :, :] - heads_year, 0))
            click.echo("... and groundwater-surface water flux...")
            gw_sw_year = np.zeros((len(timesteps_year), len(ycoords), len(xcoords)))
            for i, _timestep_year in enumerate(timesteps_year):
                timestep_year = int(_timestep_year)  # get time step index from timesteps_year
                timestep_year = cbb.get_kstpkper()[i+1][0]  # get time step index from cell budget file (add 1 to skip steady-state time step)
                click.echo(f"Processing time step {timestep_year} for year {year}... (GW-SW flux)")
                gw_sw_year[i, :, :] = np.nansum(cbb.get_data(text="SFR", kstpkper=(timestep_year, 1), full3D=True)[0].filled(fill_value=np.nan), axis=0)

            data_vars=dict(
                    head=(["Time", "layer", "lat", "lon"], heads_year),
                    depth=(["Time", "layer", "lat", "lon"], depths_year),
                    gw_sw=(["Time", "lat", "lon"], gw_sw_year/86400.0),
                )

            ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
            ds["head"].attrs["units"] = "m a.s.l."
            ds["head"].attrs["long_name"] = "Groundwater head"
            ds["depth"].attrs["units"] = "m"
            ds["depth"].attrs["long_name"] = "Groundwater depth"
            ds["gw_sw"].attrs["units"] = "m3/s"
            ds["gw_sw"].attrs["long_name"] = "Groundwater-Surface water flux"
            # create spatial reference
            ds = ds.geo.write_crs("EPSG:25832")
            ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
            click.echo(f"Writing {file}...")
            comp = dict(zlib=True, complevel=1)  # compress data to save storage
            encoding = {var: comp for var in ds.data_vars}
            ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
            files_to_compress.append(file)

    # compress files into a single archive
    if files_to_compress:
        output_file = f"{stress_test_name}_dmn_run_{model_run}.zlb"
        click.echo(f"Compressing files into {output_file}...")
        compress_files(files_to_compress, output_file, compression_level=9)
    return


if __name__ == "__main__":
    main()