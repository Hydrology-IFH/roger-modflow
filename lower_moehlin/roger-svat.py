from time import time
from pathlib import Path
import os
import numpy as np
from xmipy import XmiWrapper
import flopy
from flopy.utils import Raster
import yaml
import xarray as xr
import xesmf as xe
import click
import shutil
import subprocess


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


def aggregate_to_finer_resolution(vals, res_coarse, res_fine, method="keep", x_origin=0, y_origin=0):
    """Aggregate raster data to a finer resolution.
        
    Args
    ----
    vals : numpy.ndarray
        2D array with the raster data.

    res_coarse : int
        spatial resolution of the coarse grid in meters.

    res_fine : int
        spatial resolution of the fine grid in meters.

    method : str
        Method to aggregate the data. Options are "keep", "interpolate" and "conservative".

    x_origin : int
        x-coordinate of the grid origin (lower left corner).

    y_origin : int
        y-coordinate of the grid origin (lower left corner).  
    """
    ny_coarse, nx_coarse = vals.shape[0], vals.shape[1]
    nx_fine = int(nx_coarse * (res_coarse / res_fine))
    ny_fine = int(ny_coarse * (res_coarse / res_fine))
    meters_to_latlon = 111195
    if method == "keep":
        lat_fine = np.linspace(y_origin, y_origin + ny_fine*(res_fine/meters_to_latlon), ny_fine)/meters_to_latlon  # boundaries
        lon_fine = np.linspace(x_origin, x_origin + nx_fine*(res_fine/meters_to_latlon), nx_fine)/meters_to_latlon  # boundaries
        lat_coarse = np.linspace(y_origin, y_origin + ny_coarse*(res_coarse/meters_to_latlon), ny_coarse)/meters_to_latlon  # boundaries
        lon_coarse = np.linspace(x_origin, x_origin + nx_coarse*(res_coarse/meters_to_latlon), nx_coarse)/meters_to_latlon  # boundaries
        grid_coarse = {"lon": lon_coarse, "lat": lat_coarse}
        grid_fine = {"lon": lon_fine, "lat": lat_fine}
        regridder = xe.Regridder(grid_coarse, grid_fine, "nearest_s2d")
    elif method == "interpolate":
        lat_fine = np.linspace(y_origin, y_origin + ny_fine*(res_fine/meters_to_latlon), ny_fine)/meters_to_latlon  # boundaries
        lon_fine = np.linspace(x_origin, x_origin + nx_fine*(res_fine/meters_to_latlon), nx_fine)/meters_to_latlon  # boundaries
        lat_coarse = np.linspace(y_origin, y_origin + ny_coarse*(res_coarse/meters_to_latlon), ny_coarse)/meters_to_latlon  # boundaries
        lon_coarse = np.linspace(x_origin, x_origin + nx_coarse*(res_coarse/meters_to_latlon), nx_coarse)/meters_to_latlon  # boundaries
        grid_coarse = {"lon": lon_coarse, "lat": lat_coarse}
        grid_fine = {"lon": lon_fine, "lat": lat_fine}
        regridder = xe.Regridder(grid_coarse, grid_fine, "bilinear")
    elif method == "conservative":
        lat_fine = np.linspace(y_origin, y_origin + (ny_fine + 1) * (res_fine/meters_to_latlon), ny_fine + 1)/meters_to_latlon  # boundaries
        lon_fine = np.linspace(x_origin, x_origin + (nx_fine + 1) * (res_fine/meters_to_latlon), nx_fine + 1)/meters_to_latlon  # boundaries
        lat_coarse = np.linspace(y_origin, y_origin + (ny_coarse + 1) * (res_coarse/meters_to_latlon), ny_coarse + 1)/meters_to_latlon  # boundaries
        lon_coarse = np.linspace(x_origin, x_origin + (nx_coarse + 1) * (res_coarse/meters_to_latlon), nx_coarse + 1)/meters_to_latlon  # boundaries
        lat_fine_centers = (0.5 * (lat_fine[1] - lat_fine[0])) + lat_fine[:-1]  # centers
        lat_coarse_centers = (0.5 * (lat_coarse[1] + lat_coarse[0])) + lat_coarse[:-1]  # centers
        lon_fine_centers = (0.5 * (lon_fine[1] + lon_fine[0])) + lon_fine[:-1]  # centers
        lon_coarse_centers = (0.5 * (lon_coarse[1] + lon_coarse[0])) + lon_coarse[:-1]  # centers
        grid_coarse = {"lon": lon_coarse_centers, "lon_b": lon_coarse, "lat": lat_coarse_centers, "lat_b": lat_coarse}
        grid_fine = {"lon": lon_fine_centers, "lon_b": lon_fine, "lat": lat_fine_centers, "lat_b": lat_fine}
        regridder = xe.Regridder(grid_coarse, grid_fine, "conservative")

    data = regridder(vals)
    return data


base_path = Path(__file__).parent


@click.option("-b", "--backend", type=click.Choice(["numpy", "jax"]), default="jax", help="Computational backend of RoGeR")
@click.option("-ft", "--float_type", type=click.Choice(["float32", "float64"]), default="float32", help="Float type of RoGeR")
@click.command("main", short_help="Run MODFLOW in transient mode coupled with RoGeR.")
def main(backend, float_type):
    from roger import runtime_settings
    runtime_settings.update(
    backend=backend,
    float_type=float_type,
    )
    from bmiroger import BmiRoger
    from roger.bmimodels.svat import SVATSetup
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        roger_config = yaml.safe_load(file)

    # define the output variables of RoGeR
    roger_config['OUTPUT_COLLECT'] = ["theta", "z_gw"]
    roger_config['OUTPUT_RATE'] = ["q_hof", "q_ss"]

    # choose parameters depending on the resolution of RoGeR
    file1 = base_path / "input" / f"parameters_{int(roger_config['dx'])}.nc"
    file2 = base_path / "parameters.nc"
    shutil.copy(file1, file2)
    file = base_path / "write_parameters_to_csv_for_SVAT.py"
    subprocess.run(["python", str(file)], check=True, timeout=20)

    # set the number of grid cells in x and y direction from the parameters file
    file = base_path / "parameters.nc"
    with xr.open_dataset(file, engine="h5netcdf") as ds:
        roger_config['nx'] = ds.sizes['y']
        roger_config['ny'] = ds.sizes['x']

    # save the updated config file
    with open(file_config, "w") as file:
        yaml.dump(roger_config, file)

    res_modflow = 50  # spatial resolution of MODFLOW in meters

    modflow_config = {
        'dx': int(roger_config['dx']*(res_modflow/roger_config['dx'])),
        'dy': int(roger_config['dy']*(res_modflow/roger_config['dx'])),
        'nx': int(roger_config['nx']/(res_modflow/roger_config['dx'])),
        'ny': int(roger_config['ny']/(res_modflow/roger_config['dx'])),
        'nz': 4,
    }

    # load the spatial domain and elevation data
    file = base_path / "input" / "domain.grd"
    domain = Raster.load(file)
    modflow_mask = (domain.get_array(1)[:, :-1] == 1)
    file = base_path / "input" / "elevation.grd"
    layer_elevations = Raster.load(file)
    topography = layer_elevations.get_array(1)[:, :-1]
    topography = aggregate_to_finer_resolution(topography, modflow_config['dx'], roger_config['dx'], method="keep")

    # initialize the SVAT model of RoGeR using BMI
    model = SVATSetup(base_path)
    roger_interface = BmiRoger(model=model)
    roger_interface._model._output_dir = base_path / "output" / "transient"
    roger_interface.initialize(base_path)
    print("RoGeR model initialized")
    NDAYS = int(roger_interface.get_end_time() / (60 * 60 * 24))  # seconds to days

    soildepth = np.zeros(roger_interface.get_grid_node_count())
    roger_interface.get_value("z_soil", soildepth)
    soildepth = soildepth / 1000  # mm to m
    roger_mask = np.empty(roger_interface.get_grid_node_count(), dtype=bool)
    roger_interface.get_value("maskCatch", roger_mask)
    roger_mask = roger_mask.reshape(roger_config['nx'], roger_config['ny'])
    
    for _ in range(NDAYS):
        # # update groundwater head
        # groundwater_head = np.zeros(modflow_config['nx'] * modflow_config['ny'])
        # groundwater_head = groundwater_head.reshape(modflow_config['nx'], modflow_config['ny'])
        # # aggregate groundwater head to the resolution of RoGeR
        # groundwater_head = aggregate_to_finer_resolution(groundwater_head, modflow_config['dx'], roger_config['dx'], method="keep")
        # # RoGeR requires depth of groundwater head (in meters)
        # groundwater_depth = topography.flatten() - groundwater_head.flatten()
        # groundwater_depth[(groundwater_depth <= soildepth)] = soildepth[(groundwater_depth <= soildepth)] + 0.05  # constrain groundwater depth to soil depth
        # with roger_interface._model.state.variables.unlock():
        #     roger_interface.set_value("z_gw", groundwater_depth)

        # run RoGeR for one timestep
        roger_interface.update_until(roger_interface._model._config["OUTPUT_FREQUENCY"])

        # # update recharge and pass it to MODFLOW
        # recharge = np.zeros(roger_interface.get_grid_node_count())
        # roger_interface.get_value("q_ss", recharge)
        # recharge[(groundwater_depth <= soildepth)] = 0 # constrain recharge to zero where groundwater depth is equal to soil depth
        # recharge = recharge.reshape(roger_config['nx'], roger_config['ny']).astype(np.float64) / 1000  # mm/day to m/day
        # recharge = aggregate_to_coarser_resolution(recharge, roger_config['dx'], modflow_config['dx'], method="average")

    roger_interface.finalize()
    print("RoGeR and MODFLOW (transient) finalized")

if __name__ == "__main__":
    main()

# import matplotlib.pyplot as plt
# grid_extent = (0, 404 * 25, 0, 356 * 25)

# fig, axes = plt.subplots(figsize=(4, 4))
# topography[~roger_mask] = np.nan
# plt.imshow(topography, extent=grid_extent, cmap='viridis', aspect='equal', vmin=200, vmax=1200)
# plt.colorbar(label='[m a.s.l.]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path / "figures" / "debug_topo.png"
# fig.savefig(file, dpi=300)

# fig, axes = plt.subplots(figsize=(4, 4))
# groundwater_head[~roger_mask] = np.nan
# plt.imshow(groundwater_head, extent=grid_extent, cmap='viridis', aspect='equal', vmin=200, vmax=1200)
# plt.colorbar(label='[m a.s.l.]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path / "figures" / "debug_gw_head.png"
# fig.savefig(file, dpi=300)

# fig, axes = plt.subplots(figsize=(4, 4))
# plt.imshow(topography - groundwater_head, extent=grid_extent, cmap='viridis', aspect='equal')
# plt.colorbar(label='[m a.s.l.]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path / "figures" / "debug_gw_depth.png"
# fig.savefig(file, dpi=300)

# fig, axes = plt.subplots(figsize=(4, 4))
# recharge = recharge.reshape(modflow_config['nx'], modflow_config['ny']) * 1000
# recharge[~modflow_mask] = np.nan
# plt.imshow(recharge, extent=grid_extent, cmap='viridis', aspect='equal')
# plt.colorbar(label='[mm/day]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path / "figures" / "debug_recharge.png"
# fig.savefig(file, dpi=300)

# plt.close("all")
