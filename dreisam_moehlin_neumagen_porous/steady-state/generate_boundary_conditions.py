import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import pandas as pd
import h5netcdf
import xarray as xr
import rasterio
import datetime
import yaml
from pysheds.grid import Grid
import click

def lateral_inflow_bc_mfdir(mask_bc, outflow_dir, head, topography, bottom_elevation, kf, dx=50, dy=50):
    """
    Calculate lateral inflow based on multiple flow direction and head values.

    Parameters
    ----------
    mask_bc : 2D array
        Boolean mask array indicating boundary condition locations.
    outflow_dir : 3D array
        Outflow direction array following D8 convention.
    head : 2D array
        Groundwater head values.
    topography : 2D array
        Topography values.
    bottom_elevation : 2D array
        Bottom elevation values.
    dx : float
        Grid cell size in x-direction.
    dy : float
        Grid cell size in y-direction.

    Returns
    -------
    inflow_dir : 2D array
        Lateral inflow direction array.
    delta_gw_head : 2D array
        Difference in groundwater head between inflow cell and boundary cell.
    hydraulic_gradient : 2D array
        Hydraulic gradient array.
    inflow_m3_s : 2D array
        Lateral inflow in cubic meters per second.
    inflow_mm_day : 2D array
        Lateral inflow in millimeters per day.
    """
    mask_bc[0, :] = 0
    mask_bc[-1, :] = 0
    mask_bc[:, 0] = 0
    mask_bc[:, -1] = 0
    rows_bc, cols_bc = np.where(mask_bc == 1)
    inflow_dir = np.zeros_like(outflow_dir)
    delta_gw_head = np.zeros_like(outflow_dir)
    hydraulic_gradient = np.zeros_like(outflow_dir)
    flow_area = np.zeros_like(outflow_dir)
    lateral_grid_cell_area = np.zeros_like(outflow_dir)
    for y, x in zip(rows_bc, cols_bc):
        for i in range(0, 8):
            if outflow_dir[i, y, x+1] > 0 and not mask_bc[y, x+1]:
                inflow_dir[i, y, x] = outflow_dir[i, y, x+1] 
                delta_gw_head[i, y, x] = (head[y, x+1] - head[y, x])
                hydraulic_gradient[i, y, x] = (delta_gw_head[i, y, x] / dx)
                flow_area[i, y, x] = ((head[y, x] - bottom_elevation[y, x]) * dx)
                lateral_grid_cell_area[i, y, x] = ((topography[y, x] - bottom_elevation[y, x]) * dx)
            elif outflow_dir[i, y, x-1] > 0 and not mask_bc[y, x-1]:
                inflow_dir[i, y, x] = outflow_dir[i, y, x-1]
                delta_gw_head[i, y, x] = (head[y, x-1] - head[y, x])
                hydraulic_gradient[i, y, x] = (delta_gw_head[i, y, x] / dx)
                flow_area[i, y, x] = ((head[y, x] - bottom_elevation[y, x]) * dx)
                lateral_grid_cell_area[i, y, x] = ((topography[y, x] - bottom_elevation[y, x]) * dx)
            elif outflow_dir[i, y+1, x] > 0 and not mask_bc[y+1, x]:
                inflow_dir[i, y, x] = outflow_dir[i, y+1, x]
                delta_gw_head[i, y, x] = (head[y+1, x] - head[y, x])
                hydraulic_gradient[i, y, x] = (delta_gw_head[i, y, x] / dy)
                flow_area[i, y, x] = ((head[y, x] - bottom_elevation[y, x]) * dx)
                lateral_grid_cell_area[i, y, x] = ((topography[y, x] - bottom_elevation[y, x]) * dx)
            elif outflow_dir[i, y-1, x] > 0 and not mask_bc[y-1, x]:
                inflow_dir[i, y, x] = outflow_dir[i, y-1, x]
                delta_gw_head[i, y, x] = (head[y-1, x] - head[y, x])
                hydraulic_gradient[i, y, x] = (delta_gw_head[i, y, x] / dy)
                flow_area[i, y, x] = ((head[y, x] - bottom_elevation[y, x]) * dx)
                lateral_grid_cell_area[i, y, x] = ((topography[y, x] - bottom_elevation[y, x]) * dx)
            elif outflow_dir[i, y+1, x+1] > 0 and not mask_bc[y+1, x+1]:
                inflow_dir[i, y, x] = outflow_dir[i, y+1, x+1]
                delta_gw_head[i, y, x] = (head[y+1, x+1] - head[y, x])
                hydraulic_gradient[i, y, x] = (delta_gw_head[i, y, x] / np.sqrt(dx**2 + dy**2))
                flow_area[i, y, x] = ((head[y, x] - bottom_elevation[y, x]) * dx)
                lateral_grid_cell_area[i, y, x] = ((topography[y, x] - bottom_elevation[y, x]) * dx)
            elif outflow_dir[i, y-1, x-1] > 0 and not mask_bc[y-1, x-1]:
                inflow_dir[i, y, x] = outflow_dir[i, y-1, x-1] 
                delta_gw_head[i, y, x] = (head[y-1, x-1] - head[y, x])
                hydraulic_gradient[i, y, x] = (delta_gw_head[i, y, x] / np.sqrt(dx**2 + dy**2))
                flow_area[i, y, x] = ((head[y, x] - bottom_elevation[y, x]) * dx)
                lateral_grid_cell_area[i, y, x] = ((topography[y, x] - bottom_elevation[y, x]) * dx)
            elif outflow_dir[i, y+1, x-1] > 0 and not mask_bc[y+1, x-1]:
                inflow_dir[i, y, x] = outflow_dir[i, y+1, x-1]
                delta_gw_head[i, y, x] = (head[y+1, x-1] - head[y, x])
                hydraulic_gradient[i, y, x] = (delta_gw_head[i, y, x] / np.sqrt(dx**2 + dy**2))
                flow_area[i, y, x] = ((head[y, x] - bottom_elevation[y, x]) * dx)
                lateral_grid_cell_area[i, y, x] = ((topography[y, x] - bottom_elevation[y, x]) * dx)
            elif outflow_dir[i, y-1, x+1] > 0 and not mask_bc[y-1, x+1]:
                inflow_dir[i, y, x] = outflow_dir[i, y-1, x+1]
                delta_gw_head[i, y, x] = (head[y-1, x+1] - head[y, x])
                hydraulic_gradient[i, y, x] = (delta_gw_head[i, y, x] / np.sqrt(dx**2 + dy**2))
                flow_area[i, y, x] = ((head[y, x] - bottom_elevation[y, x]) * dx)
                lateral_grid_cell_area[i, y, x] = ((topography[y, x] - bottom_elevation[y, x]) * dx)

    # constrain hydraulic gradient to groundwater heads below surface
    hydraulic_gradient = np.where(hydraulic_gradient <= 0, 0.001, hydraulic_gradient)
    hydraulic_gradient = np.where(hydraulic_gradient > 0.3, 0.3, hydraulic_gradient)

    # normalise contributions of inflow directions
    inflow_dir_norm = inflow_dir / np.sum(inflow_dir, axis=0)[np.newaxis, :, :]
    # apply weighting to hydraulic gradient, flow area and lateral grid cell area
    _delta_gw_head = delta_gw_head * inflow_dir_norm
    _hydraulic_gradient = hydraulic_gradient * inflow_dir_norm
    _flow_area = flow_area * inflow_dir_norm
    _lateral_grid_cell_area = lateral_grid_cell_area * inflow_dir_norm

    # calculate inflow from all directions
    inflow_m3_s_ = np.where(mask_bc[np.newaxis, :, :], kf * _hydraulic_gradient * _flow_area, np.nan)
    inflow_mm_day_ = np.where(mask_bc[np.newaxis, :, :], inflow_m3_s_ * (86400 / _lateral_grid_cell_area) * 1000, np.nan)
    # calculate weighted average
    hydraulic_gradient = np.nansum(_hydraulic_gradient, axis=0)
    delta_gw_head = np.nansum(_delta_gw_head, axis=0)
    inflow_m3_s = np.nansum(inflow_m3_s_, axis=0)
    inflow_mm_day = np.nansum(inflow_mm_day_, axis=0)

    hydraulic_gradient = np.where(mask_bc, hydraulic_gradient, np.nan)
    delta_gw_head = np.where(mask_bc, delta_gw_head, np.nan)
    inflow_m3_s = np.where(mask_bc, inflow_m3_s, np.nan)
    inflow_mm_day = np.where(mask_bc, inflow_mm_day, 0)

    return inflow_dir_norm, delta_gw_head, hydraulic_gradient, inflow_m3_s, inflow_mm_day   


def lateral_inflow_bc_d8(mask_bc, outflow_dir, head, topography, bottom_elevation, kf, dx=50, dy=50):
    """
    Calculate lateral inflow based on D8 flow direction and head values.

    Parameters
    ----------
    mask_bc : 2D array
        Boolean mask array indicating boundary condition locations.
    outflow_dir : 2D array
        Outflow direction array following D8 convention.
    head : 2D array
        Groundwater head values.
    topography : 2D array
        Topography values.
    bottom_elevation : 2D array
        Bottom elevation values.
    dx : float
        Grid cell size in x-direction.
    dy : float
        Grid cell size in y-direction.

    Returns
    -------
    inflow_dir : 2D array
        Lateral inflow direction array.
    delta_gw_head : 2D array
        Difference in groundwater head between inflow cell and boundary cell.
    hydraulic_gradient : 2D array
        Hydraulic gradient array.
    inflow_m3_s : 2D array
        Lateral inflow in cubic meters per second.
    inflow_mm_day : 2D array
        Lateral inflow in millimeters per day.
    """
    mask_bc[0, :] = 0
    mask_bc[-1, :] = 0
    mask_bc[:, 0] = 0
    mask_bc[:, -1] = 0
    rows_bc, cols_bc = np.where(mask_bc == 1)
    inflow_dir = np.zeros_like(head)
    delta_gw_head = np.zeros_like(head)
    hydraulic_gradient = np.zeros_like(head)
    flow_area = np.zeros_like(head)
    for y, x in zip(rows_bc, cols_bc):
        if outflow_dir[y, x+1] == 1 and not mask_bc[y, x+1]:
            inflow_dir[y, x] = 16
            delta_gw_head[y, x] = head[y, x+1] - head[y, x]
            hydraulic_gradient[y, x] = delta_gw_head[y, x] / dx
            flow_area[y, x] = (head[y, x] - bottom_elevation[y, x]) * dx
        elif outflow_dir[y, x-1] == 16 and not mask_bc[y, x-1]:
            inflow_dir[y, x] = 1
            delta_gw_head[y, x] = head[y, x-1] - head[y, x]
            hydraulic_gradient[y, x] = delta_gw_head[y, x] / dx
            flow_area[y, x] = (head[y, x] - bottom_elevation[y, x]) * dx
        elif outflow_dir[y+1, x] == 4 and not mask_bc[y+1, x]:
            inflow_dir[y, x] = 64
            delta_gw_head[y, x] = head[y+1, x] - head[y, x]
            hydraulic_gradient[y, x] = delta_gw_head[y, x] / dy
            flow_area[y, x] = (head[y, x] - bottom_elevation[y, x]) * dx
        elif outflow_dir[y-1, x] == 64 and not mask_bc[y-1, x]:
            inflow_dir[y, x] = 4
            delta_gw_head[y, x] = head[y-1, x] - head[y, x]
            hydraulic_gradient[y, x] = delta_gw_head[y, x] / dy
            flow_area[y, x] = (head[y, x] - bottom_elevation[y, x]) * dx
        elif outflow_dir[y+1, x+1] == 2 and not mask_bc[y+1, x+1]:
            inflow_dir[y, x] = 32
            delta_gw_head[y, x] = head[y+1, x+1] - head[y, x]
            hydraulic_gradient[y, x] = delta_gw_head[y, x] / np.sqrt(dx**2 + dy**2)
            flow_area[y, x] = (head[y, x] - bottom_elevation[y, x]) * dx
        elif outflow_dir[y-1, x-1] == 32 and not mask_bc[y-1, x-1]:
            inflow_dir[y, x] = 2
            delta_gw_head[y, x] = head[y-1, x-1] - head[y, x]
            hydraulic_gradient[y, x] = delta_gw_head[y, x] / np.sqrt(dx**2 + dy**2)
            flow_area[y, x] = (head[y, x] - bottom_elevation[y, x]) * dx
        elif outflow_dir[y+1, x-1] == 8 and not mask_bc[y+1, x-1]:
            inflow_dir[y, x] = 128
            delta_gw_head[y, x] = head[y+1, x-1] - head[y, x]
            hydraulic_gradient[y, x] = delta_gw_head[y, x] / np.sqrt(dx**2 + dy**2)
            flow_area[y, x] = (head[y, x] - bottom_elevation[y, x]) * dx
        elif outflow_dir[y-1, x+1] == 128 and not mask_bc[y-1, x+1]:
            inflow_dir[y, x] = 8
            delta_gw_head[y, x] = head[y-1, x+1] - head[y, x]
            hydraulic_gradient[y, x] = delta_gw_head[y, x] / np.sqrt(dx**2 + dy**2)
            flow_area[y, x] = (head[y, x] - bottom_elevation[y, x]) * dx

    hydraulic_gradient = np.where(hydraulic_gradient <= 0, 0.001, hydraulic_gradient)
    hydraulic_gradient = np.where(hydraulic_gradient > 0.3, 0.3, hydraulic_gradient)
    lateral_grid_cell_area = np.where(mask_bc, (topography - bottom_elevation) * dx, np.nan)
    inflow_m3_s = np.where(mask_bc, kf * hydraulic_gradient * flow_area, np.nan)
    inflow_mm_day = np.where(mask_bc,inflow_m3_s * (86400 / lateral_grid_cell_area) * 1000, 0)

    return inflow_dir, delta_gw_head, hydraulic_gradient, inflow_m3_s, inflow_mm_day   

@click.option("-och", "--offset", type=float, default=1)
@click.option("--plot", type=int, is_flag=True, help="Print more output.")
@click.command("main")
def main(offset, plot):
    # run installed version of flopy or add local path
    try:
        import flopy
    except:
        fpth = os.path.abspath(os.path.join("..", ".."))
        sys.path.append(fpth)
        import flopy

    base_path = Path(__file__).parent
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # load the config file
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        roger_config = yaml.safe_load(file)

    res_modflow = 50  # spatial resolution of MODFLOW in meters
    modflow_config = {
        'dx': int(roger_config['dx']*(res_modflow/roger_config['dx'])),
        'dy': int(roger_config['dy']*(res_modflow/roger_config['dx'])),
        'nx': 621,
        'ny': 777,
        'nz': 4,
    }

    # load RoGeR recharge
    base_path = Path(__file__).parent
    src = rasterio.open(str(base_path / "input" / "recharge_roger_50m.tif"))
    recharge = src.read(1)

    # load interpolated groundwater heads
    base_path = Path(__file__).parent
    src = rasterio.open(str(base_path / "input" / "groundwater_heads_interpolated_50m.tif"))
    gw_heads_interpolated = src.read(1)

    # load MODFLOW parameters
    path = Path(__file__).parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    spatial_ref = ds_params.spatial_ref
    xcoords = ds_params.x.values
    ycoords = ds_params.y.values

    # load topography
    topography = ds_params['elevations'].isel(z=0).values
    mask = np.isfinite(topography) & (ds_params['mask_schoenberg'].values == 0) & (ds_params['mask_black_forest'].values == 0)
    topography[~mask] = np.nan
    mask_schoenberg = (ds_params['mask_schoenberg'].values == 1)
    mask_black_forest = (ds_params['mask_black_forest'].values == 1)
    mask_black_forest[:, -5:] = 0  # only consider black forest in last 5 columns
    grid_extent = (0, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])

    basin = np.zeros((modflow_config['nx'], modflow_config['ny']))
    basin[mask] = 1
    mask_fissured_aquifer = np.zeros((modflow_config['nx'], modflow_config['ny']))
    mask_fissured_aquifer[np.isfinite(ds_params['elevations'].isel(z=0).values) & (ds_params['mask_black_forest'].values == 1)] = 1

    # define location of schoenberg boundary condition
    schoenberg = np.zeros((modflow_config['nx'], modflow_config['ny']))
    schoenberg[mask_schoenberg] = 1
    schoenberg_outer = ndimage.binary_dilation(schoenberg)
    boundary_schoenberg = schoenberg_outer - schoenberg

    mask_boundary_condition_schoenberg = np.where((boundary_schoenberg == 1), 1, 0)
    mask_boundary_condition_schoenberg[mask_black_forest] = 0
    constant_head_schoenberg = np.where((mask_boundary_condition_schoenberg == 1), gw_heads_interpolated, np.nan)
    _topography = np.where(mask_boundary_condition_schoenberg == 1, topography, np.nan)
    constant_depth_schoenberg = _topography - constant_head_schoenberg
    constant_head_schoenberg[constant_depth_schoenberg < 1] = _topography[constant_depth_schoenberg < 1] - 1

    # define location of boundary condition between porous aquifers
    topography = ds_params['elevations'].isel(z=0).values
    mask_topo = np.where(np.isfinite(topography), 1, 0)
    mask_topo_inner = ndimage.binary_erosion(mask_topo)

    boundary = mask_topo - mask_topo_inner
    boundary[-1, :] = 0
    boundary[:, -5:] = 0
    boundary[mask_black_forest] = 0
    boundary[471, 164] = 0 # remove isolated cell

    mask_boundary_condition_porous_aquifer = np.where((boundary == 1), 1, 0)

    constant_head_porous_aquifer = np.where(mask_boundary_condition_porous_aquifer == 1, gw_heads_interpolated, np.nan)
    _topography = np.where(mask_boundary_condition_porous_aquifer == 1, topography, np.nan)
    constant_depth_porous_aquifer = _topography - constant_head_porous_aquifer
    constant_head_porous_aquifer[constant_depth_porous_aquifer < 1] = _topography[constant_depth_porous_aquifer < 1] - 1
    constant_depth_porous_aquifer = _topography - constant_head_porous_aquifer

    # modify constant head between Tuniberg and Rhine
    constant_depth_porous_aquifer[205:220, 44:101] = np.nan
    constant_depth_porous_aquifer[205:220, 44:101] = np.linspace(1, 6.8, constant_head_porous_aquifer[205:220, 44:101].shape[1])[np.newaxis, :]
    constant_head_porous_aquifer[205:220, 44:101] = _topography[205:220, 44:101] - constant_depth_porous_aquifer[205:220, 44:101]
    constant_head_porous_aquifer = np.where(mask_boundary_condition_porous_aquifer == 1, constant_head_porous_aquifer, np.nan)
    constant_depth_porous_aquifer = _topography - constant_head_porous_aquifer
    # modify constant head between Rhine and Staufen
    constant_depth_porous_aquifer[283:462, 6:162] = np.nan
    constant_depth_porous_aquifer[283:462, 6:41] = np.linspace(1, 10, 41-6)[np.newaxis, :]
    constant_depth_porous_aquifer[283:462, 41:86] = np.linspace(10, 7, 86-41)[np.newaxis, :]
    constant_depth_porous_aquifer[283:462, 86:162] = np.linspace(7, 1, 162-86)[np.newaxis, :]
    constant_head_porous_aquifer[283:462, 6:162] = _topography[283:462, 6:162] - constant_depth_porous_aquifer[283:462, 6:162]
    constant_head_porous_aquifer = np.where(mask_boundary_condition_porous_aquifer == 1, constant_head_porous_aquifer, np.nan)
    constant_depth_porous_aquifer = _topography - constant_head_porous_aquifer
    constant_depth_porous_aquifer[constant_depth_porous_aquifer < 0] = _topography[constant_depth_porous_aquifer < 0]
    constant_head_porous_aquifer = _topography - constant_depth_porous_aquifer

    # define location of fissured aquifer boundary condition
    black_forest = np.zeros((modflow_config['nx'], modflow_config['ny']))
    black_forest[mask] = 1
    black_forest_inner = ndimage.binary_erosion(black_forest)
    boundary_black_forest = black_forest - black_forest_inner
    boundary_black_forest[:, -5:] = 0

    mask_boundary_condition_fissured_aquifer = np.where((boundary_black_forest == 1), 1, 0)
    mask_boundary_condition_fissured_aquifer[mask_schoenberg == 1] = 0
    mask_boundary_condition_fissured_aquifer[mask_boundary_condition_porous_aquifer == 1] = 0
    constant_head_fissured_aquifer = np.where(mask_boundary_condition_fissured_aquifer == 1, gw_heads_interpolated, np.nan)
    _topography = np.where(mask_boundary_condition_fissured_aquifer == 1, topography, np.nan)
    constant_depth_fissured_aquifer = _topography - constant_head_fissured_aquifer
    constant_depth_fissured_aquifer[constant_depth_fissured_aquifer < 0] = _topography[constant_depth_fissured_aquifer < 0]
    constant_head_fissured_aquifer = _topography - constant_depth_fissured_aquifer

    mask_boundary_condition_fissured_aquifer = np.where((boundary_black_forest == 1), 1, 0)
    mask_boundary_condition_fissured_aquifer[mask_schoenberg == 1] = 0
    mask_boundary_condition_fissured_aquifer[mask_boundary_condition_porous_aquifer == 1] = 0
    constant_head_fissured_aquifer = np.where(mask_boundary_condition_fissured_aquifer == 1, gw_heads_interpolated, np.nan)
    _topography = np.where(mask_boundary_condition_fissured_aquifer == 1, topography, np.nan)
    constant_depth_fissured_aquifer = _topography - constant_head_fissured_aquifer
    constant_depth_fissured_aquifer[constant_depth_fissured_aquifer < 0] = _topography[constant_depth_fissured_aquifer < 0]
    constant_head_fissured_aquifer = _topography - constant_depth_fissured_aquifer

    # get flow direction
    file = base_path / "input" / "groundwater_heads_interpolated_50m.tif"
    file_str = str(file)
    grid = Grid.from_raster(file_str)
    gw_heads = grid.read_raster(file_str)
    # Fill pits in groundwater heads
    pit_filled_gw_heads = grid.fill_pits(gw_heads)
    # Fill depressions in groundwater heads
    flooded_gw_heads = grid.fill_depressions(pit_filled_gw_heads)
    # Resolve flats in groundwater heads
    inflated_gw_heads = grid.resolve_flats(flooded_gw_heads)
    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    # Compute flow directions
    # -------------------------------------
    fdir = grid.flowdir(gw_heads, dirmap=dirmap)
    mfdir = grid.flowdir(gw_heads, routing='mfd')
    inffdir = grid.flowdir(gw_heads, routing='dinf')
    outflow_dir = np.array(mfdir, dtype=np.float32)

    # calculate lateral inflow of fissured aquifer boundary condition
    mask_bc = (mask_boundary_condition_fissured_aquifer == 1)
    kf = 1.97 * 10e-7  # m/s
    ll_output = lateral_inflow_bc_mfdir(mask_bc, outflow_dir, gw_heads_interpolated, topography, ds_params['elevations'].isel(z=4).values, kf, dx=modflow_config['dx'], dy=modflow_config['dy'])
    inflow_dir = ll_output[0]
    delta_gw_head = ll_output[1]
    hydraulic_gradient = ll_output[2]
    inflow_m3_s = ll_output[3]
    inflow_mm_day = ll_output[4]

    # # calculate lateral inflow of fissured aquifer boundary condition
    # mask_bc = (mask_boundary_condition_fissured_aquifer == 1)
    # ll_output = lateral_inflow_bc_d8(mask_bc, outflow_dir, gw_heads_interpolated, topography, ds_params['elevations'].isel(z=4).values, 10e-7, dx=modflow_config['dx'], dy=modflow_config['dy'])
    # inflow_dir = ll_output[0]
    # delta_gw_head = ll_output[1]
    # hydraulic_gradient = ll_output[2]
    # inflow_m3_s = ll_output[3]
    # inflow_mm_day = ll_output[4]

    # write boundary condtions to netcdf
    ds = xr.Dataset()
    ds["spatial_ref"] = spatial_ref
    file = base_path / "input" / "boundary_conditions.nc"
    ds.to_netcdf(file, engine="h5netcdf")
    ds.close()

    file = base_path / "input" / "boundary_conditions.nc"
    with h5netcdf.File(file, "a", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="Boundary conditions for the porous aquifer of the Dreisam-Möhlin-Neumagen catchment",
        institution="University of Freiburg, Chair of Hydrology",
        references="",
        comment="",
        spatial_ref="EPSG:25832",
        x_origin=396331.5,
        y_origin=5325918.5,
        )
        dict_dim = {"y": len(ds_params['y'].values), "x": len(ds_params['x'].values), 'scalar': 1}
        f.dimensions = dict_dim
        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "X-coordinate"
        v.attrs["units"] = "m"
        v[:] = xcoords
        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "Y-coordinate"
        v.attrs["units"] = "m"
        v[:] = ycoords
        v = f.create_variable('cell_width', ('scalar',), float)
        v.attrs['long_name'] = 'Cell width'
        v.attrs['units'] = 'm'
        v[:] = modflow_config['dx']

        v = f.create_variable(
            "mask_porous_aquifer_bc", ("y", "x"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = mask_boundary_condition_porous_aquifer[:, :]
        v.attrs.update(long_name="location of constant head boundary condition of the porous aquifer", units=" ", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "constant_head_porous_aquifer", ("y", "x"), np.float32, compression="gzip", compression_opts=1
        )
        v[:, :] = constant_head_porous_aquifer[:, :] - offset
        v.attrs.update(long_name="constant head boundary condition of the porous aquifer", units="m a.s.l.", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "constant_depth_porous_aquifer", ("y", "x"), np.float32, compression="gzip", compression_opts=1
        )
        v[:, :] = constant_depth_porous_aquifer[:, :]
        v.attrs.update(long_name="depth of constant head boundary condition of the porous aquifer", units="m", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "mask_schoenberg_bc", ("y", "x"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = mask_boundary_condition_schoenberg[:, :]
        v.attrs.update(long_name="location of schoenberg boundary condition", units=" ", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "mask_porous_aquifer", ("y", "x"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = basin
        v.attrs.update(long_name="location of porous aquifer", units=" ", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "mask_fissured_aquifer", ("y", "x"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = mask_fissured_aquifer
        v.attrs.update(long_name="location of fissured aquifer", units=" ", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "constant_head_schoenberg", ("y", "x"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = constant_head_schoenberg[:, :] - offset
        v.attrs.update(long_name="schoenberg boundary condition", units="m a.s.l.", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "recharge", ("y", "x"), np.float32, compression="gzip", compression_opts=1
        )
        # calculate average recharge from total recharge of the period 2013-2022
        dates = pd.date_range(start="2013-01-01", end="2022-12-31", freq="D")
        recharge = np.where(mask, recharge, np.nan) / len(dates)
        # set recharge to zero in the area of Schoenberg
        src = rasterio.open(str(base_path / "input" / "schoenberg.tif"))
        mask_schoenberg = src.read(1)
        recharge = np.where(mask_schoenberg == 1, np.nan, recharge)
        # set recharge to zero in the area of Black Forest
        # recharge = np.where(mask_boundary_condition_fissured_aquifer == 1, np.nan, recharge)
        # set recharge to zero for NaN values
        recharge = np.where(np.isnan(recharge), 0, recharge)
        recharge = np.where(recharge < 0, 0, recharge)
        v[:, :] = recharge
        v.attrs.update(long_name="recharge boundary condition", units="mm/day", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "mask_fissured_aquifer_bc", ("y", "x"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = mask_boundary_condition_fissured_aquifer[:, :]
        v.attrs.update(long_name="location of constant head boundary condition of the fissured aquifer", units="-", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "constant_head_fissured_aquifer", ("y", "x"), np.float32, compression="gzip", compression_opts=1
        )
        v[:, :] = constant_head_fissured_aquifer[:, :] - offset
        v.attrs.update(long_name="constant head boundary condition of the fissured aquifer", units="m a.s.l.", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "i_lateral_inflow_bc", ("y", "x"), np.float32, compression="gzip", compression_opts=1
        )
        v[:, :] = hydraulic_gradient[:, :]
        v.attrs.update(long_name="hydraulic gradient of lateral inflow boundary condition", units="-", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "lateral_inflow_bc_mmday", ("y", "x"), np.float32, compression="gzip", compression_opts=1
        )
        v[:, :] = inflow_mm_day[:, :]
        v.attrs.update(long_name="lateral inflow boundary condition", units="mm/day", grid_mapping="spatial_ref", coordinates="spatial_ref")

        v = f.create_variable(
            "lateral_inflow_bc_m3s", ("y", "x"), np.float32, compression="gzip", compression_opts=1
        )
        v[:, :] = inflow_m3_s[:, :]
        v.attrs.update(long_name="lateral inflow boundary condition", units="m3/s", grid_mapping="spatial_ref", coordinates="spatial_ref")

    # fig, axes = plt.subplots(figsize=(4, 4))
    # plt.imshow(boundary, extent=grid_extent, cmap='Greys', aspect='equal')
    # plt.grid(zorder=0)
    # plt.xlabel('Distance in x-direction [m]')
    # plt.ylabel('Distance in y-direction [m]')
    # plt.tight_layout()
    # file = base_path_figs / "boundary.png"
    # fig.savefig(file, dpi=300)
    # plt.close(fig)

    grid_extent = (ds_params.x.values[0] / 1000, ds_params.x.values[-1] / 1000, ds_params.y.values[0] / 1000, ds_params.y.values[-1] / 1000)
    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(np.where((boundary == 1))[1]*50, np.where((boundary == 1))[0]*50, s=0.5, c='k', alpha=0.5)
    # axes.scatter(np.where((mask_boundary_condition_porous_aquifer == 1))[1]*50, np.where((mask_boundary_condition_porous_aquifer == 1))[0]*50, s=0.5, c='grey')
    # axes.scatter(np.where((mask_boundary_condition_fissured_aquifer == 1))[1]*50, np.where((mask_boundary_condition_fissured_aquifer == 1))[0]*50, s=0.5, c='red')
    # axes.scatter(np.where((mask_boundary_condition_schoenberg == 1))[1]*50, np.where((mask_boundary_condition_schoenberg == 1))[0]*50, s=0.5, c='grey')
    # axes.scatter(np.where((mask_boundary_condition1 == 1))[1]*.05, np.where((mask_boundary_condition1 == 1))[0]*.05, s=0.5, c='purple')
    # topography[mask_schoenberg] = np.nan
    # topography[mask_black_forest] = np.nan
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.colorbar(label='[m a.s.l.]', shrink=0.5, alpha=1)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.tight_layout()
    file = base_path_figs / "mask_boundary_condition_porous_aquifer.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    # grid_extent = (0, 777*modflow_config['dy'], 621*modflow_config['dx'], 0)
    # fig, axes = plt.subplots(figsize=(4, 4))
    # plt.imshow(recharge / 1000, cmap='Blues', aspect='equal', extent=grid_extent)
    # plt.colorbar(label='[m/day]', shrink=0.5)
    # plt.grid(zorder=0)
    # plt.xlabel('distance in x-direction [m]')
    # plt.ylabel('distance in y-direction [m]')
    # plt.tight_layout()
    # file = base_path_figs / "average_recharge.png"
    # fig.savefig(file, dpi=300)
    # plt.close(fig)

    # fig, axes = plt.subplots(figsize=(6, 3))
    # axes.plot(range(len(_constant_headx)), _constant_headx, label="constant head", color="black", linestyle="-")
    # axes.plot(range(len(_topographyx)), _topographyx, label="topography", color="grey", linestyle="-")
    # axes.set_ylabel("elevation [m.a.s.l.]")
    # fig.tight_layout()
    # file = Path(__file__).parent / "figures" / "constant_head_xx.png"
    # fig.savefig(file, dpi=300)
    # plt.close("all")

    # fig, axes = plt.subplots(figsize=(6, 3))
    # axes.plot(range(len(_constant_heady)), _constant_heady, label="constant head", color="black", linestyle="-")
    # axes.plot(range(len(_topographyy)), _topographyy, label="topography", color="grey", linestyle="-")
    # axes.set_ylabel("elevation [m.a.s.l.]")
    # fig.tight_layout()
    # file = Path(__file__).parent / "figures" / "constant_head_yy.png"
    # fig.savefig(file, dpi=300)
    # plt.close("all")
    return


if __name__ == "__main__":
    main()


