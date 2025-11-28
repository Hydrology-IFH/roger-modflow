import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5netcdf
import xarray as xr
import rasterio
import datetime
import yaml
import click

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

    # load topography
    topography = ds_params['elevations'].isel(z=0).values
    mask = np.isfinite(topography) & (ds_params['mask_schoenberg'].values == 0)
    topography[~mask] = np.nan
    mask_schoenberg = (ds_params['mask_schoenberg'].values == 1)
    mask_black_forest = (ds_params['mask_black_forest'].values == 1)
    grid_extent = (0, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])

    basin = np.zeros((modflow_config['nx'], modflow_config['ny']))
    basin[mask] = 1

    # define location of schoenberg boundary condition
    schoenberg = np.zeros((modflow_config['nx'], modflow_config['ny']))
    schoenberg[mask_schoenberg] = 1
    inner_boundary_schoenberg = np.zeros((modflow_config['nx']+4, modflow_config['ny']+4))
    inner_boundary_schoenberg[2:-2, 2:-2] = schoenberg
    boundary_schoenberg = np.zeros((modflow_config['nx']+4, modflow_config['ny']+4))
    boundary_schoenberg[1:-3, 2:-2] = np.where(inner_boundary_schoenberg[2:-2, 2:-2] - inner_boundary_schoenberg[1:-3, 2:-2] < 0, 1, boundary_schoenberg[1:-3, 2:-2])
    boundary_schoenberg[3:-1, 2:-2] = np.where(inner_boundary_schoenberg[2:-2, 2:-2] - inner_boundary_schoenberg[3:-1, 2:-2] < 0, 1, boundary_schoenberg[3:-1, 2:-2])
    boundary_schoenberg[2:-2, 1:-3] = np.where(inner_boundary_schoenberg[2:-2, 2:-2] - inner_boundary_schoenberg[2:-2, 1:-3] < 0, 1, boundary_schoenberg[2:-2, 1:-3])
    boundary_schoenberg[2:-2, 3:-1] = np.where(inner_boundary_schoenberg[2:-2, 2:-2] - inner_boundary_schoenberg[2:-2, 3:-1] < 0, 1, boundary_schoenberg[2:-2, 3:-1])
    boundary_schoenberg[1:-3, 1:-3] = np.where(inner_boundary_schoenberg[2:-2, 2:-2] - inner_boundary_schoenberg[1:-3, 1:-3] < 0, 1, boundary_schoenberg[1:-3, 1:-3])
    boundary_schoenberg[1:-3, 3:-1] = np.where(inner_boundary_schoenberg[2:-2, 2:-2] - inner_boundary_schoenberg[1:-3, 3:-1] < 0, 1, boundary_schoenberg[1:-3, 3:-1])
    boundary_schoenberg[3:-1, 1:-3] = np.where(inner_boundary_schoenberg[2:-2, 2:-2] - inner_boundary_schoenberg[3:-1, 1:-3] < 0, 1, boundary_schoenberg[3:-1, 1:-3])
    boundary_schoenberg[1:-3, 3:-1] = np.where(inner_boundary_schoenberg[2:-2, 2:-2] - inner_boundary_schoenberg[1:-3, 3:-1] < 0, 1, boundary_schoenberg[1:-3, 3:-1])
    boundary_schoenberg = boundary_schoenberg[2:-2, 2:-2]

    mask_boundary_condition_schoenberg = np.where((boundary_schoenberg == 1), 1, np.nan)
    mask_boundary_condition_schoenberg[mask_black_forest] = np.nan
    constant_head_schoenberg = np.where((mask_boundary_condition_schoenberg == 1), gw_heads_interpolated, np.nan)
    _topography = np.where(mask_boundary_condition_schoenberg == 1, topography, np.nan)
    constant_head_depth_schoenberg = _topography - constant_head_schoenberg
    constant_head_schoenberg[constant_head_depth_schoenberg < 1] = _topography[constant_head_depth_schoenberg < 1] - 1

    # define location of boundary condition between porous aquifers
    inner_boundary = np.zeros((modflow_config['nx']+4, modflow_config['ny']+4))
    topography = ds_params['elevations'].isel(z=0).values
    inner_boundary[2:-2, 2:-2] = np.isfinite(topography)
    boundary = np.zeros((modflow_config['nx']+4, modflow_config['ny']+4))

    boundary[1:-3, 2:-2] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[1:-3, 2:-2] < 0, 1, boundary[1:-3, 2:-2])
    boundary[3:-1, 2:-2] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[3:-1, 2:-2] < 0, 1, boundary[3:-1, 2:-2])
    boundary[2:-2, 1:-3] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[2:-2, 1:-3] < 0, 1, boundary[2:-2, 1:-3])
    boundary[2:-2, 3:-1] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[2:-2, 3:-1] < 0, 1, boundary[2:-2, 3:-1])
    boundary[1:-3, 1:-3] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[1:-3, 1:-3] < 0, 1, boundary[1:-3, 1:-3])
    boundary[1:-3, 3:-1] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[1:-3, 3:-1] < 0, 1, boundary[1:-3, 3:-1])
    boundary[3:-1, 1:-3] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[3:-1, 1:-3] < 0, 1, boundary[3:-1, 1:-3])
    boundary[1:-3, 3:-1] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[1:-3, 3:-1] < 0, 1, boundary[1:-3, 3:-1])

    boundary = boundary[2:-2, 2:-2]
    boundary[-1, :] = 0
    boundary[:, -5:] = 0
    boundary[mask_black_forest == 1] = 0

    mask_boundary_condition_porous_aquifer = np.where((boundary == 1), 1, np.nan)

    constant_head_porous_aquifer = np.where(mask_boundary_condition_porous_aquifer == 1, gw_heads_interpolated, np.nan)
    _topography = np.where(mask_boundary_condition_porous_aquifer == 1, topography, np.nan)
    constant_head_porous_aquifer_depth = _topography - constant_head_porous_aquifer
    constant_head_porous_aquifer[constant_head_porous_aquifer_depth < 1] = _topography[constant_head_porous_aquifer_depth < 1] - 1
    constant_head_porous_aquifer_depth = _topography - constant_head_porous_aquifer

    # modify constant head between Tuniberg and Rhine
    constant_head_porous_aquifer_depth[205:220, 44:101] = np.nan
    constant_head_porous_aquifer_depth[205:220, 44:101] = np.linspace(1, 6.8, constant_head_porous_aquifer[205:220, 44:101].shape[1])[np.newaxis, :]
    constant_head_porous_aquifer[205:220, 44:101] = _topography[205:220, 44:101] - constant_head_porous_aquifer_depth[205:220, 44:101]
    constant_head_porous_aquifer = np.where(mask_boundary_condition_porous_aquifer == 1, constant_head_porous_aquifer, np.nan)
    constant_head_porous_aquifer_depth = _topography - constant_head_porous_aquifer

    # modify constant head between Rhine and Staufen
    constant_head_porous_aquifer_depth[283:462, 6:162] = np.nan
    constant_head_porous_aquifer_depth[283:462, 6:41] = np.linspace(1, 10, 41-6)[np.newaxis, :]
    constant_head_porous_aquifer_depth[283:462, 41:86] = np.linspace(10, 7, 86-41)[np.newaxis, :]
    constant_head_porous_aquifer_depth[283:462, 86:162] = np.linspace(7, 1, 162-86)[np.newaxis, :]
    constant_head_porous_aquifer[283:462, 6:162] = _topography[283:462, 6:162] - constant_head_porous_aquifer_depth[283:462, 6:162]
    constant_head_porous_aquifer = np.where(mask_boundary_condition_porous_aquifer == 1, constant_head_porous_aquifer, np.nan)
    constant_head_porous_aquifer_depth = _topography - constant_head_porous_aquifer
    constant_head_porous_aquifer[constant_head_porous_aquifer_depth < 1] = _topography[constant_head_porous_aquifer_depth < 1] - 1
    constant_head_porous_aquifer_depth = _topography - constant_head_porous_aquifer

    # write boundary condtions to netcdf
    params_file = base_path / "input" / "boundary_conditions.nc"
    with h5netcdf.File(params_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="Boundary conditions for the porous aquifer of the Dreisam-Möhlin-Neumagen catchment",
        institution="University of Freiburg, Chair of Hydrology",
        references="",
        comment="",
        )
        dict_dim = {"lat": len(ds_params['y'].values), "lon": len(ds_params['x'].values), 'scalar': 1}
        f.dimensions = dict_dim
        v = f.create_variable("lon", ("lon",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "X-coordinate"
        v.attrs["units"] = "m"
        v[:] = ds_params['x'].values
        v = f.create_variable("lat", ("lat",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "Y-coordinate"
        v.attrs["units"] = "m"
        v[:] = ds_params['y'].values
        v = f.create_variable('cell_width', ('scalar',), float)
        v.attrs['long_name'] = 'Cell width'
        v.attrs['units'] = 'm'
        v[:] = modflow_config['dx']

        v = f.create_variable(
            "mask_porous_aquifer_bc", ("lat", "lon"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = mask_boundary_condition_porous_aquifer[:, :]
        v.attrs.update(long_name="location of constant head boundary condition of the porous aquifer", units="-")

        v = f.create_variable(
            "constant_head_porous_aquifer", ("lat", "lon"), np.float32, compression="gzip", compression_opts=1
        )
        v[:, :] = constant_head_porous_aquifer[:, :] - offset
        v.attrs.update(long_name="constant head boundary condition of the porous aquifer", units="m a.s.l.")

        v = f.create_variable(
            "constant_head_depth_porous_aquifer", ("lat", "lon"), np.float32, compression="gzip", compression_opts=1
        )
        v[:, :] = constant_head_porous_aquifer_depth[:, :]
        v.attrs.update(long_name="depth of constant head boundary condition of the porous aquifer", units="m")

        v = f.create_variable(
            "mask_schoenberg_bc", ("lat", "lon"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = mask_boundary_condition_schoenberg[:, :]
        v.attrs.update(long_name="location of schoenberg boundary condition", units="-")

        v = f.create_variable(
            "constant_head_schoenberg", ("lat", "lon"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = constant_head_schoenberg[:, :] - offset
        v.attrs.update(long_name="schoenberg boundary condition", units="m a.s.l.")

        v = f.create_variable(
            "recharge", ("lat", "lon"), np.float32, compression="gzip", compression_opts=1
        )

        # calculate average recharge from total recharge of the period 2013-2022
        dates = pd.date_range(start="2013-01-01", end="2022-12-31", freq="D")
        recharge = np.where(mask, recharge, np.nan) / len(dates)
        # set recharge to zero in the area of Schoenberg
        src = rasterio.open(str(base_path / "input" / "schoenberg.tif"))
        mask_schoenberg = src.read(1)
        recharge = np.where(mask_schoenberg == 1, np.nan, recharge)
        # set recharge to zero for NaN values
        recharge = np.where(np.isnan(recharge), 0, recharge)
        recharge = np.where(recharge < 0, 0, recharge)
        v[:, :] = recharge[:, :]
        v.attrs.update(long_name="recharge boundary condition", units="mm/day")

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


