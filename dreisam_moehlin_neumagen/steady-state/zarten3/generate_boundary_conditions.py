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
    xlim1 = 380
    xlim2 = 500
    ylim1 = 188
    ylim2 = 250

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

    # load observed groundwater heads (average values of the observation wells)
    path = base_path / "observations" / "observed_groundwater_heads_avg.csv"
    observed_groundwater_heads = pd.read_csv(path, sep=";", skiprows=0)

    # load rivers
    base_path = Path(__file__).parent
    src = rasterio.open(str(base_path.parent / "input" / "mask_rivers_50m.tif"))
    rivers = src.read(1)

    # load RoGeR recharge
    base_path = Path(__file__).parent
    src = rasterio.open(str(base_path.parent / "input" / "recharge_roger_50m.tif"))
    recharge = src.read(1)

    # load interpolated groundwater heads
    base_path = Path(__file__).parent
    src = rasterio.open(str(base_path.parent / "input" / "groundwater_heads_interpolated_50m.tif"))
    gw_heads_interpolated = src.read(1)

    # load MODFLOW parameters
    path = Path(__file__).parent / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")

    # load topography
    topography = ds_params['elevations'].isel(z=0).values
    mask = np.isfinite(topography)
    mask[:, :] = False
    mask[ylim1:ylim2, xlim1:xlim2] = True
    mask[ylim2-20:ylim2, xlim1:xlim1+20] = False
    hydraulic_conductivity = ds_params['kf'].isel(layer=2).values
    mask[hydraulic_conductivity < 1] = False
    mask[topography > 360] = False
    topography[~mask] = np.nan
    grid_extent = (0, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])

    basin = np.zeros((modflow_config['nx'], modflow_config['ny']))
    basin[mask] = 1

    rivers[~mask] = np.nan

    inner_boundary = np.zeros((modflow_config['nx']+4, modflow_config['ny']+4))
    inner_boundary[2:-2, 2:-2] = basin
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
    topography[~mask] = np.nan
    # define location of boundary condition
    mask_boundary_condition_porous_aquifer = np.where((boundary == 1), 1, 0)
    # mask_boundary_condition_porous_aquifer[:, xlim1+1:xlim2-1] = 0

    constant_head_porous_aquifer = np.where(mask_boundary_condition_porous_aquifer == 1, gw_heads_interpolated, np.nan)
    _topography = np.where(mask_boundary_condition_porous_aquifer == 1, topography, np.nan)
    constant_head_porous_aquifer_depth = _topography - constant_head_porous_aquifer
    constant_head_porous_aquifer[constant_head_porous_aquifer_depth < 1] = _topography[constant_head_porous_aquifer_depth < 1] - 1

    # write boundary condtions to netcdf
    params_file = base_path / "boundary_conditions.nc"
    with h5netcdf.File(params_file, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="Boundary conditions for the Dreisam-Möhlin-Neumagen catchment",
        institution="University of Freiburg, Chair of Hydrology",
        references="",
        comment="",
        )
        dict_dim = {"x": modflow_config['nx'], "y": modflow_config['ny'], 'scalar': 1}
        f.dimensions = dict_dim
        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "x"
        v.attrs["units"] = "m"
        v[:] = np.arange(dict_dim["x"]) * modflow_config['dx']
        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "y"
        v.attrs["units"] = "m"
        v[:] = np.arange(dict_dim["y"]) * modflow_config['dy']
        v = f.create_variable('cell_width', ('scalar',), float)
        v.attrs['long_name'] = 'Cell width'
        v.attrs['units'] = 'm'
        v[:] = modflow_config['dx']
        v = f.create_variable('x_origin', ('scalar',), float)
        v.attrs['long_name'] = 'Origin of x-direction'
        v.attrs['units'] = 'm'
        v[:] = 0
        v = f.create_variable('y_origin', ('scalar',), float)
        v.attrs['long_name'] = 'Origin of y-direction'
        v.attrs['units'] = 'm'
        v[:] = 0

        v = f.create_variable(
            "mask_rivers", ("x", "y"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = mask_boundary_condition_porous_aquifer[:, :]
        v.attrs.update(long_name="location of rivers", units="-")

        v = f.create_variable(
            "mask_porous_aquifer_bc", ("x", "y"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = mask_boundary_condition_porous_aquifer[:, :]
        v.attrs.update(long_name="location of constant head boundary condition of the porous aquifer", units="-")

        v = f.create_variable(
            "constant_head_porous_aquifer", ("x", "y"), np.float32, compression="gzip", compression_opts=1
        )
        v[:, :] = constant_head_porous_aquifer[:, :] - offset
        v.attrs.update(long_name="constant head boundary condition of the porous aquifer", units="m a.s.l.")

        v = f.create_variable(
            "recharge", ("x", "y"), np.float32, compression="gzip", compression_opts=1
        )
        # calculate average recharge from total recharge of the period 2013-2022
        dates = pd.date_range(start="2013-01-01", end="2022-12-31", freq="D")
        recharge = np.where(mask, recharge, np.nan) / len(dates)
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

    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(np.where((boundary == 1))[1]*50, np.where((boundary == 1))[0]*50, s=0.5, c='k', alpha=0.5)
    axes.scatter(np.where((mask_boundary_condition_porous_aquifer == 1))[1]*50, np.where((mask_boundary_condition_porous_aquifer == 1))[0]*50, s=0.5, c='grey')
    # axes.scatter(np.where((mask_porous_aquifer_bc1 == 1))[1]*.05, np.where((mask_porous_aquifer_bc1 == 1))[0]*.05, s=0.5, c='purple')
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=(0, (modflow_config['ny']*modflow_config['dy']), (modflow_config['nx']*modflow_config['dx']), 0))
    plt.colorbar(label='[m a.s.l.]', shrink=0.5, alpha=1)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / "mask_porous_aquifer_bc.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    topography[(rivers == 1)] = np.nan
    plt.imshow(topography[ylim1:ylim2, xlim1:xlim2], cmap='terrain', aspect='equal', alpha=1)
    plt.colorbar(label='[m a.s.l.]', shrink=0.5, alpha=1)
    plt.grid(zorder=0)
    plt.xlabel('x-direction')
    plt.ylabel('y-direction')
    plt.tight_layout()
    file = base_path_figs / "topography_clipped.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]) * 50
    wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]) * 50
    plt.scatter(wells_x, wells_y, marker='^', s=5, c='black')
    wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
    wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
    plt.scatter(wells_obs_x, wells_obs_y, marker='.', s=5, c='purple')
    axes.scatter(np.where((boundary == 1))[1]*50, np.where((boundary == 1))[0]*50, s=0.5, c='k', alpha=0.5)
    axes.scatter(np.where((mask_boundary_condition_porous_aquifer == 1))[1]*50, np.where((mask_boundary_condition_porous_aquifer == 1))[0]*50, s=0.5, c='grey')
    # axes.scatter(np.where((mask_porous_aquifer_bc1 == 1))[1]*.05, np.where((mask_porous_aquifer_bc1 == 1))[0]*.05, s=0.5, c='purple')
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=(0, (modflow_config['ny']*modflow_config['dy']), (modflow_config['nx']*modflow_config['dx']), 0))
    plt.colorbar(label='[m a.s.l.]', shrink=0.5, alpha=1)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / "mask_porous_aquifer_bc_.png"
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
    # axes.plot(range(len(_constant_head_porous_aquiferx)), _constant_head_porous_aquiferx, label="constant head", color="black", linestyle="-")
    # axes.plot(range(len(_topographyx)), _topographyx, label="topography", color="grey", linestyle="-")
    # axes.set_ylabel("elevation [m.a.s.l.]")
    # fig.tight_layout()
    # file = Path(__file__).parent / "figures" / "constant_head_porous_aquifer_xx.png"
    # fig.savefig(file, dpi=300)
    # plt.close("all")

    # fig, axes = plt.subplots(figsize=(6, 3))
    # axes.plot(range(len(_constant_head_porous_aquifery)), _constant_head_porous_aquifery, label="constant head", color="black", linestyle="-")
    # axes.plot(range(len(_topographyy)), _topographyy, label="topography", color="grey", linestyle="-")
    # axes.set_ylabel("elevation [m.a.s.l.]")
    # fig.tight_layout()
    # file = Path(__file__).parent / "figures" / "constant_head_porous_aquifer_yy.png"
    # fig.savefig(file, dpi=300)
    # plt.close("all")
    return


if __name__ == "__main__":
    main()


