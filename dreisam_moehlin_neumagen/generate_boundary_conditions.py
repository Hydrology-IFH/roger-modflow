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

@click.option("-och", "--offset", type=float, default=10)
@click.option("-mr", "--model-run", type=int, default=0)
@click.option("--plot", type=int, is_flag=True, help="Print more output.")
@click.command("main")
def main(offset, model_run, plot):
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

    # set offset for constant head boundary condition
    file = base_path / "fudge_parameters_modflow.csv"
    if os.path.exists(file):
        fudge_parameters = pd.read_csv(file, sep=";", skiprows=1)
        offset = fudge_parameters.loc[model_run, 'offset_constant_head']
    else:
        offset = offset

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
    path = Path(__file__).parent / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")

    # load topography
    topography = ds_params['elevations'].isel(z=0).values
    mask = np.isfinite(topography)
    grid_extent = (0, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])

    basin = np.zeros((modflow_config['nx'], modflow_config['ny']))
    basin[mask] = 1

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
    mask_boundary_condition = np.where((boundary == 1) & (topography <= 240), 1, np.nan)
    mask_boundary_condition[80:250, :180] = np.where((boundary == 1)[80:250, :180], 1, mask_boundary_condition[80:250, :180])

    # # set constant head
    # xx = np.where(mask_boundary_condition == 1)[0]
    # yy = np.where(mask_boundary_condition == 1)[1]
    # headsx = np.linspace(188, topography[xx[-1], yy[-1]] - 30, np.max(xx)+1) - 34
    # headsy = np.linspace(162, 185, np.max(yy)+1) - 27
    # boundary_condition = np.empty_like(mask_boundary_condition)
    # for x, y in zip(xx, yy):
    #     if y < 250:
    #         boundary_condition[x, y] = headsx[x]
    #     else:
    #         boundary_condition[x, y] = headsy[y]

    # set constant head to interpolated groundwater heads
    boundary_condition = np.where(mask_boundary_condition, gw_heads_interpolated, np.nan)

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
            "mask_constant_head", ("x", "y"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = mask_boundary_condition[:, :]
        v.attrs.update(long_name="location of constant head boundary condition", units="-")

        v = f.create_variable(
            "constant_head", ("x", "y"), np.int32, compression="gzip", compression_opts=1
        )
        v[:, :] = boundary_condition[:, :] - offset
        v.attrs.update(long_name="constant head boundary condition", units="m a.s.l.")

        v = f.create_variable(
            "recharge", ("x", "y"), np.float32, compression="gzip", compression_opts=1
        )
        # calculate average recharge from total recharge of the period 2013-2022
        dates = pd.date_range(start="2013-01-01", end="2022-12-31", freq="D")
        recharge = np.where(mask, recharge, np.nan) / len(dates)
        # set recharge to zero in the area of Schoenberg
        src = rasterio.open(str(base_path / "input" / "schoenberg.tif"))
        schoenberg_mask = src.read(1)
        recharge = np.where(schoenberg_mask == 1, np.nan, recharge)
        # set recharge to zero for NaN values
        recharge = np.where(np.isnan(recharge), 0, recharge)
        recharge = np.where(recharge < 0, 0, recharge)
        v[:, :] = recharge[:, :]
        v.attrs.update(long_name="recharge boundary condition", units="mm/day")

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(boundary, extent=grid_extent, cmap='Greys', aspect='equal')
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / "boundary.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(np.where((boundary == 1))[1]*.05, np.where((boundary == 1))[0]*.05, s=0.5, c='k', alpha=0.5)
    axes.scatter(np.where((mask_boundary_condition == 1))[1]*.05, np.where((mask_boundary_condition == 1))[0]*.05, s=0.5, c='r')
    # axes.scatter(np.where((mask_boundary_condition1 == 1))[1]*.05, np.where((mask_boundary_condition1 == 1))[0]*.05, s=0.5, c='purple')
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=(0, (modflow_config['ny']*modflow_config['dy'])/1000, (modflow_config['nx']*modflow_config['dx'])/1000, 0))
    plt.colorbar(label='[m a.s.l.]', shrink=0.5, alpha=1)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    plt.tight_layout()
    file = base_path_figs / "mask_boundary_condition.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    grid_extent = (0, 777*modflow_config['dy'], 0, 621*modflow_config['dx'])
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(recharge / 1000, cmap='Blues', aspect='equal', extent=grid_extent)
    plt.colorbar(label='[m/day]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('distance in x-direction [m]')
    plt.ylabel('distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / "average_recharge.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    return


if __name__ == "__main__":
    main()


