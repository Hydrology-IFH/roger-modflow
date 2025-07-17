import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import h5netcdf
import click

@click.command("main", short_help="Define drainage areas from MODFLOW output")
def main():
    # run installed version of flopy or add local path
    try:
        import flopy
    except:
        fpth = os.path.abspath(os.path.join("..", ".."))
        sys.path.append(fpth)
        import flopy

    base_path = Path(__file__).parent

    res_modflow = 50  # spatial resolution of MODFLOW in meters

    modflow_config = {
        'dx': res_modflow,
        'dy': res_modflow,
        'nx': 621,
        'ny': 777,
        'nz': 4,
    }
    grid_extent = (0, 777*modflow_config['dy'], 621*modflow_config['dx'], 0)

    # load MODFLOW parameters
    path = Path(__file__).parent / "parameters_modflow.nc"
    with xr.open_dataset(path, engine="h5netcdf") as ds_params:
        topography = ds_params['elevations'].isel(z=0).values
        mask_schoenberg = ds_params['mask_schoenberg'].values

    mask_ = np.isfinite(topography)
    mask = mask_ & (mask_schoenberg == False)
    domain = np.empty_like(topography)
    domain[mask] = 1
    domain[~mask] = -1

    # load the netcdf file
    output_file = base_path / "input" / "modflow_output_run_for_drainage_areas.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")

    # plot the saturation depth
    saturation_depth = ds_mf['head'].isel(Time=0, layer=1).values - topography
    saturation_depth[saturation_depth < 0.3] = np.nan
    saturation_depth[~mask] = np.nan
    saturation_depth[:, 400:] = np.nan
    saturation_depth[500:, :] = np.nan
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(saturation_depth, extent=grid_extent, cmap='viridis_r', aspect='equal')
    plt.colorbar(label='groundwater above surface [m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / "figures" / "saturation_depth.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    # add drainage mask to parameters_modflow.nc
    mask_saturation = np.where(saturation_depth > 0.3, 1, 0)
    path = str(base_path / "input" / "parameters_modflow.nc")
    with h5netcdf.File(path, "a", decode_vlen_strings=False) as f:
        try:
            v = f.create_variable("mask_drainage", ("y", "x"), int, compression="gzip", compression_opts=1)
            v[:, :] = mask_saturation
            v.attrs.update(long_name="Mask of drainage areas", units="")
        except ValueError:
            var_obj = f.variables.get("mask_drainage")
            var_obj[:, :] = mask_saturation
    return


if __name__ == "__main__":
    main()