from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import rasterio
import click
import contextily as ctx
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar

@click.command("main", short_help="Run MODFLOW in steady-state mode")
def main():

    base_path = Path(__file__).parent

    # load groundwater extraction data
    path = base_path.parent / "input" / "groundwater_extraction_bn_cerdia.csv"
    groundwater_extraction = pd.read_csv(path, sep=";", skiprows=0)
    wells_y = groundwater_extraction["y-coordinate"].values
    wells_x = groundwater_extraction["x-coordinate"].values

    # load the netcdf file
    output_file = base_path / "output" / "wsg_hausen.nc"
    ds_wsg_hausen = xr.open_dataset(output_file, engine="h5netcdf")

    output_file = base_path / "output" / "wsg_zartener_becken.nc"
    ds_wsg_zarten = xr.open_dataset(output_file, engine="h5netcdf")

    # plot recharge data WSG Zartener Becken
    grid_extent = (ds_wsg_zarten.lon.values[0], ds_wsg_zarten.lon.values[-1], ds_wsg_zarten.lat.values[-1], ds_wsg_zarten.lat.values[0])

    fig, axes = plt.subplots(figsize=(6, 4))
    Z = ds_wsg_zarten['indirect_recharge'].values[0, :, :]
    axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
    axes.set_xlim(grid_extent[0], grid_extent[1])
    axes.set_ylim(grid_extent[2], grid_extent[3])
    cb = axes.imshow(Z, extent=grid_extent, cmap='Blues', alpha=1, vmin=0, vmax=2.5)
    plt.colorbar(cb, label='indirect recharge [mm/day]', shrink=0.72)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.tight_layout()
    file = base_path / "figures" / "indirect_recharge_wsg_zarten.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    print(np.nansum(Z))

    fig, axes = plt.subplots(figsize=(6, 4))
    Z = ds_wsg_zarten['direct_recharge'].values[0, :, :]
    axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
    axes.set_xlim(grid_extent[0], grid_extent[1])
    axes.set_ylim(grid_extent[2], grid_extent[3])
    # positive values mean groundwater flows into the river (gaining streams)
    # negative values mean rivers leak water into the groundwater (losing streams)
    cb = axes.imshow(Z, extent=grid_extent, cmap='RdYlBu', alpha=1)
    plt.colorbar(cb, label='SW-GW interaction [$m^3$/day]', shrink=0.72)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.tight_layout()
    file = base_path / "figures" / "direct_recharge_wsg_zarten.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    cond = (Z >= 0)
    Z[cond] = 0
    print(np.nansum(Z * (-1)))

    for i in range(4):
        fig, axes = plt.subplots(figsize=(6, 4))
        Z = ds_wsg_zarten['kf'].values[i, :, :]
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
        axes.set_xlim(grid_extent[0], grid_extent[1])
        axes.set_ylim(grid_extent[2], grid_extent[3])
        cb = axes.imshow(Z, extent=grid_extent, cmap='Oranges', alpha=1, aspect='equal', vmin=10e-7, vmax=10e-2, norm='log')
        plt.colorbar(cb, label='$k_f$ [m/s]', shrink=0.72)
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.tight_layout()
        file = base_path / "figures" / f"kf_wsg_zarten_layer{i}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    for i in range(4):
        fig, axes = plt.subplots(figsize=(6, 4))
        Z = ds_wsg_zarten['kf_fudged'].values[i, :, :]
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
        axes.set_xlim(grid_extent[0], grid_extent[1])
        axes.set_ylim(grid_extent[2], grid_extent[3])
        cb = axes.imshow(Z, extent=grid_extent, cmap='Oranges', alpha=1, aspect='equal', vmin=10e-7, vmax=10e-2, norm='log')
        plt.colorbar(cb, label='$k_f$ [m/s]', shrink=0.72)
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.tight_layout()
        file = base_path / "figures" / f"kf_fudged_wsg_zarten_layer{i}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    for i in range(4):
        fig, axes = plt.subplots(figsize=(6, 4))
        Z = ds_wsg_zarten['sy'].values[i, :, :]
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
        axes.set_xlim(grid_extent[0], grid_extent[1])
        axes.set_ylim(grid_extent[2], grid_extent[3])
        cb = axes.imshow(Z, extent=grid_extent, cmap='Blues', alpha=1, aspect='equal', vmin=0.01, vmax=0.3)
        plt.colorbar(cb, label='specific yield [-]', shrink=0.72)
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.tight_layout()
        file = base_path / "figures" / f"sy_wsg_zarten_layer{i}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    for i in range(4):
        fig, axes = plt.subplots(figsize=(6, 4))
        Z = ds_wsg_zarten['sy_fudged'].values[i, :, :]
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
        axes.set_xlim(grid_extent[0], grid_extent[1])
        axes.set_ylim(grid_extent[2], grid_extent[3])
        cb = axes.imshow(Z, extent=grid_extent, cmap='Blues', alpha=1, aspect='equal', vmin=0.01, vmax=0.3)
        plt.colorbar(cb, label='specific yield [-]', shrink=0.72)
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.tight_layout()
        file = base_path / "figures" / f"sy_fudged_wsg_zarten_layer{i}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    for i in range(4):
        fig, axes = plt.subplots(figsize=(6, 4))
        Z = ds_wsg_zarten['thickness'].values[i, :, :]
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
        axes.set_xlim(grid_extent[0], grid_extent[1])
        axes.set_ylim(grid_extent[2], grid_extent[3])
        cb = axes.imshow(Z, extent=grid_extent, cmap='Blues', alpha=1, aspect='equal', vmin=0.0, vmax=25)
        plt.colorbar(cb, label='specific yield [-]', shrink=0.72)
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.tight_layout()
        file = base_path / "figures" / f"thickness_wsg_zarten_layer{i}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    fig, axes = plt.subplots(figsize=(6, 4))
    Z = ds_wsg_zarten['sfr_flow'].values[0, :, :]
    axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
    axes.set_xlim(grid_extent[0], grid_extent[1])
    axes.set_ylim(grid_extent[2], grid_extent[3])
    cb = axes.imshow(Z, extent=grid_extent, cmap='Blues', alpha=1, vmin=0, vmax=2.5)
    plt.colorbar(cb, label='streamflow [$m^3$/s]', shrink=0.72)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.tight_layout()
    file = base_path / "figures" / "streamflow_wsg_zarten.png"
    fig.savefig(file, dpi=300)
    plt.close("all")


    # plot recharge data WSG Hausen
    grid_extent = (ds_wsg_hausen.lon.values[0], ds_wsg_hausen.lon.values[-1], ds_wsg_hausen.lat.values[-1], ds_wsg_hausen.lat.values[0])

    fig, axes = plt.subplots(figsize=(6, 4))
    Z = ds_wsg_hausen['indirect_recharge'].values[0, :, :]
    axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
    axes.set_xlim(grid_extent[0], grid_extent[1])
    axes.set_ylim(grid_extent[2], grid_extent[3])
    cb = axes.imshow(Z, extent=grid_extent, cmap='Blues', alpha=1, vmin=0, vmax=2.5)
    plt.colorbar(cb, label='indirect recharge [mm/day]', shrink=0.72)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.tight_layout()
    file = base_path / "figures" / "indirect_recharge_wsg_hausen.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    print(np.nansum(Z))

    fig, axes = plt.subplots(figsize=(6, 4))
    Z = ds_wsg_hausen['direct_recharge'].values[0, :, :]
    axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
    axes.set_xlim(grid_extent[0], grid_extent[1])
    axes.set_ylim(grid_extent[2], grid_extent[3])
    # positive values mean groundwater flows into the river (gaining streams)
    # negative values mean rivers leak water into the groundwater(losing streams)
    cb = axes.imshow(Z, extent=grid_extent, cmap='RdYlBu', alpha=1)
    plt.colorbar(cb, label='SW-GW interaction [$m^3$/day]', shrink=0.72)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.tight_layout()
    file = base_path / "figures" / "direct_recharge_wsg_hausen.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    cond = (Z >= 0)
    Z[cond] = 0
    print(np.nansum(Z * (-1)))


    return


if __name__ == "__main__":
    main()