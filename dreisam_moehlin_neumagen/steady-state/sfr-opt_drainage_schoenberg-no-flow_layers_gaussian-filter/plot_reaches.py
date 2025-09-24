import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import click
import matplotlib as mpl

@click.option("-mr", "--model-run", type=int, default=5)
@click.command("main", short_help="Run MODFLOW in steady-state mode")
def main(model_run):
    # run installed version of flopy or add local path
    try:
        import flopy
    except:
        fpth = os.path.abspath(os.path.join("..", ".."))
        sys.path.append(fpth)
        import flopy

    base_path = Path(__file__).parent

    # load the config file
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        roger_config = yaml.safe_load(file)

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
    ds_params = xr.open_dataset(path, engine="h5netcdf")

    topography = ds_params['elevations'].isel(z=0).values
    elevation_bottom_layer1 = ds_params['elevations'].isel(z=1).values
    elevation_bottom_layer2 = ds_params['elevations'].isel(z=2).values
    elevation_bottom_layer3 = ds_params['elevations'].isel(z=3).values
    elevation_bottom_layer4 = ds_params['elevations'].isel(z=4).values
    elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]
    elevation_layers = [topography, elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

    hydraulic_conductivities_layer1 = ds_params['kf'].isel(layer=0).values
    hydraulic_conductivities_layer2 = ds_params['kf'].isel(layer=1).values
    hydraulic_conductivities_layer3 = ds_params['kf'].isel(layer=2).values
    hydraulic_conductivities_layer4 = ds_params['kf'].isel(layer=3).values
    hydraulic_conductivities_layers = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]

    mask_ = np.isfinite(topography)
    mask_schoenberg = ds_params['mask_schoenberg'].values
    mask = mask_ & (mask_schoenberg == False)
    domain = np.empty_like(topography)
    domain[mask] = 1
    domain[~mask] = -1

    mask_reach = np.zeros_like(topography, dtype=bool)
    mask_reach[:, :] = False

    reaches = pd.read_csv(base_path.parent / 'input' / 'sfr_packagedata.csv', sep=';')
    reaches.iloc[:, 0] = reaches.iloc[:, 0].astype(int) - 1  # convert to zero-based indexing
    reaches.iloc[:, 1] = reaches.iloc[:, 1].astype(int) - 1
    reaches.iloc[:, 2] = reaches.iloc[:, 2].astype(int) - 1  # convert to zero-based indexing
    reaches.iloc[:, 3] = reaches.iloc[:, 3].astype(int) - 1  # convert to zero-based indexing
    reaches.iloc[:, 4] = reaches.iloc[:, 4].astype(float) 
    reaches.iloc[:, 5] = reaches.iloc[:, 5].astype(int)
    reaches.iloc[:, 6] = reaches.iloc[:, 6].astype(float)
    reaches.iloc[:, 7] = reaches.iloc[:, 7].astype(float)
    reaches.iloc[:, 8] = reaches.iloc[:, 8].astype(float)
    reaches.iloc[:, 9] = reaches.iloc[:, 9].astype(float)
    reaches.iloc[:, 10] = reaches.iloc[:, 10].astype(float)
    reaches.iloc[:, 11] = reaches.iloc[:, 11].astype(int)
    reaches.iloc[:, 12] = reaches.iloc[:, 12].astype(float)
    reaches.iloc[:, 13] = reaches.iloc[:, 13].astype(int)

    # set the Manning’s roughness coefficient depending on the streambed gradient
    cond1 = (reaches['rgrd'] <= 0.0006)
    cond2 = (reaches['rgrd'] > 0.0006) & (reaches['rgrd'] <= 0.0025)
    cond3 = (reaches['rgrd'] > 0.0025) & (reaches['rgrd'] <= 0.01)
    cond4 = (reaches['rgrd'] > 0.01) & (reaches['rgrd'] <= 0.04)
    cond5 = (reaches['rgrd'] > 0.04) & (reaches['rgrd'] <= 0.07)
    cond6 = (reaches['rgrd'] > 0.07)
    reaches.loc[cond1, 'man'] = 1/50
    reaches.loc[cond2, 'man'] = 1/30
    reaches.loc[cond3, 'man'] = 1/25
    reaches.loc[cond4, 'man'] = 1/20
    reaches.loc[cond5, 'man'] = 1/15
    reaches.loc[cond6, 'man'] = 1/10   

    elev_diff = np.empty_like(topography)
    elev_diff[:, :] = np.nan
    rhk = np.empty_like(topography)
    rhk[:, :] = np.nan
    man = np.empty_like(topography)
    man[:, :] = np.nan
    reach_layer = np.empty_like(topography)
    reach_layer[:, :] = np.nan
    rwid = np.empty_like(topography)
    rwid[:, :] = np.nan
    kf_reach_cell = np.empty_like(topography)
    kf_reach_cell[:, :] = np.nan
    for rno, z, x, y in zip(reaches.iloc[:, 0], reaches.iloc[:, 1], reaches.iloc[:, 2], reaches.iloc[:, 3]):
        mask_reach[x, y] = True
        elev_diff[x, y] = reaches.loc[rno, 'rtp'] - topography[x, y]
        man[x, y] = reaches.loc[rno, 'man']
        reach_layer[x, y] = z + 1
        if z == 0:
            hydraulic_conductivities_layer1[x, y] = hydraulic_conductivities_layer1[x, y] * 5
        elif z == 1:
            hydraulic_conductivities_layer1[x, y] = hydraulic_conductivities_layer1[x, y] * 5
            hydraulic_conductivities_layer2[x, y] = hydraulic_conductivities_layer2[x, y] * 5
        elif z == 2:
            hydraulic_conductivities_layer1[x, y] = hydraulic_conductivities_layer1[x, y] * 5
            hydraulic_conductivities_layer2[x, y] = hydraulic_conductivities_layer2[x, y] * 5
            hydraulic_conductivities_layer3[x, y] = hydraulic_conductivities_layer3[x, y] * 5
        elif z == 3:
            hydraulic_conductivities_layer1[x, y] = hydraulic_conductivities_layer1[x, y] * 5
            hydraulic_conductivities_layer2[x, y] = hydraulic_conductivities_layer2[x, y] * 5
            hydraulic_conductivities_layer3[x, y] = hydraulic_conductivities_layer3[x, y] * 5
            hydraulic_conductivities_layer4[x, y] = hydraulic_conductivities_layer4[x, y] * 5 

    # set the hydraulic conductivities of the streambed using the kf of the reach cell and decrease by a factor of 0.0005
    for rno, z, x, y in zip(reaches.iloc[:, 0], reaches.iloc[:, 1], reaches.iloc[:, 2], reaches.iloc[:, 3]):
        if z == 0:
            reaches.loc[rno, 'rhk'] = hydraulic_conductivities_layer1[x, y] * 10e-3
            kf_reach_cell[x, y] = hydraulic_conductivities_layer1[x, y] / 86400
        elif z == 1:
            reaches.loc[rno, 'rhk'] = hydraulic_conductivities_layer2[x, y] * 10e-3
            kf_reach_cell[x, y] = hydraulic_conductivities_layer2[x, y] / 86400
        elif z == 2:
            reaches.loc[rno, 'rhk'] = hydraulic_conductivities_layer3[x, y] * 10e-3
            kf_reach_cell[x, y] = hydraulic_conductivities_layer3[x, y] / 86400
        elif z == 3:
            reaches.loc[rno, 'rhk'] = hydraulic_conductivities_layer4[x, y] * 10e-3
            kf_reach_cell[x, y] = hydraulic_conductivities_layer3[x, y] / 86400
        rhk[x, y] = reaches.loc[rno, 'rhk'] / 86400
        rwid[x, y] = reaches.loc[rno, 'rwid']
        man[x, y] = reaches.loc[rno, 'man']

    bounds = [1, 2, 3, 4, 5]
    norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["viridis_r"].N)
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(reach_layer, extent=grid_extent, aspect='equal', norm=norm, cmap='viridis_r')
    cbar = plt.colorbar(label='# layer', shrink=0.5)
    cbar.set_ticks(ticks=bounds, labels=['1', '2', '3', '4', ''])
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / "figures" / "reach_layer.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    bounds = [10e-10, 10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3]
    norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Oranges"].N)
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(rhk, extent=grid_extent, cmap='Oranges', aspect='equal', norm=norm)
    cbar = plt.colorbar(label='$k_f$ of streambed [m/s]', shrink=0.5)
    cbar.set_ticks(ticks=bounds, labels=[r'$10^{-10}$', r'$10^{-9}$', r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$'])
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / "figures" / "kf_streambed.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    bounds = [10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1]
    norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Oranges"].N)
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(kf_reach_cell, extent=grid_extent, cmap='Oranges', aspect='equal', norm=norm)
    cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.5)
    cbar.set_ticks(ticks=bounds, labels=[r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / "figures" / "kf_stream_cell.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(elev_diff, extent=grid_extent, cmap='Reds', aspect='equal')
    plt.colorbar(label='difference rtp-topo [m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / "figures" / "difference_rtp-topo.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(rwid, extent=grid_extent, cmap='Oranges', aspect='equal')
    plt.colorbar(label='reach width [m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / "figures" / "reach_width.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(man, extent=grid_extent, cmap='Purples', aspect='equal')
    plt.colorbar(label='Manning’s roughness coefficient [-]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / "figures" / "manning.png"
    fig.savefig(file, dpi=300)
    plt.close("all")


    return


if __name__ == "__main__":
    main()