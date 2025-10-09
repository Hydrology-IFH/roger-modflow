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

@click.command("main", short_help="Plot the reach data")
def main():
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
        modflow_config = yaml.safe_load(file)
    grid_extent = (0, (modflow_config['ny']*modflow_config['dy']) / 1000, (modflow_config['nx']*modflow_config['dx']) / 1000, 0)

    # load MODFLOW parameters
    path = base_path / "input" / "parameters_modflow.nc"
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

    reaches = pd.read_csv(base_path / 'input' / 'sfr_packagedata_modified.csv', sep=';')
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
        # if z == 0:
        #     reaches.loc[rno, 'rhk'] = hydraulic_conductivities_layer1[x, y] * 10e-3
        #     kf_reach_cell[x, y] = hydraulic_conductivities_layer1[x, y] / 86400
        # elif z == 1:
        #     reaches.loc[rno, 'rhk'] = hydraulic_conductivities_layer2[x, y] * 10e-3
        #     kf_reach_cell[x, y] = hydraulic_conductivities_layer2[x, y] / 86400
        # elif z == 2:
        #     reaches.loc[rno, 'rhk'] = hydraulic_conductivities_layer3[x, y] * 10e-3
        #     kf_reach_cell[x, y] = hydraulic_conductivities_layer3[x, y] / 86400
        # elif z == 3:
        #     reaches.loc[rno, 'rhk'] = hydraulic_conductivities_layer4[x, y] * 10e-3
        #     kf_reach_cell[x, y] = hydraulic_conductivities_layer3[x, y] / 86400
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

    bounds = [10e-10, 10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3]
    norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Oranges"].N)
    fig, axes = plt.subplots(figsize=(4, 4))
    topography[mask_reach] = np.nan
    plt.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal', alpha=0.5)
    plt.imshow(rhk, extent=grid_extent, cmap='Oranges', aspect='equal', norm=norm)
    cbar = plt.colorbar(label='$k_f$ of streambed [m/s]', shrink=0.46)
    cbar.set_ticks(ticks=bounds, labels=[r'$10^{-10}$', r'$10^{-9}$', r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$'])
    stage_x = np.array([591, 430, 191, 218]) * (50/1000)
    stage_y = np.array([309, 207, 360, 487]) * (50/1000)
    plt.scatter(stage_x, stage_y, marker='.', color='black', s=5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / "figures" / "kf_streambed_.png"
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
    topography[mask_reach] = np.nan
    plt.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal', alpha=0.5)
    plt.imshow(rwid, extent=grid_extent, cmap='Oranges', aspect='equal')
    plt.colorbar(label='reach width [m]', shrink=0.5)
    stage_x = np.array([591, 430, 191, 218]) * (50/1000)
    stage_y = np.array([309, 207, 360, 487]) * (50/1000)
    plt.scatter(stage_x, stage_y, marker='.', color='black', s=5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / "figures" / "reach_width_.png"
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

    fig, axes = plt.subplots(figsize=(4, 4))
    topography[mask_reach] = np.nan
    plt.imshow(topography, extent=grid_extent, cmap='terrain', aspect='equal', alpha=0.5)
    plt.imshow(man, extent=grid_extent, cmap='Oranges', aspect='equal')
    plt.colorbar(label='Manning’s roughness coefficient [-]', shrink=0.5)
    stage_x = np.array([591, 430, 191, 218]) * (50/1000)
    stage_y = np.array([309, 207, 360, 487]) * (50/1000)
    plt.scatter(stage_x, stage_y, marker='.', color='black', s=5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / "figures" / "manning_.png"
    fig.savefig(file, dpi=300)
    plt.close("all")


    return


if __name__ == "__main__":
    main()