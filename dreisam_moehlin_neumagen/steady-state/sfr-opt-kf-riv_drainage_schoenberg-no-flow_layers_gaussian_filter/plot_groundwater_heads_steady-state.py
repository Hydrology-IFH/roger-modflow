import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
import pandas as pd
import scipy
import click
import matplotlib as mpl

@click.option("-mr", "--model-run", type=int, default=50)
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

    path = base_path / "fudge_parameters_modflow.csv"
    fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)

    res_modflow = 50  # spatial resolution of MODFLOW in meters

    modflow_config = {
        'dx': res_modflow,
        'dy': res_modflow,
        'nx': 621,
        'ny': 777,
        'nz': 4,
    }
    grid_extent = (0, (777*modflow_config['dy']) / 1000, (621*modflow_config['dx']) / 1000, 0)

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

    mask_ = np.isfinite(topography)
    mask_schoenberg = ds_params['mask_schoenberg'].values
    mask = mask_ & (mask_schoenberg == False)
    domain = np.empty_like(topography)
    domain[mask] = 1
    domain[~mask] = -1

    hydraulic_conductivities_layer1 = ds_params['kf'].isel(layer=0).values
    hydraulic_conductivities_layer2 = ds_params['kf'].isel(layer=1).values
    hydraulic_conductivities_layer3 = ds_params['kf'].isel(layer=2).values
    hydraulic_conductivities_layer4 = ds_params['kf'].isel(layer=3).values


    hydraulic_conductivities_layer1_ = ds_params['kf'].isel(layer=0).values / 86400
    hydraulic_conductivities_layer2_ = ds_params['kf'].isel(layer=1).values / 86400
    hydraulic_conductivities_layer3_ = ds_params['kf'].isel(layer=2).values / 86400
    hydraulic_conductivities_layer4_ = ds_params['kf'].isel(layer=3).values / 86400
    
    # fudge parameters
    mask1 = (hydraulic_conductivities_layer1_ <= 10e-10)
    mask2 = (hydraulic_conductivities_layer2_ <= 10e-10)
    mask3 = (hydraulic_conductivities_layer3_ <= 10e-10)
    mask4 = (hydraulic_conductivities_layer4_ <= 10e-10)  
    hydraulic_conductivities_layer1[mask1] = hydraulic_conductivities_layer1[mask1] * 10000
    hydraulic_conductivities_layer2[mask2] = hydraulic_conductivities_layer2[mask2] * 10000
    hydraulic_conductivities_layer3[mask3] = hydraulic_conductivities_layer3[mask3] * 10000
    hydraulic_conductivities_layer4[mask4] = hydraulic_conductivities_layer4[mask4] * 10000

    mask_ = (hydraulic_conductivities_layer2_ == 1.9999999e-07)
    hydraulic_conductivities_layer2_ = np.where(mask_, 1.9722222e-07, hydraulic_conductivities_layer2_)
    mask_ = (hydraulic_conductivities_layer3_ == 1.9999999e-07)
    hydraulic_conductivities_layer3_ = np.where(mask_, 1.9722222e-07, hydraulic_conductivities_layer3_)
    mask_ = (hydraulic_conductivities_layer4_ == 1.9999999e-07)
    hydraulic_conductivities_layer4_ = np.where(mask_, 1.9722222e-07, hydraulic_conductivities_layer4_)

    mask81 = (hydraulic_conductivities_layer1_ == 1.1574075e-08) | (hydraulic_conductivities_layer1_ == 2.7777778e-08)

    mask71 = (hydraulic_conductivities_layer1_ == 1.9444444e-07) | (hydraulic_conductivities_layer1_ == 1.9722222e-07) | (hydraulic_conductivities_layer1_ == 2.3055554e-07) | (hydraulic_conductivities_layer1_ == 5.7777777e-07)
    mask72 = (hydraulic_conductivities_layer2_ == 1.9722222e-07)
    mask73 = (hydraulic_conductivities_layer3_ == 1.9722222e-07)
    mask74 = (hydraulic_conductivities_layer4_ == 1.9722222e-07)

    mask61 = (hydraulic_conductivities_layer1_ >= 1.1583334e-06) & (hydraulic_conductivities_layer1_ <= 8.1027783e-06)

    mask51 = (hydraulic_conductivities_layer1_ == 1.1575000e-05) | (hydraulic_conductivities_layer1_ == 1.8181944e-04)
    mask52 = (hydraulic_conductivities_layer2_ == 1.8180555e-05)
    mask53 = (hydraulic_conductivities_layer3_ == 1.8180555e-05)
    mask54 = (hydraulic_conductivities_layer4_ == 1.8180555e-05)

    mask42 = (hydraulic_conductivities_layer2_ == 1.8181944e-04)
    mask43 = (hydraulic_conductivities_layer3_ == 1.8181944e-04)
    mask44 = (hydraulic_conductivities_layer4_ == 1.8181944e-04)

    mask132 = (hydraulic_conductivities_layer2_ == 1.0000000e-03)
    mask133 = (hydraulic_conductivities_layer3_ == 1.0000000e-03)

    mask232 = (hydraulic_conductivities_layer2_ == 1.8181807e-03)
    mask233 = (hydraulic_conductivities_layer3_ == 1.8181807e-03)
    mask234 = (hydraulic_conductivities_layer4_ == 1.8181807e-03)

    mask332 = (hydraulic_conductivities_layer2_ == 3.0000000e-03)
    mask333 = (hydraulic_conductivities_layer3_ == 3.0000000e-03)

    mask432 = (hydraulic_conductivities_layer2_ == 4.0000002e-03)
    mask433 = (hydraulic_conductivities_layer3_ == 4.0000002e-03)

    # fudge parameters
    hydraulic_conductivities_layer1[mask81] = hydraulic_conductivities_layer1[mask81] * fudge_parameters['-8_1'].values[model_run]

    hydraulic_conductivities_layer1[mask71] = hydraulic_conductivities_layer1[mask71] * fudge_parameters['-7_1'].values[model_run]
    hydraulic_conductivities_layer2[mask72] = hydraulic_conductivities_layer2[mask72] * fudge_parameters['-7_2'].values[model_run]
    hydraulic_conductivities_layer3[mask73] = hydraulic_conductivities_layer3[mask73] * fudge_parameters['-7_3'].values[model_run]
    hydraulic_conductivities_layer4[mask74] = hydraulic_conductivities_layer4[mask74] * fudge_parameters['-7_4'].values[model_run]

    hydraulic_conductivities_layer1[mask61] = hydraulic_conductivities_layer1[mask61] * fudge_parameters['-6_1'].values[model_run]

    hydraulic_conductivities_layer1[mask51] = hydraulic_conductivities_layer1[mask51] * fudge_parameters['-5_1'].values[model_run]
    hydraulic_conductivities_layer2[mask52] = hydraulic_conductivities_layer2[mask52] * fudge_parameters['-5_2'].values[model_run]
    hydraulic_conductivities_layer3[mask53] = hydraulic_conductivities_layer3[mask53] * fudge_parameters['-5_3'].values[model_run]
    hydraulic_conductivities_layer4[mask54] = hydraulic_conductivities_layer4[mask54] * fudge_parameters['-5_4'].values[model_run]

    hydraulic_conductivities_layer2[mask42] = hydraulic_conductivities_layer2[mask42] * fudge_parameters['-4_2'].values[model_run]
    hydraulic_conductivities_layer3[mask43] = hydraulic_conductivities_layer3[mask43] * fudge_parameters['-4_3'].values[model_run]
    hydraulic_conductivities_layer4[mask44] = hydraulic_conductivities_layer4[mask44] * fudge_parameters['-4_4'].values[model_run]

    hydraulic_conductivities_layer2[mask132] = hydraulic_conductivities_layer2[mask132] * fudge_parameters['1-3_2'].values[model_run]
    hydraulic_conductivities_layer3[mask133] = hydraulic_conductivities_layer3[mask133] * fudge_parameters['1-3_3'].values[model_run]

    hydraulic_conductivities_layer2[mask232] = hydraulic_conductivities_layer2[mask232] * fudge_parameters['1.8-3_2'].values[model_run]
    hydraulic_conductivities_layer3[mask233] = hydraulic_conductivities_layer3[mask233] * fudge_parameters['1.8-3_3'].values[model_run]
    hydraulic_conductivities_layer4[mask234] = hydraulic_conductivities_layer4[mask234] * fudge_parameters['1.8-3_4'].values[model_run]

    hydraulic_conductivities_layer2[mask332] = hydraulic_conductivities_layer2[mask332] * fudge_parameters['3-3_2'].values[model_run]
    hydraulic_conductivities_layer3[mask333] = hydraulic_conductivities_layer3[mask333] * fudge_parameters['3-3_3'].values[model_run]

    hydraulic_conductivities_layer2[mask432] = hydraulic_conductivities_layer2[mask432] * fudge_parameters['4-3_2'].values[model_run]
    hydraulic_conductivities_layer3[mask433] = hydraulic_conductivities_layer3[mask433] * fudge_parameters['4-3_3'].values[model_run]

    # prepare SFR data
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

    cond = np.isnan(reaches['rwid'])
    reaches.loc[cond, 'rwid'] = 1.0  # set width to 1 m where it is NaN
    cond_widht0 = (reaches.loc[:, 'rwid'] <= 1.0)
    reaches.loc[cond_widht0, 'rwid'] = 1.0  # set width to 1 m if it is smaller than 1 m

    rwid = np.empty_like(topography)
    rwid[:, :] = np.nan
    rlen = np.empty_like(topography)
    rlen[:, :] = np.nan

    # increase the hydraulic conductivities of the reach cell by a factor of xx
    for rno, z, x, y in zip(reaches.iloc[:, 0], reaches.iloc[:, 1], reaches.iloc[:, 2], reaches.iloc[:, 3]):
        rwid[x, y] = reaches.loc[rno, 'rwid']
        rlen[x, y] = reaches.loc[rno, 'rlen']
        if z == 0:
            hydraulic_conductivities_layer1[x, y] = hydraulic_conductivities_layer1[x, y] * fudge_parameters['kf_riv'].values[model_run]
        elif z == 1:
            hydraulic_conductivities_layer1[x, y] = hydraulic_conductivities_layer1[x, y] * fudge_parameters['kf_riv'].values[model_run]
            hydraulic_conductivities_layer2[x, y] = hydraulic_conductivities_layer2[x, y] * fudge_parameters['kf_riv'].values[model_run]
        elif z == 2:
            hydraulic_conductivities_layer1[x, y] = hydraulic_conductivities_layer1[x, y] * fudge_parameters['kf_riv'].values[model_run]
            hydraulic_conductivities_layer2[x, y] = hydraulic_conductivities_layer2[x, y] * fudge_parameters['kf_riv'].values[model_run]
            hydraulic_conductivities_layer3[x, y] = hydraulic_conductivities_layer3[x, y] * fudge_parameters['kf_riv'].values[model_run]
        elif z == 3:
            hydraulic_conductivities_layer1[x, y] = hydraulic_conductivities_layer1[x, y] * fudge_parameters['kf_riv'].values[model_run]
            hydraulic_conductivities_layer2[x, y] = hydraulic_conductivities_layer2[x, y] * fudge_parameters['kf_riv'].values[model_run]
            hydraulic_conductivities_layer3[x, y] = hydraulic_conductivities_layer3[x, y] * fudge_parameters['kf_riv'].values[model_run]
            hydraulic_conductivities_layer4[x, y] = hydraulic_conductivities_layer4[x, y] * fudge_parameters['kf_riv'].values[model_run]

    rarea = rlen * rwid

    # smooth transition between fissured and porous aquifers
    hydraulic_conductivities_layer1[np.isnan(hydraulic_conductivities_layer1)] = 0
    hydraulic_conductivities_layer2[np.isnan(hydraulic_conductivities_layer2)] = 0
    hydraulic_conductivities_layer3[np.isnan(hydraulic_conductivities_layer3)] = 0
    hydraulic_conductivities_layer4[np.isnan(hydraulic_conductivities_layer4)] = 0
    hydraulic_conductivities_layer1 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer1, [1.0, 1.0], mode='constant')
    hydraulic_conductivities_layer2 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer2, [1.0, 1.0], mode='constant')
    hydraulic_conductivities_layer3 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer3, [1.0, 1.0], mode='constant')
    hydraulic_conductivities_layer4 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer4, [1.0, 1.0], mode='constant')
    hydraulic_conductivities_layer1[~mask] = np.nan
    hydraulic_conductivities_layer2[~mask] = np.nan
    hydraulic_conductivities_layer3[~mask] = np.nan
    hydraulic_conductivities_layer4[~mask] = np.nan

    hydraulic_conductivities_layers = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]

    model_type = "steady-state"
    base_path_figs = base_path / "figures"
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=base_path / "output",
        exe_name="mf6",
        version="mf6",
        verbosity_level=0,
    )

    ml = sim.get_model(f"dmn_run_{model_run}")

    # load the netcdf file
    output_file = base_path / "output" / f"modflow_output_run_{model_run}.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")


    # plot the groundwater-surface water interaction
    gw_sw = np.nanmean(ds_mf['gw_sw'].isel(Time=0).values, axis=0) / 86400
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(gw_sw * (-1), extent=grid_extent, cmap='RdYlBu', aspect='equal', vmin=-0.01, vmax=0.01)
    plt.colorbar(label='GW-SW flux \n[$m^3$/s]', shrink=0.42)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"gw-sw_steady_state_grid_{model_run}_m3_s.png"
    fig.savefig(file, dpi=600)
    plt.close("all")

    minmax = np.nanmax(np.abs(gw_sw))
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(gw_sw * (-1), extent=grid_extent, cmap='RdYlBu', aspect='equal', vmin=-minmax, vmax=minmax)
    plt.colorbar(label='GW-SW flux \n[$m^3$/s]', shrink=0.42)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"gw-sw_steady_state_grid_{model_run}_m3_s_.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    gw_sw = np.nanmean(ds_mf['gw_sw'].isel(Time=0).values, axis=0) / rarea
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(gw_sw * (-1), extent=grid_extent, cmap='RdYlBu', aspect='equal', vmin=-10, vmax=10)
    plt.colorbar(label='GW-SW flux \n[mm/day]', shrink=0.42)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"gw-sw_steady_state_grid_{model_run}_mm_day.png"
    fig.savefig(file, dpi=600)
    plt.close("all")

    minmax = np.nanmax(np.abs(gw_sw))
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(gw_sw * (-1), extent=grid_extent, cmap='RdYlBu', aspect='equal', vmin=-minmax, vmax=minmax)
    plt.colorbar(label='GW-SW flux \n[mm/day]', shrink=0.42)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"gw-sw_steady_state_grid_{model_run}_mm_day_.png"
    fig.savefig(file, dpi=300)
    plt.close("all")



    x = np.cumsum(ds_mf.lon.values - ds_mf.lon.values[0])
    y = np.cumsum(ds_mf.lat.values - ds_mf.lat.values[-1])
    yr = y[::-1]

    ll_levels = [[200, 220, 300, 400, 500, 600],
                [200, 220, 300, 400, 500, 600],
                [200, 220, 300, 400, 500, 600],
                [200, 220, 300, 400, 500, 600]]

    for layer in range(4):
        fig, axes = plt.subplots(figsize=(4, 4))
        plt.imshow(ds_mf['head'].isel(Time=0, layer=layer).values, extent=grid_extent, cmap='viridis', aspect='equal', vmin=100, vmax=600)
        plt.colorbar(label='groundwater head \n[m a.s.l.]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_head_steady_state_layer{i}_grid_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        elevation_bottom_layer = elevation_bottom_layers[layer]
        gw_thickness = ds_mf['head'].isel(Time=0, layer=layer).values - elevation_bottom_layer
        gw_thickness[gw_thickness <= 0] = 0
        plt.imshow(gw_thickness, extent=grid_extent, cmap='viridis', aspect='equal')
        plt.colorbar(label='groundwater thickness [m]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_thickness_steady_state_layer{i}_grid_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        gw_depth = topography - ds_mf['head'].isel(Time=0, layer=layer).values
        plt.imshow(gw_depth, extent=grid_extent, cmap='viridis', aspect='equal', vmin=0, vmax=20)
        plt.colorbar(label='groundwater depth [m]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_depth_steady_state_layer{i}_grid_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        y = np.arange(0, modflow_config['nx']*modflow_config['dx'], modflow_config['dx'])
        x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
        X, Y = np.meshgrid(x, y)
        Z = ds_mf['head'].isel(Time=0, layer=layer).values
        levels = ll_levels[layer]
        CS = axes.contour(X, Y, Z, levels, colors='black')
        axes.clabel(CS, inline=True, fontsize=8, colors='black')
        axes.imshow(mask, extent=grid_extent, cmap='Greys', alpha=0.25)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.title(f"Groundwater head of layer {layer + 1} [m a.s.l.]", fontsize=8)
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_head_steady_state_layer{i}_contour_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    # wells_y = [266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]
    # wells_x = [66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]

    # for yy in wells_y:
    #     fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=True)
    #     x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
    #     for layer in range(4):
    #         z0 = topography[yy, :]
    #         z1 = elevation_bottom_layer1[yy, :]
    #         z2 = elevation_bottom_layer2[yy, :]
    #         z3 = elevation_bottom_layer3[yy, :]
    #         z4 = elevation_bottom_layer4[yy, :]
    #         z = ds_mf['head'].isel(Time=0, layer=layer).values[yy, :]
    #         axes[layer].plot(x, z, ls='-', lw=1, color='black')
    #         axes[layer].plot(x, z0, ls='-', lw=1, color='grey')
    #         axes[layer].plot(x, z1, ls='-', lw=1, color='grey')
    #         axes[layer].plot(x, z2, ls='-', lw=1, color='grey')
    #         axes[layer].plot(x, z3, ls='-', lw=1, color='grey')
    #         axes[layer].plot(x, z4, ls='-', lw=1, color='grey')
    #         axes[layer].set_xlim(0, x[-1])
    #         axes[layer].set_ylabel('[m a.s.l.]')
    #     axes[-1].set_xlabel('Distance in W-E direction [m]')
    #     fig.tight_layout()
    #     file = base_path_figs / f"gw_head_x{yy}_cross_section_layer.png"
    #     fig.savefig(file, dpi=300)
    #     plt.close("all")

    # for xx in wells_x:
    #     fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=True)
    #     x = np.arange(0, modflow_config['nx']*modflow_config['dx'], modflow_config['dx'])
    #     for layer in range(4):
    #         z0 = topography[:, xx]
    #         z1 = elevation_bottom_layer1[:, xx]
    #         z2 = elevation_bottom_layer2[:, xx]
    #         z3 = elevation_bottom_layer3[:, xx]
    #         z4 = elevation_bottom_layer4[:, xx]
    #         z = ds_mf['head'].isel(Time=0, layer=layer).values[:, xx]
    #         axes[layer].plot(x, z, ls='-', lw=1, color='black')
    #         axes[layer].plot(x, z0, ls='-', lw=1, color='grey')
    #         axes[layer].plot(x, z1, ls='-', lw=1, color='grey')
    #         axes[layer].plot(x, z2, ls='-', lw=1, color='grey')
    #         axes[layer].plot(x, z3, ls='-', lw=1, color='grey')
    #         axes[layer].plot(x, z4, ls='-', lw=1, color='grey')
    #         axes[layer].set_xlim(0, x[-1])
    #         axes[layer].set_ylabel('[m a.s.l.]')
    #     axes[-1].set_xlabel('Distance in N-S direction [m]')
    #     fig.tight_layout()
    #     file = base_path_figs / f"gw_head_y{xx}_cross_section_layer.png"
    #     fig.savefig(file, dpi=300)
    #     plt.close("all")


    for layer in range(4):
        hydraulic_conductivity = hydraulic_conductivities_layers[layer]
        fig, axes = plt.subplots(figsize=(4, 4))
        bounds = [10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1]
        norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Oranges"].N)
        plt.imshow(hydraulic_conductivity/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal', norm=norm)
        cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.45)
        cbar.set_ticks(ticks=bounds, labels=[r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"kf_layer{i}_{model_run}_fudged_.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        plt.imshow(hydraulic_conductivity/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal')
        cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.45)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"kf_layer{i}_{model_run}_fudged.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    for layer in range(4):
        flow_residuals = ds_mf['flow_residual'].isel(Time=0, layer=layer).values
        flow_residuals[~mask] = np.nan
        mask1 = (flow_residuals <= 0.1) & (flow_residuals >= -0.1)

        thickness = elevation_layers[layer] - elevation_layers[layer + 1]
        thickness[mask1] = np.nan
        fig, axes = plt.subplots(figsize=(4, 4))
        plt.imshow(thickness, extent=grid_extent, cmap='viridis', aspect='equal', vmin=5, vmax=25)
        plt.colorbar(label='thickness [m]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"__thickness_layer{i}_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        hydraulic_conductivity = hydraulic_conductivities_layers[layer]
        hydraulic_conductivity[mask1] = np.nan
        hydraulic_conductivity[~mask] = np.nan
        fig, axes = plt.subplots(figsize=(4, 4))
        bounds = [10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1]
        norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Oranges"].N)
        plt.imshow(hydraulic_conductivity/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal', norm=norm)
        cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.45)
        cbar.set_ticks(ticks=bounds, labels=[r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"__kf_layer{i}_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        plt.imshow(hydraulic_conductivity/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal')
        cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.45)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"___kf_layer{i}_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")


    for layer in range(4):
        gw_depth = topography - ds_mf['head'].isel(Time=0, layer=layer).values
        gw_depth[~mask] = np.nan
        mask1 = (gw_depth > 0)
        mask2 = (gw_depth < 0)
        gw_depth[mask1] = np.nan
        fig, axes = plt.subplots(figsize=(4, 4))
        plt.imshow(gw_depth * (-1), extent=grid_extent, cmap='viridis_r', aspect='equal')
        plt.colorbar(label='groundwater above surface [m]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"_gw_depth_steady_state_layer{i}_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")
        print(f"Layer {layer} (# GW table above surface): {np.sum(mask2)}")

        thickness = elevation_layers[layer] - elevation_layers[layer + 1]
        thickness[mask1] = np.nan
        thickness[~mask] = np.nan
        fig, axes = plt.subplots(figsize=(4, 4))
        plt.imshow(thickness, extent=grid_extent, cmap='viridis', aspect='equal', vmin=5, vmax=25)
        plt.colorbar(label='thickness [m]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"_thickness_layer{i}_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        hydraulic_conductivity_ = hydraulic_conductivities_layers[layer]
        hydraulic_conductivity_[mask1] = np.nan
        hydraulic_conductivity_[~mask] = np.nan
        fig, axes = plt.subplots(figsize=(4, 4))
        bounds = [10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1]
        norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Oranges"].N)
        plt.imshow(hydraulic_conductivity_/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal', norm=norm)
        cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.45)
        cbar.set_ticks(ticks=bounds, labels=[r'$10^{-8}$', r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"_kf_layer{i}_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")


    return


if __name__ == "__main__":
    main()