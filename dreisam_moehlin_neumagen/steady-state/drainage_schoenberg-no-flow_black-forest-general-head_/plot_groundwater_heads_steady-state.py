import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from flopy.utils import Raster
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

    x = np.cumsum(ds_mf.lon.values - ds_mf.lon.values[0])
    y = np.cumsum(ds_mf.lat.values - ds_mf.lat.values[-1])
    yr = y[::-1]

    ll_levels = [[200, 225, 250, 300, 350],
                 [200, 225, 250, 300, 350],
                 [200, 225, 250, 300, 350],
                 [200, 225, 250, 300, 350]]

    for layer in range(4):
        fig, axes = plt.subplots(figsize=(4, 4))
        plt.imshow(ds_mf['head'].isel(Time=0, layer=layer).values, extent=grid_extent, cmap='viridis', aspect='equal', vmin=200, vmax=380)
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

        hydraulic_conductivity = hydraulic_conductivities_layers[layer]
        hydraulic_conductivity[mask1] = np.nan
        hydraulic_conductivity[~mask] = np.nan
        fig, axes = plt.subplots(figsize=(4, 4))
        bounds = [0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
        norm = mpl.colors.BoundaryNorm(bounds, mpl.colormaps["Oranges"].N)
        hydraulic_conductivity[~mask] = np.nan
        plt.imshow(hydraulic_conductivity/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal', norm=norm)
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