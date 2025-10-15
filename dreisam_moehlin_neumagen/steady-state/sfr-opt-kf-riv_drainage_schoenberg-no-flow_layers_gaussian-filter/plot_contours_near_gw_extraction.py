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

    file_config = base_path.parent / "config.yml"
    with open(file_config, "r") as file:
        modflow_config = yaml.safe_load(file)

    # load groundwater extraction data
    path = base_path.parent / "input" / "groundwater_extraction.csv"
    groundwater_extraction = pd.read_csv(path, sep=";", skiprows=0)
    wells_y = groundwater_extraction["y-coordinate"].values / 1000
    wells_x = groundwater_extraction["x-coordinate"].values / 1000

    # load MODFLOW parameters
    path = Path(__file__).parent.parent / "input" / "parameters_modflow.nc"
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

    # contours of groundwater heads (Cerdia wells)
    # x1 = 286
    # x2 = 379
    # y1 = 39
    # y2 = 136 

    x1 = 334
    x2 = 376
    y1 = 86
    y2 = 136 

    x1 = 334
    x2 = 334 + 25
    y1 = 136 - 25
    y2 = 136

    grid_extent = (ds_mf.lon.values[x1] / 1000, ds_mf.lon.values[x2] / 1000, ds_mf.lat.values[y2] / 1000, ds_mf.lat.values[y1] / 1000)

    levels = [1, 2, 3, 4, 5, 10, 12, 14, 16, 18]

    for layer in range(4):
        fig, axes = plt.subplots(figsize=(4, 4))
        y = ds_mf.lat.values[y1:y2] / 1000
        x = ds_mf.lon.values[x1:x2] / 1000
        X, Y = np.meshgrid(x, y)
        Z = ds_mf['head'].isel(Time=0, layer=layer).values[y1:y2, x1:x2]
        CS = axes.contour(X, Y, Z, colors='black')
        axes.clabel(CS, inline=True, fontsize=8, colors='black')
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='green')
        axes.imshow(topography[y1:y2, x1:x2], extent=grid_extent, cmap='terrain', alpha=0.25, vmin=190, vmax=450)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.title(f"Groundwater head of layer {layer + 1} [m a.s.l.]", fontsize=8)
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_head_steady_state_layer{i}_contour_{model_run}_cerdia.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        y = ds_mf.lat.values[y1:y2] / 1000
        x = ds_mf.lon.values[x1:x2] / 1000
        X, Y = np.meshgrid(x, y)
        gw_depth = topography[y1:y2, x1:x2] - ds_mf['head'].isel(Time=0, layer=layer).values[y1:y2, x1:x2]
        gw_depth[gw_depth < 0] = 0
        CS = axes.contour(X, Y, gw_depth, colors='black')
        axes.clabel(CS, inline=True, fontsize=8, colors='black')
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='green')
        axes.imshow(gw_depth, extent=grid_extent, cmap='viridis', alpha=1.0, vmin=0, vmax=20)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.title(f"Groundwater depth of layer {layer + 1} [m a.s.l.]", fontsize=8)
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_depth_steady_state_layer{i}_contour_{model_run}_cerdia.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    # contours of groundwater heads (Ebnet BN wells)
    x1 = 435
    x2 = 499
    y1 = 206
    y2 = 233 

    grid_extent = (ds_mf.lon.values[x1] / 1000, ds_mf.lon.values[x2] / 1000, ds_mf.lat.values[y2] / 1000, ds_mf.lat.values[y1] / 1000)

    for layer in range(4):
        fig, axes = plt.subplots(figsize=(4, 4))
        y = ds_mf.lat.values[y1:y2] / 1000
        x = ds_mf.lon.values[x1:x2] / 1000
        X, Y = np.meshgrid(x, y)
        Z = ds_mf['head'].isel(Time=0, layer=layer).values[y1:y2, x1:x2]
        CS = axes.contour(X, Y, Z, colors='black')
        axes.clabel(CS, inline=True, fontsize=8, colors='black')
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='green')
        axes.imshow(topography[y1:y2, x1:x2], extent=grid_extent, cmap='terrain', alpha=0.25, vmin=190, vmax=450)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.title(f"Groundwater head of layer {layer + 1} [m a.s.l.]", fontsize=8)
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_head_steady_state_layer{i}_contour_{model_run}_bn-ebnet.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        y = ds_mf.lat.values[y1:y2] / 1000
        x = ds_mf.lon.values[x1:x2] / 1000
        X, Y = np.meshgrid(x, y)
        gw_depth = topography[y1:y2, x1:x2] - ds_mf['head'].isel(Time=0, layer=layer).values[y1:y2, x1:x2]
        gw_depth[gw_depth < 0] = 0
        CS = axes.contour(X, Y, gw_depth, colors='black')
        axes.clabel(CS, inline=True, fontsize=8, colors='black')
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='green')
        axes.imshow(gw_depth, extent=grid_extent, cmap='viridis', alpha=1.0, vmin=0, vmax=20)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.title(f"Groundwater depth of layer {layer + 1} [m a.s.l.]", fontsize=8)
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_depth_steady_state_layer{i}_contour_{model_run}_bn-ebnet.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    # contours of groundwater heads (Hausen BN wells)
    x1 = 50
    x2 = 90
    y1 = 238
    y2 = 288

    grid_extent = (ds_mf.lon.values[x1] / 1000, ds_mf.lon.values[x2] / 1000, ds_mf.lat.values[y2] / 1000, ds_mf.lat.values[y1] / 1000)

    for layer in range(4):
        fig, axes = plt.subplots(figsize=(4, 4))
        y = ds_mf.lat.values[y1:y2] / 1000
        x = ds_mf.lon.values[x1:x2] / 1000
        X, Y = np.meshgrid(x, y)
        Z = ds_mf['head'].isel(Time=0, layer=layer).values[y1:y2, x1:x2]
        CS = axes.contour(X, Y, Z, colors='black')
        axes.clabel(CS, inline=True, fontsize=8, colors='black')
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='green')
        axes.imshow(topography[y1:y2, x1:x2], extent=grid_extent, cmap='terrain', alpha=0.25, vmin=190, vmax=450)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.title(f"Groundwater depth of layer {layer + 1} [m a.s.l.]", fontsize=8)
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_head_steady_state_layer{i}_contour_{model_run}_bn-hausen.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        y = ds_mf.lat.values[y1:y2] / 1000
        x = ds_mf.lon.values[x1:x2] / 1000
        X, Y = np.meshgrid(x, y)
        gw_depth = topography[y1:y2, x1:x2] - ds_mf['head'].isel(Time=0, layer=layer).values[y1:y2, x1:x2]
        gw_depth[gw_depth < 0] = 0
        CS = axes.contour(X, Y, gw_depth, colors='black')
        axes.clabel(CS, inline=True, fontsize=8, colors='black')
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='green')
        axes.imshow(gw_depth, extent=grid_extent, cmap='viridis', alpha=1.0, vmin=0, vmax=20)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.title(f"Groundwater depth of layer {layer + 1} [m a.s.l.]", fontsize=8)
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_depth_steady_state_layer{i}_contour_{model_run}_bn-hausen.png"
        fig.savefig(file, dpi=300)
        plt.close("all")
    return


if __name__ == "__main__":
    main()