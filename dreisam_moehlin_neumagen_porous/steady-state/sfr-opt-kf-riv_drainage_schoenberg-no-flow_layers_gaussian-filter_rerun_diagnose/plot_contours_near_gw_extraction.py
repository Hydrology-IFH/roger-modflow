import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
import pandas as pd
import geopandas as gpd
import rasterio
import click
import matplotlib as mpl
import contextily as ctx
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar

@click.option("-mr", "--model-run", type=int, default=8304)
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
    path = base_path.parent / "input" / "groundwater_extraction_bn_cerdia.csv"
    groundwater_extraction = pd.read_csv(path, sep=";", skiprows=0)
    wells_y = groundwater_extraction["y-coordinate"].values
    wells_x = groundwater_extraction["x-coordinate"].values

    # load MODFLOW parameters
    path = Path(__file__).parent.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")

    # load water protection areas
    path = Path(__file__).parent.parent / "input" / "wsg_hausen.tif"
    src_wsg_hausen = rasterio.open(str(path))
    wsg_hausen = src_wsg_hausen.read(1)

    path = Path(__file__).parent.parent / "input" / "mask_wsg_hausen.tif"
    src_mask_wsg_hausen = rasterio.open(str(path))
    mask_wsg_hausen = src_mask_wsg_hausen.read(1)

    path = Path(__file__).parent.parent / "input" / "wsg_zartener_becken.tif"
    src_wsg_zarten = rasterio.open(str(path))
    wsg_zarten = src_wsg_zarten.read(1)

    path = Path(__file__).parent.parent / "input" / "mask_wsg_zartener_becken.tif"
    src_mask_wsg_zarten = rasterio.open(str(path))
    mask_wsg_zarten = src_mask_wsg_zarten.read(1)

    gdf_wsg_zarten = gpd.read_file(base_path.parent / "input" / "wsg_zartener_becken.gpkg")
    gdf_wsg_hausen = gpd.read_file(base_path.parent / "input" / "wsg_hausen.gpkg")

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

    # # contours of groundwater heads (Cerdia wells)
    # x1 = 334
    # x2 = 376
    # y1 = 86
    # y2 = 136 

    # grid_extent = (ds_mf.lon.values[x1], ds_mf.lon.values[x2], ds_mf.lat.values[y2], ds_mf.lat.values[y1])

    # levels_depth = [1, 2, 3, 4, 5, 10, 12, 14, 16, 18]

    # fig, axes = plt.subplots(figsize=(4, 4))
    # y = ds_mf.lat.values[y1:y2]
    # x = ds_mf.lon.values[x1:x2]
    # X, Y = np.meshgrid(x, y)
    # Z = topography[y1:y2, x1:x2]
    # CS = axes.contour(X, Y, Z, colors='black')
    # axes.clabel(CS, inline=True, fontsize=8, colors='black')
    # axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
    # cb = axes.imshow(Z, extent=grid_extent, cmap='Blues', alpha=1)
    # plt.colorbar(cb, label='Topography [m a.s.l.]', shrink=0.72)
    # plt.xlabel('x-coordinate')
    # plt.ylabel('y-coordinate')
    # plt.tight_layout()
    # file = base_path_figs / f"topography_cerdia.png"
    # fig.savefig(file, dpi=300)
    # plt.close("all")

    # for layer in range(4):
    #     fig, axes = plt.subplots(figsize=(4, 4))
    #     y = ds_mf.lat.values[y1:y2]
    #     x = ds_mf.lon.values[x1:x2]
    #     X, Y = np.meshgrid(x, y)
    #     Z = ds_mf['head'].isel(Time=0, layer=layer).values[y1:y2, x1:x2]
    #     zmin = int(np.nanmin(Z))
    #     zmax = int(np.nanmax(Z))
    #     levels_head = list(range(zmin, zmax + 1, 1))
    #     CS = axes.contour(X, Y, Z, levels_head, colors='black')
    #     axes.clabel(CS, inline=True, fontsize=8, colors='black')
    #     axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
    #     # axes.imshow(topography[y1:y2, x1:x2], extent=grid_extent, cmap='terrain', alpha=0.25, vmin=190, vmax=450)
    #     cb = axes.imshow(Z, extent=grid_extent, cmap='Blues', alpha=0.75)
    #     plt.colorbar(cb, label='Groundwater head [m a.s.l.]', shrink=0.72)
    #     plt.xlabel('x-coordinate')
    #     plt.ylabel('y-coordinate')
    #     plt.tight_layout()
    #     i = layer + 1
    #     file = base_path_figs / f"gw_head_steady_state_layer{i}_contour_{model_run}_cerdia.png"
    #     fig.savefig(file, dpi=300)
    #     plt.close("all")

    #     fig, axes = plt.subplots(figsize=(4, 4))
    #     y = ds_mf.lat.values[y1:y2]
    #     x = ds_mf.lon.values[x1:x2]
    #     X, Y = np.meshgrid(x, y)
    #     gw_depth = topography[y1:y2, x1:x2] - ds_mf['head'].isel(Time=0, layer=layer).values[y1:y2, x1:x2]
    #     gw_depth[gw_depth < 0] = 0
    #     CS = axes.contour(X, Y, gw_depth, colors='black')
    #     axes.clabel(CS, inline=True, fontsize=8, colors='black')
    #     axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
    #     cb = axes.imshow(gw_depth, extent=grid_extent, cmap='viridis', alpha=1.0, vmin=0, vmax=20)
    #     plt.colorbar(cb, label='Groundwater depth [m]', shrink=0.75)
    #     plt.xlabel('x-coordinate')
    #     plt.ylabel('y-coordinate')
    #     plt.tight_layout()
    #     i = layer + 1
    #     file = base_path_figs / f"gw_depth_steady_state_layer{i}_contour_{model_run}_cerdia.png"
    #     fig.savefig(file, dpi=300)
    #     plt.close("all")

    # contours of groundwater heads (Ebnet BN wells)
    height = wsg_zarten.shape[0]
    width = wsg_zarten.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src_wsg_zarten.transform, rows, cols)
    lons = np.array(xs)
    lats = np.array(ys)

    grid_extent = (lons[0], lons[-1], lats[0], lats[-1])

    fig, axes = plt.subplots(figsize=(6, 4))
    cond_y = (ds_mf.lat.values <= lats[0]) & (ds_mf.lat.values >= lats[-1])
    cond_x = (ds_mf.lon.values >= lons[0]) & (ds_mf.lon.values <= lons[-1])
    y = ds_mf.lat.values[cond_y]
    x = ds_mf.lon.values[cond_x]
    x1 = np.where(cond_x)[0][0]
    x2 = np.where(cond_x)[0][-1] + 1
    y1 = np.where(cond_y)[0][0]
    y2 = np.where(cond_y)[0][-1] + 1
    X, Y = np.meshgrid(x, y)
    Z = topography[y1:y2, x1:x2]
    # CS = axes.contour(X, Y, Z, colors='black')
    # axes.clabel(CS, inline=True, fontsize=8, colors='black')
    axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
    axes.set_xlim(grid_extent[0], grid_extent[1])
    axes.set_ylim(grid_extent[3], grid_extent[2])
    cb = axes.imshow(Z, extent=grid_extent, cmap='Blues', alpha=1)
    plt.colorbar(cb, label='Topography [m a.s.l.]', shrink=0.5)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.tight_layout()
    file = base_path_figs / f"topography_bn-ebnet.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    for layer in range(4):
        fig, axes = plt.subplots(figsize=(6, 4))
        cond_y = (ds_mf.lat.values <= lats[0]) & (ds_mf.lat.values >= lats[-1])
        cond_x = (ds_mf.lon.values >= lons[0]) & (ds_mf.lon.values <= lons[-1])
        y = ds_mf.lat.values[cond_y]
        x = ds_mf.lon.values[cond_x]
        x1 = np.where(cond_x)[0][0]
        x2 = np.where(cond_x)[0][-1] + 1
        y1 = np.where(cond_y)[0][0]
        y2 = np.where(cond_y)[0][-1] + 1
        X, Y = np.meshgrid(x, y)
        Z = ds_mf['head'].isel(Time=0, layer=layer).values
        cond_nan = mask_wsg_zarten != 1
        Z[cond_nan] = np.nan
        Z = Z[y1:y2, x1:x2]
        zmin = int(np.nanmin(Z))
        zmax = int(np.nanmax(Z))
        levels_head = [310, 320, 325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385, 390, 395, 400, 405, 410, 415, 420]
        CS = axes.contour(X, Y, Z, levels_head, colors='blue')
        axes.clabel(CS, inline=True, fontsize=8, colors='blue')
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
        axes.set_xlim(grid_extent[0], grid_extent[1])
        axes.set_ylim(grid_extent[3], grid_extent[2])
        gdf_wsg_zarten.boundary.plot(ax=axes, edgecolor='black', linewidth=1)
        ctx.add_basemap(axes, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:25832")
        north_arrow(
            axes, scale=0.2, location="upper right", rotation={"crs": 25832, "reference": "center"}
        )
        scale_bar(axes, location="lower right", style="boxes", bar={"projection": 25832, "minor_type": "none"}, labels={"style": "first_last"})
        axes.text(
        0.145,
        0.11,
        "EPSG: 25832",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes.transAxes,
        )
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_head_steady_state_layer{i}_contour_{model_run}_bn-ebnet.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(6, 4))
        cond_y = (ds_mf.lat.values <= lats[0]) & (ds_mf.lat.values >= lats[-1])
        cond_x = (ds_mf.lon.values >= lons[0]) & (ds_mf.lon.values <= lons[-1])
        y = ds_mf.lat.values[cond_y]
        x = ds_mf.lon.values[cond_x]
        x1 = np.where(cond_x)[0][0]
        x2 = np.where(cond_x)[0][-1] + 1
        y1 = np.where(cond_y)[0][0]
        y2 = np.where(cond_y)[0][-1] + 1
        X, Y = np.meshgrid(x, y)
        Z = ds_mf['head'].isel(Time=0, layer=layer).values
        cond_nan = mask_wsg_zarten != 1
        Z[cond_nan] = np.nan
        gw_depth = topography[y1:y2, x1:x2] - Z[y1:y2, x1:x2]
        gw_depth[gw_depth < 0] = 0
        CS = axes.contour(X, Y, gw_depth, colors='black')
        axes.clabel(CS, inline=True, fontsize=8, colors='black')
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
        axes.set_xlim(grid_extent[0], grid_extent[1])
        axes.set_ylim(grid_extent[3], grid_extent[2])
        ctx.add_basemap(axes, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:25832")
        north_arrow(
            axes, scale=0.2, location="upper right", rotation={"crs": 25832, "reference": "center"}
        )
        scale_bar(axes, location="lower right", style="boxes", bar={"projection": 25832, "minor_type": "none"}, labels={"style": "first_last"})
        axes.text(
        0.16,
        0.1,
        "EPSG: 25832",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes.transAxes,
        )
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_depth_steady_state_layer{i}_contour_{model_run}_bn-ebnet.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    # contours of groundwater heads (Hausen BN wells)
    width = wsg_hausen.shape[1]
    height = wsg_hausen.shape[0]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src_wsg_hausen.transform, rows, cols)
    lons = np.array(xs)
    lats = np.array(ys)

    grid_extent = (lons[0], lons[-1], lats[0], lats[-1])
    
    fig, axes = plt.subplots(figsize=(6, 6))
    cond_y = (ds_mf.lat.values <= lats[0]) & (ds_mf.lat.values >= lats[-1])
    cond_x = (ds_mf.lon.values >= lons[0]) & (ds_mf.lon.values <= lons[-1])
    y = ds_mf.lat.values[cond_y]
    x = ds_mf.lon.values[cond_x]
    x1 = np.where(cond_x)[0][0]
    x2 = np.where(cond_x)[0][-1] + 1
    y1 = np.where(cond_y)[0][0]
    y2 = np.where(cond_y)[0][-1] + 1
    X, Y = np.meshgrid(x, y)
    Z = topography[y1:y2, x1:x2]
    CS = axes.contour(X, Y, Z, colors='black')
    axes.clabel(CS, inline=True, fontsize=8, colors='black')
    axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
    axes.set_xlim(grid_extent[0], grid_extent[1])
    axes.set_ylim(grid_extent[3], grid_extent[2])
    ctx.add_basemap(axes, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:25832")
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.tight_layout()
    file = base_path_figs / f"topography_bn-hausen.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    for layer in range(4):
        fig, axes = plt.subplots(figsize=(6, 6))
        cond_y = (ds_mf.lat.values <= lats[0]) & (ds_mf.lat.values >= lats[-1])
        cond_x = (ds_mf.lon.values >= lons[0]) & (ds_mf.lon.values <= lons[-1])
        y = ds_mf.lat.values[cond_y]
        x = ds_mf.lon.values[cond_x]
        x1 = np.where(cond_x)[0][0]
        x2 = np.where(cond_x)[0][-1] + 1
        y1 = np.where(cond_y)[0][0]
        y2 = np.where(cond_y)[0][-1] + 1
        X, Y = np.meshgrid(x, y)
        Z = ds_mf['head'].isel(Time=0, layer=layer).values
        cond_nan = mask_wsg_hausen != 1
        Z[cond_nan] = np.nan
        Z = Z[y1:y2, x1:x2]
        zmin = int(np.nanmin(Z))
        zmax = int(np.nanmax(Z))
        levels_head = [195, 196, 197, 199, 201, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250]
        CS = axes.contour(X, Y, Z, levels_head, colors='blue')
        axes.clabel(CS, inline=True, fontsize=8, colors='blue')
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
        axes.set_xlim(grid_extent[0], grid_extent[1])
        axes.set_ylim(grid_extent[3], grid_extent[2])
        gdf_wsg_hausen.boundary.plot(ax=axes, edgecolor='black', linewidth=1)
        ctx.add_basemap(axes, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:25832")
        north_arrow(
            axes, scale=0.2, location="upper right", rotation={"crs": 25832, "reference": "center"}
        )
        scale_bar(axes, location="lower left", style="boxes", bar={"projection": 25832, "minor_type": "none"}, labels={"style": "first_last"})
        axes.text(
        0.88,
        0.025,
        "EPSG: 25832",
        fontsize=12,
        horizontalalignment="center",
        verticalalignment="center",
        transform=axes.transAxes,
        )
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_head_steady_state_layer{i}_contour_{model_run}_bn-hausen.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(6, 6))
        cond_y = (ds_mf.lat.values <= lats[0]) & (ds_mf.lat.values >= lats[-1])
        cond_x = (ds_mf.lon.values >= lons[0]) & (ds_mf.lon.values <= lons[-1])
        y = ds_mf.lat.values[cond_y]
        x = ds_mf.lon.values[cond_x]
        x1 = np.where(cond_x)[0][0]
        x2 = np.where(cond_x)[0][-1] + 1
        y1 = np.where(cond_y)[0][0]
        y2 = np.where(cond_y)[0][-1] + 1
        X, Y = np.meshgrid(x, y)
        Z = ds_mf['head'].isel(Time=0, layer=layer).values
        cond_nan = mask_wsg_zarten != 1
        Z[cond_nan] = np.nan
        gw_depth = topography[y1:y2, x1:x2] - Z[y1:y2, x1:x2]
        gw_depth[gw_depth < 0] = 0
        CS = axes.contour(X, Y, gw_depth, colors='black')
        axes.clabel(CS, inline=True, fontsize=8, colors='black')
        axes.scatter(wells_x, wells_y, marker='^', s=10, c='magenta')
        axes.set_xlim(grid_extent[0], grid_extent[1])
        axes.set_ylim(grid_extent[3], grid_extent[2])
        ctx.add_basemap(axes, source=ctx.providers.OpenStreetMap.Mapnik, crs="EPSG:25832")
        plt.xlabel('x-coordinate')
        plt.ylabel('y-coordinate')
        plt.tight_layout()
        i = layer + 1
        file = base_path_figs / f"gw_depth_steady_state_layer{i}_contour_{model_run}_bn-hausen.png"
        fig.savefig(file, dpi=300)
        plt.close("all")
    return


if __name__ == "__main__":
    main()