from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import scipy as sp
import rasterio
import matplotlib.pyplot as plt
import click
import yaml

@click.option("-mr", "--model-run", type=int, default=9491)
@click.command("main", short_help="Evaluate the steady-state simulation")
def main(model_run):
    base_path = Path(__file__).parent
    base_path_figs = base_path / "figures"

    # load MODFLOW parameters
    path = Path(__file__).parent.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    grid_extent = (ds_params.x.values[0] / 1000, ds_params.x.values[-1] / 1000, ds_params.y.values[0] / 1000, ds_params.y.values[-1] / 1000)

    file_config = base_path.parent / "config.yml"
    with open(file_config, "r") as file:
        modflow_config = yaml.safe_load(file)

    # load the topography and elevation of the aquifer layers
    topography = ds_params['elevations'].isel(z=0).values
    # derive the model domain from the topography
    mask = np.isfinite(topography)
    # set Schoenberg to inactive
    mask_schoenberg = (ds_params['mask_schoenberg'].values == 1)
    mask = np.where(mask_schoenberg, False, mask)

    elevation_bottom_layer1 = ds_params['elevations'].isel(z=1).values
    elevation_bottom_layer2 = ds_params['elevations'].isel(z=2).values
    elevation_bottom_layer3 = ds_params['elevations'].isel(z=3).values
    elevation_bottom_layer4 = ds_params['elevations'].isel(z=4).values
    elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

    # load observed groundwater heads (average values of the observation wells)
    path = base_path.parent / "observations" / "observed_groundwater_heads_avg.csv"
    observed_groundwater_heads = pd.read_csv(path, sep=";", skiprows=0)
    observed_groundwater_heads = observed_groundwater_heads.iloc[:-2, :]

    # load observed streamflow
    path = base_path.parent / "observations" / "observed_streamflow.csv"
    observed_streamflow = pd.read_csv(path, sep=";", skiprows=0, index_col=0)

    # load interpolated groundwater heads
    base_path = Path(__file__).parent
    src = rasterio.open(str(base_path.parent / "input" / "groundwater_heads_interpolated_50m.tif"))
    gw_heads_interpolated = src.read(1)

    # load observed groundwater heads
    rows = observed_groundwater_heads.iloc[:, -2].values  # row IDs of the observation wells
    cols = observed_groundwater_heads.iloc[:, -3].values  # column IDs of the observation wells
    obs_depths = topography[rows, cols].flatten() - observed_groundwater_heads.iloc[:, -1].values  # observed groundwater depths
    obs = observed_groundwater_heads.iloc[:, -1].values

    # load the SFR reaches
    reaches = pd.read_csv(base_path.parent / 'input' / 'sfr_packagedata_modified.csv', sep=';')
    
    # load the simulated groundwater heads
    output_file = base_path / "output" / f"modflow_output_run_{model_run}.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
    groundwater_heads = ds_mf["head"].values[0, 1, ...]
    gw_sw = np.nansum(ds_mf['gw_sw'].isel(Time=0).values, axis=0) / 86400

    rwid = ds_mf["sfr_width"].values
    man = ds_mf["sfr_manning_coefficient"].values
    rhk = ds_mf["sfr_hydraulic_conductivity"].values
    rgrd = ds_mf["sfr_gradient"].values
    sfr_head = ds_mf["sfr_head"].values
    sfr_stage = ds_mf["sfr_stage"].values
    sfr_flow = ds_mf["sfr_flow"].values

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(rwid, extent=grid_extent, cmap='Blues', aspect='equal', vmin=1, vmax=15)
    plt.colorbar(label='reach width [m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = base_path_figs / "reach_width.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(rgrd, extent=grid_extent, cmap='Oranges', aspect='equal', vmin=0.01, vmax=0.45)
    plt.colorbar(label='reach gradient [-]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = base_path_figs / "reach_gradient.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(rhk, extent=grid_extent, cmap='Oranges', aspect='equal', norm='log')
    plt.colorbar(label='reach hydraulic conductivity [m/s]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = base_path_figs / f"reach_hydraulic_conductivity_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(man, extent=grid_extent, cmap='viridis', aspect='equal')
    plt.colorbar(label='Manning\'s n', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = base_path_figs / f"reach_manning_coefficient_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close("all")


    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(sfr_head, extent=grid_extent, cmap='viridis', aspect='equal')
    plt.colorbar(label='surface water head [m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = base_path_figs / f"sfr_head_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(sfr_stage, extent=grid_extent, cmap='viridis_r', aspect='equal', vmin=0, vmax=0.6)
    plt.colorbar(label='surface water depth [m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = base_path_figs / f"sfr_stage_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(sfr_flow, extent=grid_extent, cmap='viridis_r', aspect='equal', vmin=0, vmax=6)
    plt.colorbar(label='streamflow [$m^3/s$]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = base_path_figs / f"sfr_flow_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close("all")
    return

if __name__ == "__main__":
    main()
