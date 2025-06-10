from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import scipy as sp
import rasterio
import matplotlib.pyplot as plt
import click

@click.option("-mr", "--model-run", type=int, default=0)
@click.command("main", short_help="Evaluate the steady-state simulation")
def main(model_run):
    base_path = Path(__file__).parent

    # load MODFLOW parameters
    path = Path(__file__).parent / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")

    # load the topography and elevation of the aquifer layers
    topography = ds_params['elevations'].isel(z=0).values
    # derive the model domain from the topography
    mask = np.isfinite(topography)
    # set Schoenberg to inactive
    mask_schoenberg = (ds_params['mask_schoenberg'].values == 1)
    mask = np.where(mask_schoenberg, False, mask)

    # load observed groundwater heads (average values of the observation wells)
    path = base_path / "observations" / "observed_groundwater_heads_avg.csv"
    observed_groundwater_heads = pd.read_csv(path, sep=";", skiprows=0)

    # load interpolated groundwater heads
    base_path = Path(__file__).parent
    src = rasterio.open(str(base_path.parent / "input" / "groundwater_heads_interpolated_50m.tif"))
    gw_heads_interpolated = src.read(1)

    grid_extent = (0, 777*50, 621*50, 0)
    fig, axes = plt.subplots(figsize=(4, 4))
    gw_heads_interpolated[~mask] = np.nan
    plt.imshow(gw_heads_interpolated, cmap='terrain', aspect='equal')
    plt.colorbar(label='[m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-direction')
    plt.ylabel('y-direction')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / "groundwater_heads_interpolated.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    gw_depth_interpolated = topography - gw_heads_interpolated
    plt.imshow(gw_depth_interpolated, cmap='viridis', aspect='equal', vmin=0, vmax=50)
    plt.colorbar(label='[m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-direction')
    plt.ylabel('y-direction')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / "groundwater_depths_interpolated.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    # load observed groundwater heads
    rows = observed_groundwater_heads.iloc[:, -2].values  # row IDs of the observation wells
    cols = observed_groundwater_heads.iloc[:, -3].values  # column IDs of the observation wells
    obs_depths = topography[rows, cols].flatten() - observed_groundwater_heads.iloc[:, -1].values  # observed groundwater depths
    obs = observed_groundwater_heads.iloc[:, -1].values


    # load the netcdf file
    output_file = base_path / "output" / f"modflow_output_run_{model_run}.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
    groundwater_heads = ds_mf["head"].values[0, 1, ...]
    groundwater_heads[groundwater_heads > topography] = topography[groundwater_heads > topography]

    # extract the simulated groundwater heads at the location of the observation wells
    sim_depths = topography[rows, cols].flatten() - groundwater_heads[rows, cols].flatten()
    sim = groundwater_heads[rows, cols].flatten()

    interp_heads = gw_heads_interpolated[rows, cols].flatten()
    interp_depths = topography[rows, cols].flatten() - gw_heads_interpolated[rows, cols].flatten()
    observed_groundwater_heads["sim-obs"] = sim - obs
    observed_groundwater_heads["sim-int"] = sim - interp_heads
    observed_groundwater_heads["int-obs"] = interp_heads - obs
    observed_groundwater_heads.to_csv(base_path / "observations" / "observed_groundwater_heads_avg_.csv", sep=";", index=False)

    # calculate mean error
    print(np.mean(sim - obs))
    # calculate mean absolute error
    print(np.mean(np.abs(sim - obs)))
    print(sp.stats.spearmanr(sim_depths, obs_depths)[0])

    diff_sim_obs = sim - obs
    cm = plt.get_cmap('PuOr')
    grid_extent = (0, 777*50, 621*50, 0)
    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    # wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]) * 50
    # wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]) * 50
    # plt.scatter(wells_x, wells_y, marker='x', s=5, c='black')
    wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
    wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
    plt.scatter(wells_obs_x, wells_obs_y, c=diff_sim_obs, s=5, cmap=cm, vmin=-5, vmax=5)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / f"difference_sim_obs_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    grid_extent = (0, 777*50, 621*50, 0)
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(groundwater_heads[:, :] - gw_heads_interpolated, cmap='PuOr', aspect='equal', vmin=-10, vmax=10, extent=grid_extent)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / f"difference_sim_{model_run}_int.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    grid_extent = (0, 777*50, 621*50, 0)
    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]) * 50
    wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]) * 50
    plt.scatter(wells_x, wells_y, marker='x', s=5, c='black')
    wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
    wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
    plt.scatter(wells_obs_x, wells_obs_y, c=sim, s=5, cmap="viridis", vmin=150, vmax=400)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / f"groundwater_heads_sim{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    grid_extent = (0, 777*50, 621*50, 0)
    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]) * 50
    wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]) * 50
    plt.scatter(wells_x, wells_y, marker='x', s=5, c='black')
    wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
    wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
    plt.scatter(wells_obs_x, wells_obs_y, c=obs, s=5, cmap="viridis", vmin=150, vmax=400)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / "observed_groundwater_heads.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(obs_depths, sim_depths, marker='.', s=5, c='black')
    axes.set_ylabel('Simulated groundwater depth [m]')
    axes.set_xlabel('Observed groundwater depth [m]')
    axes.set_xlim(np.nanmin(sim_depths) - 1, np.nanmax(sim_depths) + 1)
    axes.set_ylim(np.nanmin(sim_depths) - 1, np.nanmax(sim_depths) + 1)
    axes.plot(axes.get_xlim(), axes.get_ylim(), ls="--", c=".3", zorder=1, alpha=0.5)
    # axes.text(axes.get_xlim()[0] + 0.1, axes.get_ylim()[1] - 0.1, f"ME: {df_params_metrics.loc[model_run, 'ME']:.2f} m")
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"scatter_obs_sim{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(range(len(diff_sim_obs)), diff_sim_obs, marker='.', s=5, c='black')
    axes.set_ylabel('Bias [m]')
    axes.set_xlabel('# Observation well')
    axes.axhline(y=0, color='grey', linestyle='--', zorder=1, alpha=0.5)
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"scatter_diff_obs_sim{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)
    return

if __name__ == "__main__":
    main()
