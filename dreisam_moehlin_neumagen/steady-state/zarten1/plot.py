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

    res_modflow = 50  # spatial resolution of MODFLOW in meters
    modflow_config = {
        'dx': res_modflow,
        'dy': res_modflow,
        'nx': 621,
        'ny': 777,
        'nz': 4,
    }

    dates = pd.date_range(start="2013-01-01", end="2022-12-31", freq="D")

    # load MODFLOW parameters
    path = Path(__file__).parent / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")

    # load topography
    topography = ds_params['elevations'].isel(z=0).values
    mask = np.isfinite(topography)
    grid_extent = (0, modflow_config['ny']*modflow_config['dy'], 0, modflow_config['nx']*modflow_config['dx'])

    basin = np.zeros((modflow_config['nx'], modflow_config['ny']))
    basin[mask] = 1

    inner_boundary = np.zeros((modflow_config['nx']+4, modflow_config['ny']+4))
    inner_boundary[2:-2, 2:-2] = basin
    boundary = np.zeros((modflow_config['nx']+4, modflow_config['ny']+4))

    boundary[1:-3, 2:-2] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[1:-3, 2:-2] < 0, 1, boundary[1:-3, 2:-2])
    boundary[3:-1, 2:-2] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[3:-1, 2:-2] < 0, 1, boundary[3:-1, 2:-2])
    boundary[2:-2, 1:-3] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[2:-2, 1:-3] < 0, 1, boundary[2:-2, 1:-3])
    boundary[2:-2, 3:-1] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[2:-2, 3:-1] < 0, 1, boundary[2:-2, 3:-1])
    boundary[1:-3, 1:-3] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[1:-3, 1:-3] < 0, 1, boundary[1:-3, 1:-3])
    boundary[1:-3, 3:-1] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[1:-3, 3:-1] < 0, 1, boundary[1:-3, 3:-1])
    boundary[3:-1, 1:-3] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[3:-1, 1:-3] < 0, 1, boundary[3:-1, 1:-3])
    boundary[1:-3, 3:-1] = np.where(inner_boundary[2:-2, 2:-2] - inner_boundary[1:-3, 3:-1] < 0, 1, boundary[1:-3, 3:-1])

    boundary = boundary[2:-2, 2:-2]
    topography[~mask] = np.nan
    # define location of boundary condition
    mask_boundary_condition = np.where((boundary == 1) & (topography <= 240), 1, np.nan)
    mask_boundary_condition[:, :163] = np.where((boundary == 1)[:, :163], 1, mask_boundary_condition[:, :163])
    mask_boundary_condition[50:200, :180] = np.where((boundary == 1)[50:200, :180], 1, mask_boundary_condition[50:200, :180])

    # # set Schoenberg to inactive
    # mask_schoenberg = (ds_params['mask_schoenberg'].values == 1)
    # mask = np.where(mask_schoenberg, False, mask)

    # load observed groundwater heads (average values of the observation wells)
    path = base_path / "observations" / "observed_groundwater_heads_avg.csv"
    observed_groundwater_heads = pd.read_csv(path, sep=";", skiprows=0)

    # load observed groundwater heads
    obs = observed_groundwater_heads.iloc[:, -1].values  # observed groundwater heads
    rows = observed_groundwater_heads.iloc[:, -2].values  # row IDs of the observation wells
    cols = observed_groundwater_heads.iloc[:, -3].values  # column IDs of the observation wells
    obs_depths = topography[rows, cols].flatten() - observed_groundwater_heads.iloc[:, -1].values  # observed groundwater heads


    # load the netcdf file
    output_file = base_path / "output" / f"modflow_output_run_{model_run}.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
    groundwater_heads = ds_mf["head"].values[0, 1, ...]
    groundwater_heads[groundwater_heads > topography] = topography[groundwater_heads > topography] - 1

    # extract the simulated groundwater heads at the location of the observation wells
    sim = groundwater_heads[rows, cols].flatten()
    sim_depths = topography[rows, cols].flatten() - sim

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
    axes.scatter(np.where((boundary == 1))[1]*50, np.where((boundary == 1))[0]*50, s=0.5, c='k', alpha=0.5)
    axes.scatter(np.where((mask_boundary_condition == 1))[1]*50, np.where((mask_boundary_condition == 1))[0]*50, s=0.5, c='Grey')
    # wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]) * 50
    # wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]) * 50
    # plt.scatter(wells_x, wells_y, marker='x', s=5, c='black')
    wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
    wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
    plt.scatter(wells_obs_x, wells_obs_y, c=diff_sim_obs, s=5, cmap=cm, vmin=-30, vmax=30)
    plt.colorbar(label='Bias [m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('Distanz in x-Richtung [m]')
    plt.ylabel('Distanz in y-Richtung [m]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / "difference_sim_obs.pdf"
    fig.savefig(file, dpi=300)
    file = Path(__file__).parent / "figures" / "difference_sim_obs.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 2.7))
    y = np.arange(0, modflow_config['nx']*modflow_config['dx'], modflow_config['dx'])
    x = np.arange(0, modflow_config['ny']*modflow_config['dy'], modflow_config['dy'])
    X, Y = np.meshgrid(x, y)
    Z = ds_mf['head'].isel(Time=0, layer=2).values
    levels = [200, 225, 250, 300, 400, 500]
    CS = axes.contour(X, Y, Z, levels, colors='black')
    axes.clabel(CS, inline=True, fontsize=7, colors='black')
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('Distanz in x-Richtung [m]')
    plt.ylabel('Distanz in y-Richtung [m]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / "gw_head_steady_state_contour.pdf"
    fig.savefig(file, dpi=300)
    file = Path(__file__).parent / "figures" / "gw_head_steady_state_contour.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    fig, axes = plt.subplots(figsize=(3.2, 3))
    axes.scatter(obs_depths, sim_depths, marker='.', s=8, c='black')
    axes.set_ylabel('Simulierter GW-Flurabstand\n [m]')
    axes.set_xlabel('[m]\n Gemessener GW-Flurabstand')
    axes.set_xlim(np.nanmin(sim_depths) - 1, np.nanmax(sim_depths) + 1)
    axes.set_ylim(np.nanmin(sim_depths) - 1, np.nanmax(sim_depths) + 1)
    axes.plot(axes.get_xlim(), axes.get_ylim(), ls="--", c=".3", zorder=1, alpha=0.5)
    mae = np.mean(np.abs(sim - obs))
    axes.text(.55, .9, f"MAE: {mae:.2f} m", transform=axes.transAxes)
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / "scatter_obs_sim.pdf"
    fig.savefig(file, dpi=300)
    file = Path(__file__).parent / "figures" / "scatter_obs_sim.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    src = rasterio.open(str(base_path / "input" / "recharge_roger_50m.tif"))
    recharge = src.read(1) / len(dates)
    recharge = np.where(mask, recharge, np.nan)
    grid_extent = (0, 777*modflow_config['dy'], 621*modflow_config['dx'], 0)
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(recharge, cmap='Blues', aspect='equal', extent=grid_extent)
    plt.colorbar(label='GWN [mm/Tag]', shrink=0.45)
    plt.grid(zorder=0)
    plt.xlabel('Distanz in x-Richtung [m]')
    plt.ylabel('Distanz in y-Richtung [m]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / "average_recharge.pdf"
    fig.savefig(file, dpi=300)
    file = Path(__file__).parent / "figures" / "average_recharge.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    return

if __name__ == "__main__":
    main()
