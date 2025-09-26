from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import scipy as sp
import rasterio
import matplotlib.pyplot as plt
import click
import yaml

@click.option("-mr", "--model-run", type=int, default=5)
@click.command("main", short_help="Evaluate the steady-state simulation")
def main(model_run):
    base_path = Path(__file__).parent

    # load MODFLOW parameters
    path = Path(__file__).parent.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")

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

    grid_extent = (0, (777*50) / 1000, (621*50) / 1000, 0)
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

    dict_obs_stage_id = modflow_config["dict_obs_stage_rnos"]
    dict_obs_flow_id = modflow_config["dict_obs_flow_rnos"]

    dict_obs_stage_id_inv = {v: k for k, v in dict_obs_stage_id.items()}
    dict_obs_flow_id_inv = {v: k for k, v in dict_obs_flow_id.items()}

    # load the SFR reaches
    reaches = pd.read_csv(base_path.parent / 'input' / 'sfr_packagedata_modified.csv', sep=';')
    
    output_file = base_path / "output" / f"dmn_run_{model_run}_sfr.obs.csv"
    df_sfr_ = pd.read_csv(output_file, sep=",")

    df_sfr = pd.DataFrame(index=["falkensteig", "ebnet", "ehrenkirchen", "muenstertal"], columns=["rno", "layer", "x", "y", "rlen", "rwid", "rtp", "rgrd", "man", "rhk", "water_depth", "flow"])
    df_sfr["rno"] = [key for key in dict_obs_stage_id.values()]
    for rno in df_sfr["rno"].values:
        df_sfr.loc[df_sfr["rno"] == rno, "layer"] = reaches.loc[reaches["rno"] == rno, "k"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "y"] = reaches.loc[reaches["rno"] == rno, "i"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "x"] = reaches.loc[reaches["rno"] == rno, "j"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rlen"] = reaches.loc[reaches["rno"] == rno, "rlen"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rwid"] = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rtp"] = reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rgrd"] = reaches.loc[reaches["rno"] == rno, "rgrd"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rhk"] = reaches.loc[reaches["rno"] == rno, "rhk"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "man"] = reaches.loc[reaches["rno"] == rno, "man"].values[0]
        rwidth = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
        stage_depth = df_sfr_.loc[0, dict_obs_stage_id_inv[rno]] - reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "water_depth"] = stage_depth
        flow = (df_sfr_.loc[0, dict_obs_flow_id_inv[rno]] * (-1)) / 86400
        df_sfr.loc[df_sfr["rno"] == rno, "flow"] = flow * stage_depth * rwidth

    file = base_path / "output" / f"dmn_run_{model_run}_sfr.csv"
    df_sfr.to_csv(file, sep=";")

    sim_water_depth = df_sfr["water_depth"].values
    sim_water_depth[sim_water_depth < 0] = 0
    obs_water_depth = observed_streamflow["WDavg"].values
    diff_sim_obs_water_depth = sim_water_depth - obs_water_depth

    # load the netcdf file
    output_file = base_path / "output" / f"modflow_output_run_{model_run}.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
    groundwater_heads = ds_mf["head"].values[0, 1, ...]
    groundwater_heads[groundwater_heads > topography] = topography[groundwater_heads > topography]

    # extract the simulated groundwater heads at the location of the observation wells
    sim_depths = topography[rows, cols].flatten() - groundwater_heads[rows, cols].flatten()
    sim_depths = np.where(sim_depths < 0, 0, sim_depths)
    sim = groundwater_heads[rows, cols].flatten()

    interp_depths = topography[rows, cols].flatten() - gw_heads_interpolated[rows, cols].flatten()
    observed_groundwater_heads["sim-obs"] = sim_depths - obs_depths
    observed_groundwater_heads["sim-int"] = sim_depths - interp_depths
    observed_groundwater_heads["int-obs"] = interp_depths - obs_depths
    observed_groundwater_heads.to_csv(base_path.parent / "observations" / "observed_groundwater_heads_avg_.csv", sep=";", index=False)


    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    gauge_obs_y = observed_streamflow["y"].values * 50  # row IDs of the observation wells
    gauge_obs_x = observed_streamflow["x"].values * 50  # column IDs of the observation wells
    plt.scatter(gauge_obs_x/1000, gauge_obs_y/1000, c=diff_sim_obs_water_depth, s=5, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"difference_sim_obs_water_depth_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    # calculate mean error
    print(np.mean(sim - obs))
    # calculate mean absolute error
    print(np.mean(np.abs(sim - obs)))
    print(sp.stats.spearmanr(sim_depths, obs_depths)[0])

    diff_sim_obs = sim - obs
    cm = plt.get_cmap('PuOr')
    cm.set_bad(color='grey')
    grid_extent = (0, (777*50)/1000, (621*50)/1000, 0)
    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    # wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]) * 50
    # wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]) * 50
    # plt.scatter(wells_x, wells_y, marker='x', s=5, c='black')
    wells_obs_y = observed_groundwater_heads["Zelle_y"].values * 50  # row IDs of the observation wells
    wells_obs_x = observed_groundwater_heads["Zelle_x"].values * 50  # column IDs of the observation wells
    plt.scatter(wells_obs_x/1000, wells_obs_y/1000, c=diff_sim_obs, s=5, cmap=cm, vmin=-5, vmax=5)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [km]')
    plt.ylabel('Distance in y-direction [km]')
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"difference_sim_obs_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    grid_extent = (0, (777*50) / 1000, (621*50) / 1000, 0)
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

    grid_extent = (0, (777*50) / 1000, (621*50) / 1000, 0)
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

    grid_extent = (0, (777*50) / 1000, (621*50) / 1000, 0)
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
    axes.scatter(obs_water_depth, sim_water_depth, marker='.', s=8, c='black')
    axes.set_ylabel('Simulated water depth [m]')
    axes.set_xlabel('Observed water depth [m]')
    axes.set_xlim(-0.1, 1)
    axes.set_ylim(-0.1, 1)
    axes.plot(axes.get_xlim(), axes.get_ylim(), ls="--", c=".3", zorder=1, alpha=0.5)
    # axes.text(axes.get_xlim()[0] + 0.1, axes.get_ylim()[1] - 0.1, f"ME: {df_params_metrics.loc[model_run, 'ME']:.2f} m")
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"scatter_obs_sim_water_depth{model_run}.png"
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
    file = Path(__file__).parent / "figures" / f"scatter_obs_sim{model_run}_.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(obs_depths, sim_depths, marker='.', s=5, c='black')
    axes.set_ylabel('Simulated groundwater depth [m]')
    axes.set_xlabel('Observed groundwater depth [m]')
    axes.set_xlim(0, 30)
    axes.set_ylim(0, 30)
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


    for layer in range(4):
        fig, axes = plt.subplots(figsize=(4, 4))
        plt.imshow(ds_mf['head'].isel(Time=0, layer=layer).values, extent=grid_extent, cmap='viridis', aspect='equal')
        plt.colorbar(label='groundwater head \n[m a.s.l.]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = Path(__file__).parent / "figures" / f"gw_head_steady_state_layer{i}_grid_{model_run}.png"
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
        file = Path(__file__).parent / "figures" / f"gw_thickness_steady_state_layer{i}_grid_{model_run}.png"
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
        file = Path(__file__).parent / "figures" / f"gw_depth_steady_state_layer{i}_grid_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        flow_residuals = ds_mf['flow_residual'].isel(Time=0, layer=layer).values
        mask1 = (flow_residuals <= 10) & (flow_residuals >= -10)
        flow_residuals[mask1] = np.nan
        minmax = np.nanmax(np.abs(flow_residuals))
        plt.imshow(flow_residuals, extent=grid_extent, cmap='PuOr', aspect='equal', vmin=-minmax, vmax=minmax)
        plt.colorbar(label='groundwater flow residuals [m/day]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = Path(__file__).parent / "figures" / f"gw_flow_residuals_steady_state_layer{i}_grid_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        print(f"Layer {layer} flow residual (min, max): {np.nanmin(flow_residuals):.2f}, {np.nanmax(flow_residuals):.2f} m/day")

        fig, axes = plt.subplots(figsize=(4, 4))
        specific_discharge = ds_mf['specific_discharge'].isel(Time=0, layer=layer).values
        minmax = np.nanmax(np.abs(specific_discharge))
        plt.imshow(specific_discharge, extent=grid_extent, cmap='PuOr', aspect='equal', vmin=-minmax, vmax=minmax)
        plt.colorbar(label='groundwater specific discharge [m/day]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('Distance in x-direction [m]')
        plt.ylabel('Distance in y-direction [m]')
        plt.tight_layout()
        i = layer + 1
        file = Path(__file__).parent / "figures" / f"gw_specific_discharge_steady_state_layer{i}_grid_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

    return

if __name__ == "__main__":
    main()
