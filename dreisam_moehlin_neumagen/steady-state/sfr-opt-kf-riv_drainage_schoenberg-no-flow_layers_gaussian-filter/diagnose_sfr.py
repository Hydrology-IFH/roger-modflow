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

    # load observed groundwater heads
    rows = observed_groundwater_heads.iloc[:, -2].values  # row IDs of the observation wells
    cols = observed_groundwater_heads.iloc[:, -3].values  # column IDs of the observation wells
    obs_depths = topography[rows, cols].flatten() - observed_groundwater_heads.iloc[:, -1].values  # observed groundwater depths
    obs = observed_groundwater_heads.iloc[:, -1].values

    dict_obs_stage_id = modflow_config["dict_obs_stage_rnos"]
    dict_obs_flow_id = modflow_config["dict_obs_flow_rnos"]
    sfr_obs_points = [key.split('_')[0] for key in dict_obs_stage_id.keys()]

    dict_obs_stage_id_inv = {v: k for k, v in dict_obs_stage_id.items()}
    dict_obs_flow_id_inv = {v: k for k, v in dict_obs_flow_id.items()}

    # load the SFR reaches
    reaches = pd.read_csv(base_path.parent / 'input' / 'sfr_packagedata_modified.csv', sep=';')
    
    # load the simulated groundwater heads
    output_file = base_path / "output" / f"modflow_output_run_{model_run}.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
    groundwater_heads = ds_mf["head"].values[0, 1, ...]
    gw_sw = np.nanmean(ds_mf['gw_sw'].isel(Time=0).values * (-1), axis=0) / 86400

    # load the SFR output file
    output_file = base_path / "output" / f"dmn_run_{model_run}_sfr.obs.csv"
    df_sfr_ = pd.read_csv(output_file, sep=",")

    rwid = np.nan * np.ones(topography.shape)

    df_sfr = pd.DataFrame(index=sfr_obs_points, columns=["rno", "layer", "x", "y", "rlen", "rwid", "rtp", "rgrd", "man", "rhk", "water_head", "water_depth", "flow", "kf", "topo", "gw_head", "gw-sw", "sw-gw_flux"])
    df_sfr["rno"] = [key for key in dict_obs_stage_id.values()]
    df_sfr["rno"] = df_sfr["rno"] + 1
    for rno in df_sfr["rno"].values:
        z = reaches.loc[reaches["rno"] == rno, "k"].values[0] - 1
        y = reaches.loc[reaches["rno"] == rno, "i"].values[0] - 1
        x = reaches.loc[reaches["rno"] == rno, "j"].values[0] - 1
        kf = ds_params["kf"].isel(layer=1).values[y, x] / 86400
        df_sfr.loc[df_sfr["rno"] == rno, "layer"] = reaches.loc[reaches["rno"] == rno, "k"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "y"] = reaches.loc[reaches["rno"] == rno, "i"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "x"] = reaches.loc[reaches["rno"] == rno, "j"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rlen"] = reaches.loc[reaches["rno"] == rno, "rlen"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rwid"] = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rtp"] = reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rgrd"] = reaches.loc[reaches["rno"] == rno, "rgrd"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rhk"] = reaches.loc[reaches["rno"] == rno, "rhk"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "man"] = reaches.loc[reaches["rno"] == rno, "man"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "kf"] = kf
        rwidth = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
        stage_depth = df_sfr_.loc[0, dict_obs_stage_id_inv[rno-1]] - reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "water_head"] = df_sfr_.loc[0, dict_obs_stage_id_inv[rno-1]]
        df_sfr.loc[df_sfr["rno"] == rno, "water_depth"] = stage_depth
        flow = (df_sfr_.loc[0, dict_obs_flow_id_inv[rno-1]] * (-1)) / 86400
        df_sfr.loc[df_sfr["rno"] == rno, "flow"] = flow * stage_depth * rwidth
        df_sfr.loc[df_sfr["rno"] == rno, "gw_head"] = ds_mf["head"].values[0, z, y, x]
        df_sfr.loc[df_sfr["rno"] == rno, "gw-sw"] = ds_mf["head"].values[0, z, y, x] - df_sfr_.loc[0, dict_obs_stage_id_inv[rno-1]]
        df_sfr.loc[df_sfr["rno"] == rno, "sw-gw_flux"] = gw_sw[y, x] 
        df_sfr.loc[df_sfr["rno"] == rno, "topo"] = topography[y, x] 

    file = base_path / "output" / f"dmn_run_{model_run}_sfr_.csv"
    df_sfr.to_csv(file, sep=";")


    return

if __name__ == "__main__":
    main()
