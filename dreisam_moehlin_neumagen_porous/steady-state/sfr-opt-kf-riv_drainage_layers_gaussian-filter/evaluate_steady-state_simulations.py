from pathlib import Path
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import os
import yaml

obs_stations_sfr = ["EBNET_STAGE", "OBERAMBRINGEN_STAGE", "WIESNECK_STAGE"]
obs_flow_stage = ["EBNET", "OBERAMBRINGEN", "WIESNECK"]

base_path = Path(__file__).parent

# load MODFLOW parameters
path = base_path.parent / "input" / "parameters_modflow.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")

file_config = base_path.parent / "config.yml"
with open(file_config, "r") as file:
    modflow_config = yaml.safe_load(file)

# load the topography and elevation of the aquifer layers
topography = ds_params['elevations'].isel(z=0).values
mask = (ds_params["mask_porous_aquifer"].values == 1)

# load the fudge parameters
path = Path(f"/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/steady-state/{base_path.name}/output") / "fudge_parameters_modflow.csv"
fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)
df_params_metrics = fudge_parameters.copy()
df_params_metrics["ME"] = np.nan
df_params_metrics["50E"] = np.nan
df_params_metrics["MAE"] = np.nan
df_params_metrics["50AE"] = np.nan
df_params_metrics["RBIAS"] = np.nan
df_params_metrics["RABIAS"] = np.nan
df_params_metrics["r_rank"] = np.nan
df_params_metrics["r_lin"] = np.nan
df_params_metrics["n_5m"] = np.nan
df_params_metrics["n_3m"] = np.nan
df_params_metrics["n_1m"] = np.nan


# load observed groundwater heads (average values of the observation wells)
path = base_path.parent / "observations" / "observed_groundwater_heads_avg.csv"
observed_groundwater_heads = pd.read_csv(path, sep=";", skiprows=0)
observed_groundwater_heads = observed_groundwater_heads.iloc[:-2, :]
n_obs = len(observed_groundwater_heads.index)

# load observed streamflow
path = base_path.parent / "observations" / "observed_streamflow.csv"
observed_streamflow = pd.read_csv(path, sep=";", skiprows=0, index_col=0)

dict_obs_stage_id = modflow_config["dict_obs_stage_rnos"]
dict_obs_flow_id = modflow_config["dict_obs_flow_rnos"]
dict_obs_stage_id_inv = {v: k for k, v in dict_obs_stage_id.items()}
dict_obs_flow_id_inv = {v: k for k, v in dict_obs_flow_id.items()}

# load the SFR reaches
reaches = pd.read_csv(base_path.parent / 'input' / 'sfr_packagedata_modified.csv', sep=';')
reaches["rno"] = reaches["rno"] - 1
reaches["k"] = reaches["k"] - 1
reaches["i"] = reaches["i"] - 1
reaches["j"] = reaches["j"] - 1

# load observed groundwater heads
rows = observed_groundwater_heads.iloc[:, -2].values  # row IDs of the observation wells
cols = observed_groundwater_heads.iloc[:, -3].values  # column IDs of the observation wells
obs = observed_groundwater_heads.iloc[:, -1].values # observed groundwater depths
obs_depth = topography[rows, cols].flatten() - observed_groundwater_heads.iloc[:, -1].values  # observed groundwater depths

for model_run in range(0, 10000):
    output_file = Path(f"/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/steady-state/{base_path.name}/output") / f"modflow_output_run_{model_run}.nc"
    if os.path.exists(output_file):
        # load the netcdf file
        ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
        groundwater_heads = ds_mf["head"].values[0, ...]

        # extract the simulated groundwater depths at the location of the observation wells
        sim = groundwater_heads[1, rows, cols].flatten()
        sim_depth = topography[rows, cols].flatten() - groundwater_heads[1, rows, cols].flatten()
        sim_depth = np.where(sim_depth < 0, 0, sim_depth)

        # calculate mean error
        df_params_metrics.loc[model_run, "ME"] = np.mean(sim - obs)
        # calculate mean error
        df_params_metrics.loc[model_run, "50E"] = np.median(sim - obs)
        # calculate mean absolute error
        df_params_metrics.loc[model_run, "MAE"] = np.mean(np.abs(sim - obs))
        # calculate mean absolute error
        df_params_metrics.loc[model_run, "50AE"] = np.median(np.abs(sim - obs))
        # calculate relative bias
        df_params_metrics.loc[model_run, "RBIAS"] = np.mean((sim_depth - obs_depth) / obs_depth)
        df_params_metrics.loc[model_run, "RABIAS"] = np.mean(np.abs((sim_depth - obs_depth) / obs_depth))
        # calculate spearman correlation
        df_params_metrics.loc[model_run, "r_rank"] = sp.stats.spearmanr(sim_depth, obs_depth)[0]
        # calculate spearman correlation
        df_params_metrics.loc[model_run, "r_lin"] = sp.stats.pearsonr(sim_depth, obs_depth)[0]
        # calculate number of cells with a bias less than 5m
        df_params_metrics.loc[model_run, "n_5m"] = np.sum(np.abs(sim - obs) <= 5) / n_obs
        # calculate number of cells with a bias less than 3m
        df_params_metrics.loc[model_run, "n_3m"] = np.sum(np.abs(sim - obs) <= 3) / n_obs
        # calculate number of cells with a bias less than 1m
        df_params_metrics.loc[model_run, "n_1m"] = np.sum(np.abs(sim - obs) <= 1) / n_obs
        # calculate number of cells with a bias less than 0.7m
        df_params_metrics.loc[model_run, "n_07m"] = np.sum(np.abs(sim - obs) <= 0.7) / n_obs

        # calculate SFR metrics
        output_file = Path(f"/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/steady-state/{base_path.name}/output") / f"dmn_run_{model_run}_sfr.obs.csv"
        df_sfr_ = pd.read_csv(output_file, sep=",")

        idx = [obs_station_sfr.split('_')[0] for obs_station_sfr in obs_stations_sfr]
        df_sfr = pd.DataFrame(index=idx, columns=["rno", "layer", "x", "y", "rlen", "rwid", "rtp" "rgrd", "man", "rhk", "water_depth", "flow"])
        df_sfr["rno"] = [dict_obs_stage_id[obs_station_sfr] for obs_station_sfr in obs_stations_sfr]
        for rno in df_sfr["rno"].values:
            df_sfr.loc[df_sfr["rno"] == rno, "layer"] = reaches.loc[reaches["rno"] == rno, "k"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "x"] = reaches.loc[reaches["rno"] == rno, "i"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "y"] = reaches.loc[reaches["rno"] == rno, "j"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rlen"] = reaches.loc[reaches["rno"] == rno, "rlen"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rwid"] = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rtp"] = reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rgrd"] = reaches.loc[reaches["rno"] == rno, "rgrd"].values[0]
            rwidth = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
            stage_depth = df_sfr_.loc[0, dict_obs_stage_id_inv[rno]] - reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "water_depth"] = stage_depth
            flow = (df_sfr_.loc[0, dict_obs_flow_id_inv[rno]] * (-1)) / 86400
            if stage_depth < 0:
                flow = 0
                stage_depth = 0
            df_sfr.loc[df_sfr["rno"] == rno, "flow"] = flow

        df_sfr = df_sfr.loc[obs_flow_stage, :]
        sim_water_depth = df_sfr["water_depth"].values.astype(float)
        obs_water_depth = observed_streamflow["WDavg"].values
        obs_streamflow = observed_streamflow["Qavg"].values
        sim_streamflow = df_sfr["flow"].values.astype(float)

        df_params_metrics.loc[model_run, "MAE_sfr"] = np.nanmean(np.abs(sim_streamflow - obs_streamflow))
        df_params_metrics.loc[model_run, "ME_sfr"] = np.nanmean(sim_streamflow - obs_streamflow)
        df_params_metrics.loc[model_run, "RBIAS_sfr"] = np.nanmean((sim_streamflow - obs_streamflow) / obs_streamflow)
        df_params_metrics.loc[model_run, "RABIAS_sfr"] = np.nanmean(np.abs((sim_streamflow - obs_streamflow) / obs_streamflow))
        df_params_metrics.loc[model_run, "r_rank_sfr"] = sp.stats.spearmanr(sim_streamflow, obs_streamflow)[0]
        df_params_metrics.loc[model_run, "r_lin_sfr"] = sp.stats.pearsonr(sim_streamflow, obs_streamflow)[0]
        df_params_metrics.loc[model_run, "r_rank_multi"] = (39/42) * df_params_metrics.loc[model_run, "r_rank"] + (3/42) * df_params_metrics.loc[model_run, "r_rank_sfr"]
        df_params_metrics.loc[model_run, "MAE_multi"] = (39/42) * df_params_metrics.loc[model_run, "MAE"] + (3/42) * df_params_metrics.loc[model_run, "MAE_sfr"]

# save the metrics
path = base_path / "fudge_parameters_metrics_porous.csv"
df_params_metrics.index.name = 'model_run'
df_params_metrics.to_csv(path, sep=";", index=True)