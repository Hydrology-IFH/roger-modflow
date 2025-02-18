from pathlib import Path
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd

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

# load the fudge parameters
path = base_path / "fudge_parameters_modflow.csv"
fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)
df_params_metrics = fudge_parameters.copy()
df_params_metrics["ME"] = np.nan
df_params_metrics["50E"] = np.nan
df_params_metrics["MAE"] = np.nan
df_params_metrics["50AE"] = np.nan
df_params_metrics["RBIAS"] = np.nan
df_params_metrics["r_rank"] = np.nan
df_params_metrics["r_lin"] = np.nan


# load observed groundwater heads (average values of the observation wells)
path = base_path / "observations" / "observed_groundwater_heads_avg.csv"
observed_groundwater_heads = pd.read_csv(path, sep=";", skiprows=0)

# load observed groundwater heads
rows = observed_groundwater_heads.iloc[:, -2].values  # row IDs of the observation wells
cols = observed_groundwater_heads.iloc[:, -3].values  # column IDs of the observation wells
obs = observed_groundwater_heads.iloc[:, -1].values # observed groundwater depths
obs_depth = topography[rows, cols].flatten() - observed_groundwater_heads.iloc[:, -1].values  # observed groundwater depths

for model_run in range(0, 4501):
    # load the netcdf file
    output_file = Path(f"/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen/steady-state/{base_path.name}/output") / f"modflow_output_run_{model_run}.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
    groundwater_heads = ds_mf["head"].values[0, ...]

    # extract the simulated groundwater depths at the location of the observation wells
    sim = groundwater_heads[1, rows, cols].flatten()
    sim_depth = topography[rows, cols].flatten() - groundwater_heads[1, rows, cols].flatten()

    # calculate mean error
    df_params_metrics.loc[model_run, "ME"] = np.mean(sim - obs)
    # calculate mean error
    df_params_metrics.loc[model_run, "50E"] = np.median(sim - obs)
    # calculate mean absolute error
    df_params_metrics.loc[model_run, "MAE"] = np.mean(np.abs(sim - obs))
    # calculate mean absolute error
    df_params_metrics.loc[model_run, "50AE"] = np.median(np.abs(sim - obs))
    # calculate relative bias
    df_params_metrics.loc[model_run, "RBIAS"] = np.mean((sim - obs) / obs)
    # calculate spearman correlation
    df_params_metrics.loc[model_run, "r_rank"] = sp.stats.spearmanr(sim_depth, obs_depth)[0]
    # calculate spearman correlation
    df_params_metrics.loc[model_run, "r_lin"] = sp.stats.pearsonr(sim_depth, obs_depth)[0]

# save the metrics
path = base_path / "fudge_parameters_metrics.csv"
df_params_metrics.to_csv(path, sep=";", index=False)