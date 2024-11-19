import os
from pathlib import Path
import sys
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd


base_path = Path(__file__).parent

# load the fudge parameters
path = base_path / "fudge_parameters_modflow.csv"
fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)
df_params_metrics = fudge_parameters.copy()
df_params_metrics["ME"] = np.nan
df_params_metrics["MAE"] = np.nan
df_params_metrics["RBIAS"] = np.nan
df_params_metrics["r"] = np.nan

# load observed groundwater heads

obs = np.array([185, 185, 185, 185, 185, 185, 200, 200, 200, 200, 200, 200, 200])  # observed groundwater heads
xx = [66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]   # row IDs of the observation wells
yy = [266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]   # column IDs of the observation wells

for model_run in range(0, 1):
    complete = df_params_metrics.loc[model_run, "complete"]
    # skip if steady-state simulation did not converged
    if complete == 1:
        # load the netcdf file
        output_file = base_path / "output" / "steady-state" / f"modflow_output_run_{model_run}.nc"
        ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
        groundwater_heads = ds_mf["head"].values[0, ...]

        # extract the simulated groundwater heads at the location of the observation wells
        sim = groundwater_heads[1, xx, yy].flatten()

        # calculate mean error
        df_params_metrics.loc[model_run, "ME"] = np.nanmean(sim - obs)
        # calculate mean absolute error
        df_params_metrics.loc[model_run, "MAE"] = np.nanmean(np.abs(sim - obs))
        # calculate relative bias
        df_params_metrics.loc[model_run, "RBIAS"] = np.nanmean((sim - obs) / obs)
        # calculate spearman correlation
        df_params_metrics.loc[model_run, "r"] = sp.stats.spearmanr(sim, obs)[0]


# save the metrics
path = base_path / "fudge_parameters_metrics.csv"
df_params_metrics.to_csv(path, sep=";", index=False)
