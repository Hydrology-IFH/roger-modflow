from pathlib import Path
from cftime import num2date
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import seaborn as sns
import matplotlib as mpl

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["legend.title_fontsize"] = 8
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.0,
    },
)


base_path = Path(__file__).parent

# load the config file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    config = yaml.safe_load(file)

# load the measured groundwater heads
gw_stations = ['37', '83', '84', '87']
file = base_path / "observations" / "WaterLevel.PH_037_-_Badenova@EX_Ehrenkirchen.EntireRecord.csv"
df_heads_raw = pd.read_csv(file, delimiter=',', skiprows=14, index_col=0, parse_dates=True)
mask_na = (df_heads_raw['Grade'] == -2)
df_heads_raw.loc[mask_na, 'Value'] = np.nan
df_heads = pd.DataFrame(index=df_heads_raw.iloc[:, 0], data=df_heads_raw.iloc[:, 1].values, columns=['head'])
df_heads.index = pd.to_datetime(df_heads.index)
df_heads.index = pd.DatetimeIndex(df_heads.index).normalize()
idx_new = pd.date_range(start='1/1/1989', end='31/12/2023')

base_path_figs = base_path / "figures"
fig, axes = plt.subplots(figsize=(4, 2))
df_heads = df_heads.dropna()
axes.plot(df_heads.index, df_heads['head'], ls='-', lw=1, color='black', marker='o', markersize=2)
head_avg = np.nanmean(df_heads['head'].values)
axes.hlines(head_avg, df_heads.index[0], df_heads.index[-1], ls='--', color='grey', lw=1)
x = np.arange(len(df_heads.index))
trend = np.polyfit(x, df_heads['head'].values, 1)
p = np.poly1d(trend)
axes.plot(df_heads.index, p(x), ls='-', color='grey', lw=1.5)
axes.set_xlabel('Time [days]')
axes.set_ylabel('Groundwater head \n[m a.s.l.]')
axes.set_xlim(df_heads.index[0], df_heads.index[-1])
fig.tight_layout()
path = base_path_figs / "observed groundwater_head_PH_037.png"
fig.savefig(path, dpi=300)

df_heads = pd.DataFrame(index=idx_new).join(df_heads)

model_type = "transient"
base_path_figs = base_path / "figures" / model_type

# load the netcdf file
output_file = base_path / "output" / model_type / "modflow_output.nc"
ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
ndays = ds_mf.sizes['time']

output_file = base_path / "output" / model_type / "roger_output.nc"
ds_roger = xr.open_dataset(output_file, engine="h5netcdf")
days = ds_roger["Time"].values / np.timedelta64(24 * 60 * 60, "s")
date = num2date(
    days,
    units=f"days since {ds_roger['Time'].attrs['time_origin']}",
    calendar="standard",
    only_use_cftime_datetimes=False,
)
dates = date[1:]

# plot the heads for all time steps
for layer in range(4):
    vals = ds_mf['head'].isel(layer=layer).values
    vals_avg = np.array([np.nanmean(vals[t, :, :]) for t in range(ndays)])
    vals_p50 = np.array([np.nanmedian(vals[t, :, :]) for t in range(ndays)])
    vals_p5 = np.array([np.nanpercentile(vals[t, :, :], 5) for t in range(ndays)])
    vals_p95 = np.array([np.nanpercentile(vals[t, :, :], 95) for t in range(ndays)])
    vals_min = np.array([np.nanmin(vals[t, :, :]) for t in range(ndays)])
    vals_max = np.array([np.nanmax(vals[t, :, :]) for t in range(ndays)])

    fig, axes = plt.subplots(figsize=(4, 2))
    axes.plot(dates, vals_avg, ls='--', label='average', lw=1.5, color='black')
    axes.plot(dates, vals_p50, ls='-', label='median', lw=2, color='black')
    axes.plot(dates, vals_min, ls='-.', label='min/max', alpha=0.5, color='black', lw=1.0)
    axes.plot(dates, vals_max, ls='-.', alpha=0.5, color='black', lw=1.0)
    axes.set_xlabel('Time [days]')
    axes.set_ylabel('Groundwater head \n[m a.s.l.]')
    axes.set_xlim(dates[0], dates[-1])
    axes.legend(frameon=False, fontsize=9)
    axes.tick_params(axis="x", rotation=33)
    fig.tight_layout()
    path = base_path_figs / f"heads_time_series_layer{layer}.png"
    fig.savefig(path, dpi=300)

    vals_single_cell = ds_mf['head'].isel(layer=layer).values[:, 82, 2]
    fig, axes = plt.subplots(figsize=(4, 2))
    axes.plot(dates, vals_single_cell, ls='-', lw=1, color='black')
    axes.set_xlabel('Time [days]')
    axes.set_ylabel('Groundwater head \n[m a.s.l.]')
    axes.set_xlim(dates[0], dates[-1])
    axes.tick_params(axis="x", rotation=33)
    fig.tight_layout()
    path = base_path_figs / f"heads_time_series_single_cell_layer{layer}.png"
    fig.savefig(path, dpi=300)

# compare simulated time series with observed time series
idx = pd.date_range(start='1/11/2019', end='31/10/2022')
df_heads = pd.DataFrame(index=idx).join(df_heads)
for layer in range(4):
    fig, axes = plt.subplots(figsize=(4, 2))
    vals_single_cell = ds_mf['head'].isel(layer=layer).values[:, 82, 0]
    axes.plot(dates, vals_single_cell, ls='-', lw=2, color='red', alpha=1)
    vals_single_cell = ds_mf['head'].isel(layer=layer).values[:, 82, 1]
    axes.plot(dates, vals_single_cell, ls='-', lw=1.8, color='red', alpha=.9)
    vals_single_cell = ds_mf['head'].isel(layer=layer).values[:, 82, 2]
    axes.plot(dates, vals_single_cell, ls='-', lw=1.6, color='red', alpha=.8)
    vals_single_cell = ds_mf['head'].isel(layer=layer).values[:, 82, 3]
    axes.plot(dates, vals_single_cell, ls='-', lw=1.4, color='red', alpha=.7)
    vals_single_cell = ds_mf['head'].isel(layer=layer).values[:, 82, 4]
    axes.plot(dates, vals_single_cell, ls='-', lw=1.2, color='red', alpha=.6)
    axes.plot(df_heads.index, df_heads['head'], ls='-', lw=1, color='blue', marker='o', markersize=1.5)
    axes.set_xlabel('Time [days]')
    axes.set_ylabel('Groundwater head \n[m a.s.l.]')
    axes.set_xlim(dates[0], dates[-1])
    axes.tick_params(axis="x", rotation=33)
    fig.tight_layout()
    path = base_path_figs / f"heads_sim_obs_single_cell_layer{layer}.png"
    fig.savefig(path, dpi=300)

    fig, axes = plt.subplots(figsize=(4, 2))
    vals_single_cell = ds_mf['head'].isel(layer=layer).values[:, 82, 2]
    vals_single_cell_avg = np.mean(ds_mf['head'].isel(layer=layer).values[:, 82, 0])
    vals_sim = vals_single_cell - vals_single_cell_avg
    axes.plot(dates, vals_sim, ls='-', lw=2, color='red', alpha=1)
    vals_obs = df_heads['head'].values - np.nanmean(df_heads['head'].values)
    axes.plot(df_heads.index, vals_obs, ls='-', lw=1, color='blue', marker='o', markersize=1.5)
    axes.set_xlabel('Time [days]')
    axes.set_ylabel('[m]')
    axes.set_xlim(dates[0], dates[-1])
    axes.tick_params(axis="x", rotation=33)
    fig.tight_layout()
    path = base_path_figs / f"heads_sim_obs_single_cell_layer{layer}.png"
    fig.savefig(path, dpi=300)


# plot recharge time series
vals = ds_roger['q_ss'].values[:, :, 1:]
vals[vals < 0] = np.nan
vals_avg = np.array([np.nanmean(vals[:, :, t]) for t in range(ndays)])
vals_p50 = np.array([np.nanmedian(vals[:, :, t]) for t in range(ndays)])
vals_p5 = np.array([np.nanpercentile(vals[:, :, t], 5) for t in range(ndays)])
vals_p95 = np.array([np.nanpercentile(vals[:, :, t], 95) for t in range(ndays)])
vals_min = np.array([np.nanmin(vals[:, :, t]) for t in range(ndays)])
vals_max = np.array([np.nanmax(vals[:, :, t]) for t in range(ndays)])

fig, axes = plt.subplots(figsize=(5, 2))
axes.plot(dates, vals_avg, ls='--', label='average', lw=1.5, color='black')
axes.plot(dates, vals_p50, ls='-', label='median', lw=2, color='black')
# axes.plot(dates, vals_min, ls='-.', label='min/max', alpha=0.5, color='black', lw=1.0)
# axes.plot(dates, vals_max, ls='-.', alpha=0.5, color='black', lw=1.0)
axes.plot(dates, vals_p5, ls='-.', label='5%/95%-percentile', alpha=0.5, color='black', lw=1.0)
axes.plot(dates, vals_p95, ls='-.', alpha=0.5, color='black', lw=1.0)
axes.set_xlabel('Time [days]')
axes.set_ylabel('Groundwater recharge \n[mm/day]')
axes.set_xlim(dates[0], dates[-1])
axes.set_ylim(0, 40)
axes.tick_params(axis="x", rotation=33)
axes.legend(frameon=False, fontsize=9, ncol=3)
fig.tight_layout()
path = base_path_figs / "recharge_time_series.png"
fig.savefig(path, dpi=300)

fig, axes = plt.subplots(figsize=(5, 2))
axes.plot(dates, vals_avg, ls='--', label='average', lw=1, color='black')
axes.plot(dates, vals_p50, ls='-', label='median', lw=1.5, color='black')
axes.set_ylim(0, 20)
axes.set_xlabel('Time [days]')
axes.set_ylabel('Groundwater recharge \n[mm/day]')
axes.set_xlim(dates[0], dates[-1])
axes.tick_params(axis="x", rotation=33)
axes.legend(frameon=False, fontsize=9, ncol=3)
fig.tight_layout()
path = base_path_figs / "recharge_time_series_meadian_avg.png"
fig.savefig(path, dpi=300)

vals = ds_roger['z_gw'].values[:, :, 1:]
vals[vals < 0] = np.nan
vals_avg = np.array([np.nanmean(vals[:, :, t]) for t in range(ndays)])
vals_p50 = np.array([np.nanmedian(vals[:, :, t]) for t in range(ndays)])
vals_p5 = np.array([np.nanpercentile(vals[:, :, t], 5) for t in range(ndays)])
vals_p95 = np.array([np.nanpercentile(vals[:, :, t], 95) for t in range(ndays)])
vals_min = np.array([np.nanmin(vals[:, :, t]) for t in range(ndays)])
vals_max = np.array([np.nanmax(vals[:, :, t]) for t in range(ndays)])

fig, axes = plt.subplots(figsize=(4, 2))
axes.plot(dates, vals_avg, ls='--', label='average', lw=1.5, color='black')
axes.plot(dates, vals_p50, ls='-', label='median', lw=2, color='black')
axes.plot(dates, vals_min, ls='-.', label='min/max', alpha=0.5, color='black', lw=1.0)
axes.plot(dates, vals_max, ls='-.', alpha=0.5, color='black', lw=1.0)
axes.tick_params(axis="x", rotation=33)
axes.set_xlabel('Time [days]')
axes.set_ylabel('Groundwater depth \n[m]')
axes.set_xlim(dates[0], dates[-1])
axes.legend(frameon=False, fontsize=9, ncol=3)
fig.tight_layout()
path = base_path_figs / "gw_depth_time_series.png"
fig.savefig(path, dpi=300)

# load the netcdf file
model_type = "steady-state"
output_file = base_path / "output" / model_type / "modflow_output.nc"
ds_mf = xr.open_dataset(output_file, engine="h5netcdf")

df_heads_steady_state = pd.DataFrame(index=range(5), columns=['head_outlet', 'head_avg'])
df_heads_steady_state.iloc[0, 0] = np.nanmean(df_heads['head'].values)
for layer in range(4):
    vals = ds_mf['head'].isel(layer=layer).values[0, :, :]
    vals_single_cell = ds_mf['head'].isel(layer=layer).values[0, 82, 2]
    i = layer + 1
    df_heads_steady_state.iloc[i, 0] = vals_single_cell
    df_heads_steady_state.iloc[i, 1] = np.nanmean(vals)

file = output_file = base_path / "figures" / model_type / "heads_steady_state.csv"
df_heads_steady_state.to_csv(file, sep=";", header=True)

