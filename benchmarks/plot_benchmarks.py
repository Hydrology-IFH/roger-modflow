from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 9
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["legend.title_fontsize"] = 8
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 9.0,
        "axes.labelsize": 12.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 8.0,
    },
)


base_path = Path(__file__).parent
base_path_figs = base_path / "figures"

file = base_path / "benchmark_times_modflow_steady-state.csv"
df_modflow_steadystate = pd.read_csv(file, sep=";")
df_modflow_steadystate["time"] = pd.to_datetime(df_modflow_steadystate["time"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_modflow_steadystate["time"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_modflow_steadystate["time"], format="%H:%M:%S.%f").dt.microsecond / 1e6
df_modflow_steadystate["time_per_step"] = pd.to_datetime(df_modflow_steadystate["time_per_step"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_modflow_steadystate["time_per_step"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_modflow_steadystate["time_per_step"], format="%H:%M:%S.%f").dt.microsecond / 1e6

file = base_path / "benchmark_times_modflow_transient.csv"
df_modflow = pd.read_csv(file, sep=";")
df_modflow["time"] = pd.to_datetime(df_modflow["time"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_modflow["time"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_modflow["time"], format="%H:%M:%S.%f").dt.microsecond / 1e6
df_modflow["time_per_step"] = pd.to_datetime(df_modflow["time_per_step"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_modflow["time_per_step"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_modflow["time_per_step"], format="%H:%M:%S.%f").dt.microsecond / 1e6

file = base_path / "benchmark_times_roger_numpy.csv"
df_roger_numpy = pd.read_csv(file, sep=";")
df_roger_numpy["time"] = pd.to_datetime(df_roger_numpy["time"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_numpy["time"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_numpy["time"], format="%H:%M:%S.%f").dt.microsecond / 1e6
df_roger_numpy["time_per_step"] = pd.to_datetime(df_roger_numpy["time_per_step"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_numpy["time_per_step"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_numpy["time_per_step"], format="%H:%M:%S.%f").dt.microsecond / 1e6

file = base_path / "benchmark_times_roger_jax.csv"
df_roger_jax = pd.read_csv(file, sep=";")
df_roger_jax["time"] = pd.to_datetime(df_roger_jax["time"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_jax["time"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_jax["time"], format="%H:%M:%S.%f").dt.microsecond / 1e6
df_roger_jax["time_per_step"] = pd.to_datetime(df_roger_jax["time_per_step"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_jax["time_per_step"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_jax["time_per_step"], format="%H:%M:%S.%f").dt.microsecond / 1e6

file = base_path / "benchmark_times_roger_numpy_modflow_transient.csv"
df_roger_numpy_modflow = pd.read_csv(file, sep=";")
df_roger_numpy_modflow["time"] = pd.to_datetime(df_roger_numpy_modflow["time"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_numpy_modflow["time"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_numpy_modflow["time"], format="%H:%M:%S.%f").dt.microsecond / 1e6
df_roger_numpy_modflow["time_per_step"] = pd.to_datetime(df_roger_numpy_modflow["time_per_step"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_numpy_modflow["time_per_step"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_numpy_modflow["time_per_step"], format="%H:%M:%S.%f").dt.microsecond / 1e6

file = base_path / "benchmark_times_roger_jax_modflow_transient.csv"
df_roger_jax_modflow = pd.read_csv(file, sep=";")
df_roger_jax_modflow["time"] = pd.to_datetime(df_roger_jax_modflow["time"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_jax_modflow["time"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_jax_modflow["time"], format="%H:%M:%S.%f").dt.microsecond / 1e6
df_roger_jax_modflow["time_per_step"] = pd.to_datetime(df_roger_jax_modflow["time_per_step"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_jax_modflow["time_per_step"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_jax_modflow["time_per_step"], format="%H:%M:%S.%f").dt.microsecond / 1e6

file = base_path / "benchmark_times_roger_numpy_aggregate_modflow_transient.csv"
df_roger_numpy_aggregate_modflow = pd.read_csv(file, sep=";")
df_roger_numpy_aggregate_modflow["time"] = pd.to_datetime(df_roger_numpy_aggregate_modflow["time"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_numpy_aggregate_modflow["time"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_numpy_aggregate_modflow["time"], format="%H:%M:%S.%f").dt.microsecond / 1e6
df_roger_numpy_aggregate_modflow["time_per_step"] = pd.to_datetime(df_roger_numpy_aggregate_modflow["time_per_step"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_numpy_aggregate_modflow["time_per_step"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_numpy_aggregate_modflow["time_per_step"], format="%H:%M:%S.%f").dt.microsecond / 1e6

file = base_path / "benchmark_times_roger_jax_aggregate_modflow_transient.csv"
df_roger_jax_aggregate_modflow = pd.read_csv(file, sep=";")
df_roger_jax_aggregate_modflow["time"] = pd.to_datetime(df_roger_jax_aggregate_modflow["time"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_jax_aggregate_modflow["time"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_jax_aggregate_modflow["time"], format="%H:%M:%S.%f").dt.microsecond / 1e6
df_roger_jax_aggregate_modflow["time_per_step"] = pd.to_datetime(df_roger_jax_aggregate_modflow["time_per_step"], format="%H:%M:%S.%f").dt.minute * 60 + pd.to_datetime(df_roger_jax_aggregate_modflow["time_per_step"], format="%H:%M:%S.%f").dt.second + pd.to_datetime(df_roger_jax_aggregate_modflow["time_per_step"], format="%H:%M:%S.%f").dt.microsecond / 1e6


fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=True)
axes[0].plot(df_roger_numpy["ncells"], df_roger_numpy["time_per_step"], label="RoGeR (numpy)", ls="--", color="#6baed6", lw=1)
axes[0].plot(df_roger_jax["ncells"], df_roger_jax["time_per_step"], label="RoGeR (jax)", ls="-", color="#6baed6", lw=1.5)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].legend(frameon=False, fontsize=9)
axes[2].set_ylabel('Computation time per daily time step [seconds]')
axes[1].plot(df_modflow_steadystate["ncells"], df_modflow_steadystate["time_per_step"], label="MODFLOW (steady-state)", marker="x", color="#bcbddc", lw=1)
axes[1].plot(df_modflow["ncells"], df_modflow["time_per_step"], label="MODFLOW (transient)", marker="o", color="#bcbddc", lw=1.5)
axes[1].set_ylabel('Computation time per daily time step [seconds]')
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].legend(frameon=False, fontsize=9)
axes[2].plot(df_roger_numpy_modflow["ncells"], df_roger_numpy_modflow["time_per_step"], label="RoGeR (numpy) - MODFLOW (transient)", ls="--", marker="o", color="#807dba", lw=1)
axes[2].plot(df_roger_jax_modflow["ncells"], df_roger_jax_modflow["time_per_step"], label="RoGeR (jax) - MODFLOW (transient)", ls="-", marker="o", color="#807dba", lw=1.5)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].legend(frameon=False, fontsize=9)
axes[3].plot(df_roger_numpy_aggregate_modflow["ncells"], df_roger_numpy_aggregate_modflow["time_per_step"], label="RoGeR (numpy) - aggregate - MODFLOW (transient)", ls="--", marker="o", color="#4a1486", lw=1)
axes[3].plot(df_roger_jax_aggregate_modflow["ncells"], df_roger_jax_aggregate_modflow["time_per_step"], label="RoGeR (jax) - aggregate - MODFLOW (transient)", ls="-", marker="o", color="#4a1486", lw=1.5)
axes[3].set_xlabel('Number of grid cells')
axes[3].set_xscale('log')
axes[3].set_yscale('log')
axes[3].legend(frameon=False, fontsize=9)
plt.tight_layout()
file = base_path_figs / "benchmarks_per_day.png"
fig.savefig(file, dpi=300)
plt.close("all")


fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=True)
axes[0].plot(df_roger_numpy["ncells"], df_roger_numpy["time"], label="RoGeR (numpy)", ls="--", color="#6baed6", lw=1)
axes[0].plot(df_roger_jax["ncells"], df_roger_jax["time"], label="RoGeR (jax)", ls="-", color="#6baed6", lw=1.5)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].legend(frameon=False, fontsize=9)
axes[2].set_ylabel('Computation time per 30 daily time steps [seconds]')
axes[1].plot(df_modflow_steadystate["ncells"], df_modflow_steadystate["time"], label="MODFLOW (steady-state)", marker="x", color="#bcbddc", lw=1)
axes[1].plot(df_modflow["ncells"], df_modflow["time"], label="MODFLOW (transient)", marker="o", color="#bcbddc", lw=1.5)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].legend(frameon=False, fontsize=9)
axes[2].plot(df_roger_numpy_modflow["ncells"], df_roger_numpy_modflow["time"], label="RoGeR (numpy) - MODFLOW (transient)", ls="--", marker="o", color="#807dba", lw=1)
axes[2].plot(df_roger_jax_modflow["ncells"], df_roger_jax_modflow["time"], label="RoGeR (jax) - MODFLOW (transient)", ls="-", marker="o", color="#807dba", lw=1.5)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].legend(frameon=False, fontsize=9)
axes[3].plot(df_roger_numpy_aggregate_modflow["ncells"], df_roger_numpy_aggregate_modflow["time"], label="RoGeR (numpy) - aggregate - MODFLOW (transient)", ls="--", marker="o", color="#4a1486", lw=1)
axes[3].plot(df_roger_jax_aggregate_modflow["ncells"], df_roger_jax_aggregate_modflow["time"], label="RoGeR (jax) - aggregate - MODFLOW (transient)", ls="-", marker="o", color="#4a1486", lw=1.5)
axes[3].set_xlabel('Number of grid cells')
axes[3].set_xscale('log')
axes[3].set_yscale('log')
axes[3].legend(frameon=False, fontsize=9)
plt.tight_layout()
file = base_path_figs / "benchmarks_per_30days.png"
fig.savefig(file, dpi=300)
plt.close("all")

fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=True)
axes[0].plot(df_roger_numpy["ncells"], df_roger_numpy["time"]/60, label="RoGeR (numpy)", ls="--", color="#6baed6", lw=1)
axes[0].plot(df_roger_jax["ncells"], df_roger_jax["time"]/60, label="RoGeR (jax)", ls="-", color="#6baed6", lw=1.5)
axes[0].set_xscale('log')
axes[0].legend(frameon=False, fontsize=9)
axes[1].plot(df_modflow_steadystate["ncells"], df_modflow_steadystate["time"]/60, label="MODFLOW (steady-state)", marker="x", color="#bcbddc", lw=1)
axes[1].plot(df_modflow["ncells"], df_modflow["time"]/60, label="MODFLOW (transient)", marker="o", color="#bcbddc", lw=1.5)
axes[2].set_ylabel('Computation time per 30 daily time steps [minutes]')
axes[1].set_xscale('log')
axes[1].legend(frameon=False, fontsize=9)
axes[2].plot(df_roger_numpy_modflow["ncells"], df_roger_numpy_modflow["time"]/60, label="RoGeR (numpy) - MODFLOW (transient)", ls="--", marker="o", color="#807dba", lw=1)
axes[2].plot(df_roger_jax_modflow["ncells"], df_roger_jax_modflow["time"]/60, label="RoGeR (jax) - MODFLOW (transient)", ls="-", marker="o", color="#807dba", lw=1.5)
axes[2].set_xscale('log')
axes[2].legend(frameon=False, fontsize=9)
axes[3].plot(df_roger_numpy_aggregate_modflow["ncells"], df_roger_numpy_aggregate_modflow["time"]/60, label="RoGeR (numpy) - aggregate - MODFLOW (transient)", ls="--", marker="o", color="#4a1486", lw=1)
axes[3].plot(df_roger_jax_aggregate_modflow["ncells"], df_roger_jax_aggregate_modflow["time"]/60, label="RoGeR (jax) - aggregate - MODFLOW (transient)", ls="-", marker="o", color="#4a1486", lw=1.5)
axes[3].set_xlabel('Number of grid cells')
axes[3].set_xscale('log')
axes[3].legend(frameon=False, fontsize=9)
plt.tight_layout()
file = base_path_figs / "benchmarks_per_30days_minutes.png"
fig.savefig(file, dpi=300)
plt.close("all")

fig, axes = plt.subplots(1, 1, figsize=(4, 2))
time1 = df_roger_jax_aggregate_modflow["time"]/60
time2 = df_roger_jax_modflow["time"]/60
axes.plot(df_roger_numpy["ncells"], np.abs(time1 - time2), ls="-", marker="o", color="black", lw=1)
axes.set_xscale('log')
axes.set_ylabel('[minutes]')
axes.set_xlabel('Number of grid cells')
fig.tight_layout()
file = base_path_figs / "pace_per_30days_minutes.png"
fig.savefig(file, dpi=300)
plt.close("all")

fig, axes = plt.subplots(1, 1, figsize=(4, 2))
time1 = df_roger_jax_aggregate_modflow["time"]/60
time2 = df_roger_jax_modflow["time"]/60
axes.plot(df_roger_numpy["ncells"], 1 / (time1 / time2), ls="-", marker="o", color="black", lw=1)
axes.set_ylabel('[-]')
axes.set_ylim(1, 2)
axes.set_xscale('log')
axes.set_xlabel('Number of grid cells')
fig.tight_layout()
file = base_path_figs / "pace_per_30days_relative.png"
fig.savefig(file, dpi=300)
plt.close("all")

# plot the spatial scales
rowscols = [(10, 10),
            (50, 50),
            (100, 100),
            (200, 200),
            (400, 400),
            (800, 800),
            (1000, 1000)]

# dataframe to store the number of cells and the time it took to run the model
df_time = pd.DataFrame(index=range(len(rowscols)), columns=['ncells', 'time', 'time_per_step', 'backend'])


# plot the spatial scales
nxny = rowscols[0]
nx = nxny[0]
ny = nxny[1]
dx = 25
dy = 25
grid_extent = (0, ny * dy, 0, nx * dx)
x = np.arange(0, (nx+1) * dx, dx)
y = np.arange(0, (ny+1) * dy, dy)

grid = np.zeros((nx, ny))
grid[:, :] = 1

fig, axes = plt.subplots(figsize=(3, 3))
axes.imshow(grid, extent=grid_extent, cmap='Greys_r', vmin=0, vmax=1.2, aspect='equal')
axes.set_yticks(y, minor=True)
axes.set_xticks(x, minor=True)
axes.grid(zorder=0, which='both', color='black', linestyle='-', linewidth=0.5)

axes.set_xlabel('Distance in x-direction [m]')
axes.set_ylabel('Distance in y-direction [m]')
fig.tight_layout()
file = base_path_figs / f"benchmark_grid_{nx}.png"
fig.savefig(file, dpi=300)
plt.close("all")

nxny = rowscols[1]
nx = nxny[0]
ny = nxny[1]
dx = 25
dy = 25
grid_extent = (0, ny * dy, 0, nx * dx)
x = np.arange(0, (nx+1) * dx, dx)
y = np.arange(0, (ny+1) * dy, dy)

grid = np.zeros((nx, ny))
grid[:, :] = 1

fig, axes = plt.subplots(figsize=(4, 4))
axes.imshow(grid, extent=grid_extent, cmap='Greys_r', vmin=0, vmax=1.2, aspect='equal')
axes.set_yticks(y, minor=True)
axes.set_xticks(x, minor=True)
axes.grid(zorder=0, which='both', color='black', linestyle='-', linewidth=0.5)

axes.set_xlabel('Distance in x-direction [m]')
axes.set_ylabel('Distance in y-direction [m]')
fig.tight_layout()
file = base_path_figs / f"benchmark_grid_{nx}.png"
fig.savefig(file, dpi=300)
plt.close("all")
