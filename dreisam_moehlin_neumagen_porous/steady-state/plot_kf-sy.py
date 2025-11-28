from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

base_path = Path(__file__).parent

# load MODFLOW parameters
path = base_path / "input" / "parameters_modflow.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")

hydraulic_conductivities_layer1 = ds_params['kf'].isel(layer=0).values
hydraulic_conductivities_layer2 = ds_params['kf'].isel(layer=1).values
hydraulic_conductivities_layer3 = ds_params['kf'].isel(layer=2).values
hydraulic_conductivities_layer4 = ds_params['kf'].isel(layer=3).values
hydraulic_conductivities_layers = np.array([hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4])
kf_ = np.unique(hydraulic_conductivities_layers)/86400

kf = 10**np.linspace(-8, -1, 1000)

specific_yield = 0.462 + 0.045 * np.log(kf)
cond = specific_yield < 0.05
specific_yield[cond] = 0.3 + 0.016 * np.log(kf[cond])
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel("$k_f$ [m/s]")
ax.set_ylabel("$sy$ [-]")
ax.scatter(kf, specific_yield, color="black")
specific_yield = 0.3 + 0.016 * np.log(kf)
ax.scatter(kf, specific_yield, color="red")
ax.set_xscale("log")
file = base_path / "figures" / "kf_sy_marotz.png"
fig.savefig(file, dpi=300)
plt.close(fig)

specific_yield = 0.3 + 0.016 * np.log(kf)
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel("$k_f$ [m/s]")
ax.set_ylabel("$sy$ [-]")
ax.scatter(kf, specific_yield, color="black")
ax.set_xscale("log")
file = base_path / "figures" / "kf_sy_marotz19.png"
fig.savefig(file, dpi=300)
plt.close(fig)

specific_yield = 0.4 + 0.05 * np.log10(kf)
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel("$k_f$ [m/s]")
ax.set_ylabel("$sy$ [-]")
ax.scatter(kf, specific_yield, color="black")
ax.set_xscale("log")
file = base_path / "figures" / "kf_sy_henning.png"
fig.savefig(file, dpi=300)
plt.close(fig)

specific_yield_ = 0.462 + 0.045 * np.log(kf_)
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel("$k_f$ [m/s]")
ax.set_ylabel("$sy$ [-]")
ax.scatter(kf_, specific_yield_, color="black")
ax.set_xscale("log")
file = base_path / "figures" / "kf_sy_marotz_dmn.png"
fig.savefig(file, dpi=300)
plt.close(fig)

specific_yield_ = 0.4 + 0.05 * np.log10(kf_)
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlabel("$k_f$ [m/s]")
ax.set_ylabel("$sy$ [-]")
ax.scatter(kf_, specific_yield_, color="black")
ax.set_xscale("log")
file = base_path / "figures" / "kf_sy_henning_dmn.png"
fig.savefig(file, dpi=300)
plt.close(fig)