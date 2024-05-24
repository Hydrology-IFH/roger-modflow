import os
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from flopy.utils import Raster
import yaml

# run installed version of flopy or add local path
try:
    import flopy
except:
    fpth = os.path.abspath(os.path.join("..", ".."))
    sys.path.append(fpth)
    import flopy

base_path = Path(__file__).parent
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# load the config file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    config = yaml.safe_load(file)

# load topography and domain
file = base_path / "input" / "elevation.grd"
layer_elevations = Raster.load(file)
topography = layer_elevations.get_array(1)
elevation_bottom_layer1 = layer_elevations.get_array(2)
topography[topography <= elevation_bottom_layer1] = elevation_bottom_layer1[topography <= elevation_bottom_layer1]
elevation_bottom_layer2 = layer_elevations.get_array(3)
elevation_bottom_layer3 = layer_elevations.get_array(4)
elevation_bottom_layer4 = layer_elevations.get_array(5)
elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

file = base_path / "input" / "domain.grd"
domain = Raster.load(file)
mask = (domain.get_array(1) == 1)
grid_extent = (0, config['ny']*config['dy'], 0, config['nx']*config['dx'])

basin = np.empty_like(domain.get_array(1))
basin[mask] = 1
fig, axes = plt.subplots(figsize=(4, 4))
plt.imshow(basin, extent=grid_extent, cmap='Greys', aspect='equal')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "basin.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
plt.imshow(topography, extent=grid_extent, cmap='viridis', vmin=-100, vmax=1200, aspect='equal')
plt.colorbar(label='[m a.s.l.]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "topography.png"
fig.savefig(file, dpi=300)
plt.close(fig)

for i, elevation_bottom_layer in enumerate(elevation_bottom_layers):
    i = i + 1
    fig, axes = plt.subplots(figsize=(4, 4))
    elevation_bottom_layer[~mask] = np.nan
    plt.imshow(elevation_bottom_layer, extent=grid_extent, cmap='viridis', vmin=-100, vmax=1200, aspect='equal')
    plt.colorbar(label='[m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"elevation_bottom_layer_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

file = base_path / "input" / "initial_conditions.grd"
initial_conditions = Raster.load(file)
initial_conditions_layer1 = initial_conditions.get_array(1)
initial_conditions_layer2 = initial_conditions.get_array(2)
initial_conditions_layer3 = initial_conditions.get_array(3)
initial_conditions_layer4 = initial_conditions.get_array(4)
initial_conditions_layers = [initial_conditions_layer1, initial_conditions_layer2, initial_conditions_layer3, initial_conditions_layer4]
for i, initial_conditions_layer in enumerate(initial_conditions_layers):
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(initial_conditions_layer, extent=grid_extent, cmap='viridis', aspect='equal')
    plt.colorbar(label='groundwater head [m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"initial_conditions_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

file = base_path / "input" / "boundary_condition.grd"
boundary_condition = Raster.load(file).get_array(1)
# boundary_condition = np.empty_like(domain.get_array(1))
# mask1 = (boundary_condition_ == -1)
# boundary_condition[mask1] = 1
fig, axes = plt.subplots(figsize=(4, 4))
plt.imshow(boundary_condition, extent=grid_extent, cmap='Greys', aspect='equal')
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "boundary_condition.png"
fig.savefig(file, dpi=300)
plt.close(fig)

file = base_path / "input" / "hydraulic_conductivity_layer1.grd"
hydraulic_conductivities_layer1 = Raster.load(file).get_array(1)
file = base_path / "input" / "hydraulic_conductivity_layer2.grd"
hydraulic_conductivities_layer2 = Raster.load(file).get_array(1)
file = base_path / "input" / "hydraulic_conductivity_layer3.grd"
hydraulic_conductivities_layer3 = Raster.load(file).get_array(1)
file = base_path / "input" / "hydraulic_conductivity_layer1.grd"
hydraulic_conductivities_layer4 = Raster.load(file).get_array(1)
hydraulic_conductivities_layers = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]
for i, hydraulic_conductivities_layer in enumerate(hydraulic_conductivities_layers):
    i = i + 1
    fig, axes = plt.subplots(figsize=(4, 4))
    hydraulic_conductivities_layer[~mask] = np.nan
    plt.imshow(hydraulic_conductivities_layer, extent=grid_extent, cmap='Oranges', aspect='equal')
    plt.colorbar(label='$k_f$ [m/day]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path_figs / f"hydraulic_conductivity_layer_{i}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)



fig, axes = plt.subplots(figsize=(4, 4))
topography[~mask] = np.nan
elevation_bottom_layer1[~mask] = np.nan
plt.imshow(topography - elevation_bottom_layer1, extent=grid_extent, cmap='viridis', aspect='equal')
plt.colorbar(label='[m]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"thickness_layer1.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
elevation_bottom_layer1[~mask] = np.nan
elevation_bottom_layer2[~mask] = np.nan
plt.imshow(elevation_bottom_layer1 - elevation_bottom_layer2, extent=grid_extent, cmap='viridis', aspect='equal')
plt.colorbar(label='[m]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"thickness_layer2.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
elevation_bottom_layer2[~mask] = np.nan
elevation_bottom_layer3[~mask] = np.nan
plt.imshow(elevation_bottom_layer2 - elevation_bottom_layer3, extent=grid_extent, cmap='viridis', aspect='equal')
plt.colorbar(label='[m]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"thickness_layer3.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(figsize=(4, 4))
elevation_bottom_layer3[~mask] = np.nan
elevation_bottom_layer4[~mask] = np.nan
plt.imshow(elevation_bottom_layer3 - elevation_bottom_layer4, extent=grid_extent, cmap='viridis', aspect='equal')
plt.colorbar(label='[m]', shrink=0.5)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / f"thickness_layer4.png"
fig.savefig(file, dpi=300)
plt.close(fig)