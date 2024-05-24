# Simple hillslope example considering recharge and well extraction

MODFLOW6 example for a hillslope with a constant slope. The aquifer has the following properties:
- single layer
- homogenous hydraulic conductivity

Short description of files and folders:
- `output/`: contains the model output
- `figures/`: contains the figures
- `config.yml`: File to set settings and parameters of MODFLOW6
- `elevation.png`: Topography of the hillslope
- `modflow6_transient.py`: Python script to run MODFLOW6 in transient mode using xmipy
- `export_binary_output_to_netcdf.py`: Writes the MODFLOW6 output into a single netCDF file
- `plot_groundwater_heads.py`: Plots the groundwater head at different time steps

Workflow:
1. Run `python modflow6_transient.py`
2. Run `python export_binary_output_to_netcdf.py`
3. Run `plot_groundwater_heads.py`