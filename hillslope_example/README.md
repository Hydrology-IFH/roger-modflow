# Simple hillslope example

MODFLOW6 example for a hillslope with a constant slope.

- `output/`: contains the model output
- `figures/`: contains the figures
- `elevation.tiff`: Topography of the hillslope
- `mf6_xmi.py`: Python script to run MODFLOW6 using xmipy
- `export_binary_output_to_netcdf.py`: Writes the MODFLOW6 output into a single netCDF file
- `plot_groundwater_heads.py`: Plots the groundwater head at different time steps

Workflow:
1. Run `python mf6_xmi.py`
2. Run `python export_binary_output_to_netcdf.py`
3. Run `plot_groundwater_heads.py`