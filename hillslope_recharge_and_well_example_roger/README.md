# Simple hillslope example considering recharge and well extraction

Example to couple RoGeR and MODFLOW6. We use a hillslope with a constant slope. The aquifer has the following properties:
- single layer
- homogenous hydraulic conductivity

Short description of files and folders:
- `input/`: contains precipitation data (`PREC.txt`; 10 minutes time steps), air temperature data (`TA.txt`; daily time steps) and potential evapotranspiration data (`PET.txt`; daily time steps).
- `output/`: contains the model output
- `figures/`: contains the figures
- `elevation.png`: Topography of the hillslope
- `config.yml`: File to set parameters and output variables of RoGeR
- `write_parameters.py`: Generates model parameters (`parameters.csv`) and writes model parameter file of RoGeR
- `parameters.csv`: Model parameters of RoGeR
- `roger_modflow6_steady-state.py`: Python script to run RoGeR and MODFLOW6 using xmipy for steady-state simulations
- `roger_modflow6_transient.py`: Python script to run RoGeR and MODFLOW6 using xmipy for transient simulations
- `export_binary_output_to_netcdf.py`: Writes the MODFLOW6 output into a single netCDF file
- `plot_groundwater_heads.py`: Plots the groundwater head at different time steps

Workflow:
1. Run `python roger_mf6.py`
2. Run `python export_binary_output_to_netcdf.py`
3. Run `plot_groundwater_heads.py`