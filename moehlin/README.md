# Moehlin catchment


Short description of files and folders:
- `input/`: contains precipitation data (`PREC.txt`; 10 minutes time steps), air temperature data (`TA.txt`; daily time steps) and potential evapotranspiration data (`PET.txt`; daily time steps).
- `output/`: contains the model output
- `figures/`: contains the figures
- `elevation.tiff`: Topography of the hillslope
- `config.yml`: File to set parameters and output variables of RoGeR
- `write_parameters_to_csv_for_SVAT.py`: Generates model parameters (`parameters.csv`) and writes model parameter file of RoGeR with vertical fluxes only
- `write_parameters_to_csv_for_ONED.py`: Generates model parameters (`parameters.csv`) and writes model parameter file of RoGeR with lateral fluxes (i.e. no transfer between grid cells)
- `parameters.csv`: Model parameters of RoGeR
- `modflow6_steady-state.py`: Python script to run MODFLOW6 using xmipy for steady-state simulations
- `roger_modflow6_transient.py`: Python script to run RoGeR and MODFLOW6 using xmipy for transient simulations
- `export_binary_output_to_netcdf.py`: Writes the MODFLOW6 output into a single netCDF file
- `plot_groundwater_heads.py`: Plots the groundwater head at different time steps
- `test_regrid.py`: Minimal working example for the spatial aggregation (from finer to coarser resolution and from coarser to finer resolution)

Workflow:
1. Run the steady-state simulation `python modflow6_steady-state.py`. Steady-state simulation is required to set the boundary and initial conditions of the transient simulation. 
2. Run the transient simulation `python roger_modflow6_transient.py`
3. Run `python export_binary_output_to_netcdf.py`
4. Run `plot_groundwater_heads.py`

Coordinate system:
proj=utm +zone=32 +ellps=GRS80 +units=m +no_defs
