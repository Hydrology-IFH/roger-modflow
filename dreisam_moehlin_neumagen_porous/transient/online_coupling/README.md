# Porous Aquifer of the Dreisam-Moehlin-Neumagen catchment

Simulation of the soil water and groundwater of the porous Aquifer of the Dreisam-Moehlin-Neumagen catchment (Germany) using MODFLOW coupled with RoGeR. The models are coupled online (i.e. variables/boundary conditions are updated after every time step).

Short description of files and folders:
- `input/`: contains precipitation data (`PREC.txt`; 10 minutes time steps), air temperature data (`TA.txt`; daily time steps) and potential evapotranspiration data (`PET.txt`; daily time steps).
- `output/`: contains the model output
- `figures/`: contains the figures
- `config_roger.yml`: File to set parameters and output variables of RoGeR
- `write_parameters_to_csv_for_SVAT.py`: Generates model parameters (`parameters.csv`) and writes model parameter file of RoGeR with vertical fluxes only
- `write_parameters_to_csv_for_ONED.py`: Generates model parameters (`parameters.csv`) and writes model parameter file of RoGeR with lateral fluxes (i.e. no transfer between grid cells)
- `calculate_PET_for_climate_projections.py`: Calculates potential evapotranspiration from projected air temperature and projected solar radiation using the Makkink formula.
- `parameters.csv`: Model parameters of RoGeR
- `modflow6_steady-state.py`: Python script to run MODFLOW6 using xmipy for steady-state simulations
- `roger-xxxx_modflow6_transient.py`: Python script to run RoGeR and MODFLOW6 using xmipy for transient simulations (`xxxx` represents the model structure either SVAT or ONED). SVAT means only vertical fluxes are considered. ONED includes lateral fluxes.
- `export_binary_output_to_netcdf.py`: Writes the MODFLOW6 output into a single netCDF file
- `plot_input_data.py`: Plots the topography and model parameters (e.g. hydraulic conductivity)
- `plot_groundwater_heads.py`: Plots the groundwater head at different time steps
- `plot_time_series.py`: Plots observed and simulated groundwater head time series and simulated recharge time series
- `make_gif.py`: Animated visualisation of spatially distributed simulations of the groundwater recharge and groundwater head
- `test_regrid.py`: Minimal working example for the spatial aggregation (from finer to coarser resolution and from coarser to finer resolution)

`input/`, `output/` and larger *.nc.files are stored on FUHYS018 in `StressRes_RoGeR-ModFlow/` since GitHub is not meant to be a large data storage facility. Please contact [Jürgen Strub](juergen.strub@hydrology.uni-freiburg.de) or [Markus Weiler](markus.weiler@hydrology.uni-freiburg.de) to access the data and put the required data into your local disk.

Workflow:
1. Run `plot_input_data.py`
2. Run the steady-state simulation `python modflow6_steady-state.py`. Steady-state simulation is required to set the boundary and initial conditions of the transient simulation. 
3. Run the transient simulation `python roger_modflow6_transient.py`
4. Run `python export_binary_output_to_netcdf.py`
5. Run `python plot_groundwater_heads.py`
6. Run `python plot_time_series.py`
7. Run `python make_gif.py`

