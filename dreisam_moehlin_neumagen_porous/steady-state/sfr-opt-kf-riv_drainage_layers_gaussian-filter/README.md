# Steady-state groundwater model of porous Aquifer of the Dreisam-Moehlin-Neumagen catchment

## Files
- `cleanup.py`: Cleanup after every model run and remove that is not needed anymore
- `evaluate_stead-state_simulation.py`: Evaluate a single MODFLOW simulation and plot comparions between observations and simulations
- `evaluate_stead-state_simulations.py`: Evaluate all MODFLOW simulation and calculate performance metrics
- `make_parallel_jobs.py`: Prepare parallel Monte-Carlo simulations
- `modflow6_steady-state.py`: Runs a steady-state simulation using flopy and xmipy
- `modflow6_steady-state_.py`: Copy of `modflow6_steady-state.py` but with modfications that it can be used to run parallel simulations
- `modflow6_steady-state_single_run.sh`: Runs simulation and evaluation.
- `modflow6_steady-state_monte_carlo.sh`: Sequential Monte-Carlo simulations
- `modflow6_steady-state_monte_carlo.sh`: Sequential Monte-Carlo simulations
- `submit_jobs.sh`: submit multiple jobs to perform naive parallel simulations on BwUniCluster3.0
- `write_binary_to_netcdf_steady-state.py`: Writes the values of the MODFLOW6 output files in a single netCDF-file
- `write_fudge_parameters.py`: Write fudge parameters used for Monte Carlo simualtions. Parameter ranges are defined in the file. Fudging means hydraulic conductivity is decreased/increased by a factor.


## Workflow
1. `write_fudge_parameters.py`
2. `modflow6_steady-state.py --model-run 5`: Run the model. --model-run 5 means initial paramters are used
3. `write_binary_to_netcdf_steady-state.py --model-run 5`: Write the output to netCDF
4. `plot_groundwater_heads_steady-state.py --model-run 5`: Plot the groundwater heads
5. `evaluate_steady-state_simulation.py --model-run 5`: Compare to observations