# Backward particle tracking using the steady-state groundwater model of porous Aquifer in the Dreisam-Moehlin-Neumagen catchment

A backward particle tracking using MODPATH 7 is applied to every groundwater extraction well. The particles are released in circles with following radia: 1m, 5m, 10m and 25m. The following three release scenarios are considered:
- near-surface: particles released within the upper 10% of the screen length
- pump_installation_depth: particles from top of the screen until installation depth of the extraction pump
- deep: particles released over the entire screen length


## Files
- `modflow6_modpath7_steady-state.py`: Backward particle tracking using MODPATH 7 for each groundwater extraction well
- `define_well_catchments.py`: Backward particle tracking using MODPATH 7 for each groundwater extraction well



## Workflow
1. `write_fudge_parameters.py`
2. `modflow6_modpath7_steady-state.py`: Run the backward particle tracking
3. `write_binary_to_netcdf_steady-state.py --model-run 5`: Write the output to netCDF
4. `plot_groundwater_heads_steady-state.py --model-run 5`: Plot the groundwater heads
5. `evaluate_steady-state_simulation.py --model-run 5`: Compare to observations