#!/bin/bash
python write_fudge_parameters.py

# python modflow6_steady-state.py --model-run 5
# python evaluate_boundary_condition.py --model-run 5

python modflow6_steady-state.py --model-run 5
python write_binary_to_netcdf_steady-state.py --model-run 5
python modflow6_steady-state_rerun.py --model-run 5
python write_binary_to_netcdf_steady-state_diagnose_sfr.py --model-run 5
# python evaluate_boundary_condition.py --model-run 5 --plot
python plot_groundwater_heads_steady-state.py --model-run 5
# python plot_contours_near_gw_extraction.py --model-run 5
python diagnose_sfr.py --model-run 5
