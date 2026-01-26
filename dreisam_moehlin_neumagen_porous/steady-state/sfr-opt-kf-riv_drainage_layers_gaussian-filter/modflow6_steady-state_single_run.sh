#!/bin/bash
python write_fudge_parameters.py

# python modflow6_steady-state.py --model-run 6768
# python evaluate_boundary_condition.py --model-run 6768

python modflow6_steady-state.py --model-run 6768
python write_binary_to_netcdf_steady-state.py --model-run 6768
# python evaluate_boundary_condition.py --model-run 6768 --plot
python plot_groundwater_heads_steady-state.py --model-run 6768
# python plot_contours_near_gw_extraction.py --model-run 6768
python evaluate_steady-state_simulation.py --model-run 6768
