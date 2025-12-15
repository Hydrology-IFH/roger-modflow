#!/bin/bash
python write_fudge_parameters.py
python modflow6_steady-state.py --model-run 2531
python write_binary_to_netcdf_steady-state_pre.py --model-run 2531
python modflow6_steady-state_rerun.py --model-run 2531
python write_binary_to_netcdf_steady-state_pre1.py --model-run 2531
python modflow6_steady-state_rerun_rerun.py --model-run 2531
python write_binary_to_netcdf_steady-state.py --model-run 2531
python evaluate_steady-state_simulation.py --model-run 2531
python plot_groundwater_heads_steady-state.py --model-run 2531
python plot_contours_near_gw_extraction.py --model-run 2531

