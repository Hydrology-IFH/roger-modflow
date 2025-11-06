#!/bin/bash
python write_fudge_parameters.py
# python modflow6_steady-state.py --model-run 9491
# python write_binary_to_netcdf_steady-state_pre.py --model-run 9491
# python modflow6_steady-state_rerun.py --model-run 9491
# python write_binary_to_netcdf_steady-state_pre1.py --model-run 9491
# python write_binary_to_netcdf_steady-state_diagnose_sfr_pre.py --model-run 9491
python modflow6_steady-state_rerun_rerun.py --model-run 9491
python write_binary_to_netcdf_steady-state_diagnose_sfr.py --model-run 9491
python plot_groundwater_heads_steady-state.py --model-run 9491
python diagnose_sfr.py --model-run 9491
python plot_contours_near_gw_extraction.py --model-run 9491
python write_wsg_data.py --model-run 9491
# python plot_wsg_data.py
