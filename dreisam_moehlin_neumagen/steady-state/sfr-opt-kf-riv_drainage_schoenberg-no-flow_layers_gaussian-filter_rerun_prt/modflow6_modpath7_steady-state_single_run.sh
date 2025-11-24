#!/bin/bash
python write_fudge_parameters.py
python modflow6_steady-state.py --model-run 8304
python write_binary_to_netcdf_steady-state_pre.py --model-run 8304
python modflow6_modpath7_steady-state.py --model-run 8304
python write_binary_to_netcdf_steady-state.py --model-run 8304