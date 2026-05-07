#!/bin/bash
python write_fudge_parameters.py
python modflow6_modpath7_steady-state.py --model-run 1806
python write_binary_to_netcdf_steady-state.py --model-run 1806