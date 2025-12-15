#!/bin/bash
cd ..
python add_streamflow_observation_points_to_diagnose_sfr.py
cd sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers_gaussian-filter_diagnose
python write_fudge_parameters.py
python modflow6_steady-state.py --model-run 2531
python write_binary_to_netcdf_steady-state_diagnose_sfr.py --model-run 2531
python plot_groundwater_heads_steady-state.py --model-run 2531
python diagnose_sfr.py --model-run 2531
python plot_contours_near_gw_extraction.py --model-run 2531
python write_dmn_data.py --model-run 2531
python write_wsg_data.py --model-run 2531
python plot_wsg_data.py
cd ..
python add_streamflow_observation_points_to_sfr.py
cd sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers_gaussian-filter_diagnose
