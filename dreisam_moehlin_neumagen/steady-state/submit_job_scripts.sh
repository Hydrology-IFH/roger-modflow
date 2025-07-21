#!/bin/bash

cd sfr_drainage_schoenberg-no-flow_layers_gaussian_filter
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..

cd sfr-opt_drainage_schoenberg-no-flow_layers_gaussian_filter
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..

cd sfr_drainage_schoenberg-no-flow_layers
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..

cd sfr_schoenberg-no-flow_layers_gaussian_filter
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..

cd schoenberg-no-flow_layers_gaussian_filter
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..

cd schoenberg-no-flow_no-wells_layers_gaussian_filter
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..

cd sfr_drainage_schoenberg-no-flow_fudge4_gaussian_filter
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..
