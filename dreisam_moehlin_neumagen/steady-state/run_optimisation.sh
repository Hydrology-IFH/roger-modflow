#!/bin/bash

cd schoenberg-no-flow_no-wells_layers_gaussian-filter
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..

cd schoenberg-no-flow_layers_gaussian-filter
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..

cd drainage_schoenberg-no-flow_layers_gaussian-filter
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..

cd sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers_gaussian-filter
nohup ./modflow6_steady-state_monte_carlo.sh &
cd ..

# cd sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers
# nohup ./modflow6_steady-state_monte_carlo.sh &
# cd ..
