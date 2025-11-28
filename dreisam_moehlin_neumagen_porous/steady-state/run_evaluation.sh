#!/bin/bash

cd schoenberg-no-flow_no-wells_layers_gaussian-filter
python evaluate_steady-state_simulations.py
cd ..

cd schoenberg-no-flow_layers_gaussian-filter
python evaluate_steady-state_simulations.py
cd ..

cd drainage_schoenberg-no-flow_layers_gaussian-filter
python evaluate_steady-state_simulations.py
cd ..

cd sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers_gaussian-filter
python evaluate_steady-state_simulations.py
cd ..

# cd sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers
# python evaluate_steady-state_simulations.py
# cd ..