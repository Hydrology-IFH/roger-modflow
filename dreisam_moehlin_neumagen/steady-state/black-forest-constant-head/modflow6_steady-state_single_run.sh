#!/bin/bash

python add_subcatchment_masks.py
python modify_elevations.py
python modify_hydraulic_conductivity_and_specific_yield_of_layer1.py
python generate_boundary_conditions.py
python write_fudge_parameters.py

# python modflow6_steady-state.py --model-run 5
# python evaluate_boundary_condition.py --model-run 5

python modflow6_steady-state.py --model-run 5
python write_binary_to_netcdf_steady-state.py --model-run 5
python plot_groundwater_heads_steady-state.py --model-run 5
