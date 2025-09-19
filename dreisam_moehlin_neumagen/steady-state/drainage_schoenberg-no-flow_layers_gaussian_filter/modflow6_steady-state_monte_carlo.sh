#!/bin/bash

# python add_schoenberg_mask.py
# python add_drainage_mask.py
# python modify_elevations.py
# python modify_hydraulic_conductivity_and_specific_yield_of_layer1.py
# python generate_boundary_conditions.py
# python write_fudge_parameters.py
for i in {216..10000}
do
    converged=$(python modflow6_steady-state.py --model-run $i | grep "converged: " | awk '{print $NF}')
    python write_binary_to_netcdf_steady-state.py --model-run $i --converged $converged
    python cleanup.py --model-run $i
    progress=$i/10000
    echo "Model run $i done ($progress)"
done