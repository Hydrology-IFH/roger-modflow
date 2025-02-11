#!/bin/bash

# python add_schoenberg_mask.py
# python add_drainage_mask.py
# python add_subcatchment_masks.py
# python modify_elevations.py
# python modify_hydraulic_conductivity_and_specific_yield_of_layer1.py
python generate_boundary_conditions.py
python write_fudge_parameters.py
for i in {0..5000}
do
    python modflow6_steady-state.py --model-run $i
    python write_binary_to_netcdf_steady-state.py --model-run $i
    python cleanup.py --model-run $i
    progress=$i/5000
    echo "Model run $i done ($progress)"
done