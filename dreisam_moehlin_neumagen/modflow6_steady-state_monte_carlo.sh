#!/bin/bash

python write_fudge_parameters.py
for i in {0..10}
do
    python generate_boundary_conditions.py

    python modflow6_steady-state.py --model-run $i
    python evaluate_boundary_condition.py --model-run $i

    python modflow6_steady-state.py --model-run $i
    python evaluate_boundary_condition.py --model-run $i

    python modflow6_steady-state.py --model-run $i
    python write_binary_to_netcdf_steady-state.py --model-run $i
    python cleanup.py --model-run $i
    progress=$i/5000
    echo "Model run $i done ($progress)"
done