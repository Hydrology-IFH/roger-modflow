#!/bin/bash

python write_fudge_parameters.py
for i in {5..10000}
do
    converged=$(python modflow6_steady-state.py --model-run $i | grep "converged: " | awk '{print $NF}')
    python write_binary_to_netcdf_steady-state.py --model-run $i --converged $converged
    python cleanup.py --model-run $i
    progress=$i/10000
    echo "Model run $i done ($progress; $converged)"
done