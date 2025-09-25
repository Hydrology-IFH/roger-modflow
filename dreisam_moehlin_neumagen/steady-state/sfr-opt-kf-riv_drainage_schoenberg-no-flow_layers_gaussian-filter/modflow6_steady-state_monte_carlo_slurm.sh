#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers_gaussian-filter
#SBATCH --output=sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers_gaussian-filter.out
#SBATCH --error=sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers_gaussian-filter_err.out
#SBATCH --export=ALL
 
module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate roger-modflow
cd /pfs/work9/workspace/scratch/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/steady-state/sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers_gaussian-filter
 
python write_fudge_parameters.py
for i in {0..10000}
do
    converged=$(python modflow6_steady-state.py --model-run $i | grep "converged: " | awk '{print $NF}')
    python write_binary_to_netcdf_steady-state.py --model-run $i --converged $converged
    python cleanup.py --model-run $i
    progress=$i/10000
    echo "Model run $i done ($progress; $converged)"
done