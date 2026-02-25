#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=write_modflow_data_base_soil-compaction_irrigation
#SBATCH --output=write_modflow_data_base_soil-compaction_irrigation.out
#SBATCH --error=write_modflow_data_base_soil-compaction_irrigation_err.out
#SBATCH --export=ALL

module load devel/miniforge
eval "$(conda shell.bash hook)"
conda activate roger-modflow
cd /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling

python write_binary_to_netcdf_transient.py --stress-test-meteo base --soil-compaction soil-compaction --irrigation irrigation

