#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=write_modflow_data_spring-summer-drought_magnitude0_duration3_soil-compaction
#SBATCH --output=write_modflow_data_spring-summer-drought_magnitude0_duration3_soil-compaction.out
#SBATCH --error=write_modflow_data_spring-summer-drought_magnitude0_duration3_soil-compaction_err.out
#SBATCH --export=ALL

module load devel/miniforge
conda activate roger-modflow
cd /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling

python write_binary_to_netcdf_transient.py --stress-test-meteo spring-summer-drought --stress-test-meteo-magnitude 0 --stress-test-meteo-duration 3 --soil-compaction soil-compaction

