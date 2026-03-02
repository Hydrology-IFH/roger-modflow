#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=write_modflow_data_summer-drought_magnitude2_duration3_soil-compaction_yellow-mustard
#SBATCH --output=write_modflow_data_summer-drought_magnitude2_duration3_soil-compaction_yellow-mustard.out
#SBATCH --error=write_modflow_data_summer-drought_magnitude2_duration3_soil-compaction_yellow-mustard_err.out
#SBATCH --export=ALL

module load devel/miniforge
conda activate roger-modflow
cd /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling

python write_binary_to_netcdf_transient.py --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction soil-compaction --yellow-mustard yellow-mustard

