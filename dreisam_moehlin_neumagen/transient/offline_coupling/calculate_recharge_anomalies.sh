#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=calculate_recharge_anomalies
#SBATCH --output=calculate_recharge_anomalies.out
#SBATCH --error=calculate_recharge_anomalies_err.out
#SBATCH --export=ALL

module load devel/miniforge
conda activate roger-modflow
cd /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling

# # list with stress test scenarios
# model_runs=(
#     "base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"
#     "summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction"
#     "summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress"
# )

# # loop over stress test scenarios and copy files from project to work directory
# for model_run in "${model_runs[@]}"; do
#     cp -r /pfs/10/project/bw22g004/fr_rs1092/workspace-1773831854/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/${model_run}/*.gz /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/.
#     sleep 120
#     tar -xzf /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/${model_run}/*.gz -C /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/${model_run}/
# done

# tar -xzf /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/*_rci_*.tar.gz -C /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/

python calculate_recharge_anomalies.py

# # remove .nc-files from work directory
# for model_run in "${model_runs[@]}"; do
#     rm /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/${model_run}/*.nc
# done
# rm /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_*.nc
# rm /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_*.nc
