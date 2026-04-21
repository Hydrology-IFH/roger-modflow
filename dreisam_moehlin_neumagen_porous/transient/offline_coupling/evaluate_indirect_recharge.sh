#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=256000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=evaluate_indirect_recharge
#SBATCH --output=evaluate_indirect_recharge.out
#SBATCH --error=evaluate_indirect_recharge_err.out
#SBATCH --export=ALL

module load devel/miniforge
conda activate roger-modflow
cd /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling

cp -r /pfs/10/project/bw22g004/fr_rs1092/workspace-1773831854/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction/*.gz /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction/
sleep 120
tar -xzf /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction/modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_run_1806.tar.gz -C /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction/

python evaluate_indirect_recharge.py

# rm /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction/*.nc
# rm /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction/*.gz