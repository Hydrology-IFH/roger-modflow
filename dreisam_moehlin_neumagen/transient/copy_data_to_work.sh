#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=copy_data_to_work
#SBATCH --output=copy_data_to_work.out
#SBATCH --error=copy_data_to_work_err.out
#SBATCH --export=ALL

cp -r /pfs/10/project/bw22g004/fr_rs1092/workspace-1773831854/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/*.gz /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/.
tar -xzf /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/*.gz -C /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/