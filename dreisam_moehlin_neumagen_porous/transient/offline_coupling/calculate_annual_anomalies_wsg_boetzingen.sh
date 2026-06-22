#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=calculate_annual_anomalies_wsg_boetzingen
#SBATCH --output=calculate_annual_anomalies_wsg_boetzingen.out
#SBATCH --error=calculate_annual_anomalies_wsg_boetzingen_err.out
#SBATCH --export=ALL
module load devel/miniforge
conda activate roger-modflow
cd /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling

python calculate_annual_anomaly_metrics.py --area wsg_boetzingen
ls
