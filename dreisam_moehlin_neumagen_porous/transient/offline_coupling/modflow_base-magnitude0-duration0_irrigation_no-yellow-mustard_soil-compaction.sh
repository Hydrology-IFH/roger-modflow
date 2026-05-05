#!/bin/bash
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=modflow_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction
#SBATCH --output=modflow_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction.out
#SBATCH --error=modflow_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction_err.out
#SBATCH --export=ALL
module load devel/miniforge
conda activate roger-modflow
cd /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling

mkdir ${TMPDIR}/roger-modflow
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output
echo "Start copying data ..."
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/bin ${TMPDIR}/roger-modflow &&
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/roger_modflow6.py ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling &&
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/config_modflow.yml ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient &&
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/input ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient &&
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/fudge_parameters_modflow.csv ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient &&
cp -r /pfs/10/project/bw22g004/fr_rs1092/workspace-1773831854/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/ONEDCROP_rci_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction.tar.gz /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/ &&
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/ONEDCROP_rci_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction.tar.gz ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/input/ &&
echo "Start extracting archives ..."
tar -xf ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/input/ONEDCROP_rci_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction.tar.gz -C ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/input/ &&
echo "Finished copying data and extracting archives ..."
cd ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling
echo "Start simulation ..."
python roger_modflow6.py --stress-test-meteo base --soil-compaction soil-compaction --irrigation irrigation
echo "... finalised simulation"
# Remove files from workspace
rm /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/ONEDCROP_rci_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction.tar.gz
# Move output from local SSD to global workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction"
mkdir -p /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction
mv -v "${TMPDIR}"/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction/{*,.[!.]*} /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction