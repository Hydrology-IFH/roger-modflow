#!/bin/bash
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=modflow_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress
#SBATCH --output=modflow_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress.out
#SBATCH --error=modflow_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress_err.out
#SBATCH --export=ALL
module load devel/miniforge
conda activate roger-modflow
cd /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling

mkdir ${TMPDIR}/roger-modflow
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/bin ${TMPDIR}/roger-modflow
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/roger_modflow6.py ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/config_modflow.yml ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/input ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/fudge_parameters_modflow.csv ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient
cp -r /pfs/10/project/bw22g004/fr_rs1092/workspace-1773831854/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/ONEDCROP_rci_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction.tar.gz /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/
sleep 160
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/ONEDCROP_rci_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction.tar.gz ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
tar -xf ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/ONEDCROP_rci_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction.tar.gz -C ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
sleep 160
rm /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/ONEDCROP_rci_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction.tar.gz
cd ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling
echo "Start simulation ..."
python roger_modflow6.py --stress-test-meteo long-term --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction soil-compaction --irrigation irrigation --stress-test-well-extraction stress
echo "... finalised simulation."
# Move output from local SSD to global workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress"
mkdir -p /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress
mv -v "${TMPDIR}"/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress/{*,.[!.]*} /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress