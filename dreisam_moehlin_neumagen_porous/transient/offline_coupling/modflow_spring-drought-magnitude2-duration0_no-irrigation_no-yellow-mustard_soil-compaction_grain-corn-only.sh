#!/bin/bash
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=modflow_spring-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only
#SBATCH --output=modflow_spring-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only.out
#SBATCH --error=modflow_spring-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_err.out
#SBATCH --export=ALL
module load devel/miniforge
conda activate roger-modflow
cd /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling

mkdir ${TMPDIR}/roger-modflow
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling
mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/bin ${TMPDIR}/roger-modflow
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/roger_modflow6.py ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/config_modflow.yml ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/input ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/fudge_parameters_modflow.csv ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/ONEDCROP_rci_spring-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only.tar.gz ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
tar -xf ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/ONEDCROP_rci_spring-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only.tar.gz
sleep 120
cd ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling
echo "Start simulation ..."
python roger_modflow6.py --stress-test-meteo spring-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 0 --soil-compaction soil-compaction --grain-corn-only grain-corn-only
echo "... finalised simulation"
# Move output from local SSD to global workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_spring-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only"
mkdir -p /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_spring-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only
mv -v "${TMPDIR}"/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_spring-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only/{*,.[!.]*} /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/modflow_spring-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only