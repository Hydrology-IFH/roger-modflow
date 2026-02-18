#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=modflow_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only
#SBATCH --output=modflow_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only.out
#SBATCH --error=modflow_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_err.out
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
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/roger-oneD_modflow6_transient_with_well_extraction.py ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/config_modflow.yml ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/input ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/fudge_parameters_modflow.csv ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2013.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2014.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2015.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2016.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2017.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2018.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2019.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2020.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2021.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2022.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2023.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2013.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2014.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2015.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2016.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2017.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2018.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2019.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2020.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2021.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2022.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only_year2023.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
sleep 120
cd ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling
echo "Start simulation ..."
python roger-oneD_modflow6_transient_with_well_extraction.py --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction soil-compaction --grain-corn-only grain-corn-only
echo "... finalised simulation."
# Move output from local SSD to global workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only"
mkdir -p /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only
mv -v "${TMPDIR}"/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output
