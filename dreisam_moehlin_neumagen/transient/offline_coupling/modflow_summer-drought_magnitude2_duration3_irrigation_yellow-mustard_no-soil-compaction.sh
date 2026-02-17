#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32000
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de
#SBATCH --job-name=modflow_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction
#SBATCH --output=modflow_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction.out
#SBATCH --error=modflow_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_err.out
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
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/config_modflow.yml ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient
cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/input ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2013.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2014.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2015.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2016.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2017.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2018.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2019.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2020.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2021.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2022.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/recharge_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2023.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2013.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2014.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2015.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2016.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2017.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2018.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2019.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2020.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2021.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2022.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/capillary_rise_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2023.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2013.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2014.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2015.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2016.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2017.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2018.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2019.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2020.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2021.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2022.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
cp -r /pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed/output/irrigation_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction_year2023.nc ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/input/
sleep 120
cd ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling
python roger-oneD_modflow6_transient_with_well_extraction.py --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --irrigation irrigation --yellow-mustard yellow-mustard
# Move output from local SSD to global workspace
echo "Move output to /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction"
mkdir -p /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction
mv -v "${TMPDIR}"/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction /pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen/transient/offline_coupling/output/modflow_summer-drought_magnitude2_duration3_irrigation_yellow-mustard_no-soil-compaction