# Porous Aquifer of the Dreisam-Moehlin-Neumagen catchment

Simulation of the soil water and groundwater of the porous Aquifer of the Dreisam-Moehlin-Neumagen catchment (Germany) using MODFLOW coupled with RoGeR. The two models are coupled offline (i.e. run RoGeR first and pass the simulated values to MODFLOW loading the RoGeR model output).

Short description of files and folders:
- `output/`: contains the model output
- `figures/`: contains the figures
- `config_roger.yml`: Configuration-File of RoGeR
- `config_modflow.yml`: Configuration-File of MODFLOW6
- `roger_modflow6.py`: Python script to run MODFLOW6 using xmipy for transient simulations
- `write_binary_to_netcdf_transient.py`: Writes the MODFLOW6 output files of the transient simulations into a single netCDF file
- `evaluate_groundwater_depths.py`: Compare simulated groundwater depths with observed groundwater depths
- `evaluate_groundwater_depths.sh`: Runs `evaluate_groundwater_depths.py` as computing job on BinAC2
- `evaluate_indirect_recharge.py`: Compare simulated streamflow with observed groundwater streamflow
- `evaluate_indirect_recharge.sh`: Runs `evaluate_indirect_recharge.py` as computing job on BinAC2
- `calculate_anomaly_metrics.py`: Calculate anomalies of evapotranspiration, precipitation, air temperature, irrigation, groundwater recharge and groundwater heads for different years and water protection areas within the catchment
- `calculate_anomaly_metrics.sh`: Runs `calculate_anomaly_metrics.py` as computing job on BinAC2
- `plot_anomaly_metrics.py`: Plot anomalies of evapotranspiration, precipitation, air temperature, irrigation, groundwater recharge and groundwater heads for different years and water protection areas within the catchment
- `plot_anomaly_metrics.sh`: Runs `plot_anomaly_metrics.py` as computing job on BinAC2
- `calculate_gw_anomalies.py`: Calculate groundwater head anomalies of the entire catchment, the water protection area Hausen and Zartener Becken and plot anomalies as time series and maps
- `calculate_gw_anomalies.sh`: Runs `calculate_gw_anomalies.py` as computing job on BinAC2
- `calculate_gw_extraction_balance.py`: Calculate the groundwater extraction balance of the water protection area Hausen and Zartener Becken and make bar plots
- `calculate_gw_extraction_balance.sh`: Runs `calculate_gw_extraction_balance.py` as computing job on BinAC2
- `calculate_recharge_anomalies.py`: Calculate groundwater recharge anomalies of the entire catchment, the water protection area Hausen and Zartener Becken and plot anomalies as time series and maps
- `calculate_recharge_anomalies.sh`: Runs `calculate_recharge_anomalies.py` as computing job on BinAC2
- `write_data_job_scripts_slurm.py`: Write SLURM job scripts to write the MODFLOW6 output files of the transient simulations into a single netCDF file
- `submit_data_jobs.sh`: Submit writing output data of scenario runs as computing jobs on BinAC2
- `write_job_scripts_slurm.py`: Write SLURM job scripts to compute RoGeR-MODFLOW6 simulations
- `submit_jobs.sh`: Submit scenario runs as computing jobs on BinAC2

`input/`, `output/` and larger *.nc.files are stored on FUHYS018 in `StressRes_RoGeR-ModFlow/` since GitHub is not meant to be a large data storage facility. Please contact [Jürgen Strub](juergen.strub@hydrology.uni-freiburg.de) or [Markus Weiler](markus.weiler@hydrology.uni-freiburg.de) to access the data and put the required data into your local disk.


## Reference scenario (base)
Workflow on local computer:
1. Run the [RoGeR-ONED model](https://github.com/Hydrology-IFH/roger/tree/main/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed)
2. `python roger_modflow6.py`
3. `python write_binary_to_netcdf_transient.py`
4. `python evaluate_groundwater_depths.py`
5. `python evaluate_indirect_recharge.py`

Workflow on BinAC2-cluster: 
1. Run the [RoGeR-ONED model](https://github.com/Hydrology-IFH/roger/tree/main/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed)
2. `python write_job_scripts_slurm.py`
3. `sbatch -p compute modflow_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction.sh`
4. `sbatch -p compute write_modflow_data_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction.sh`
5. `sbatch -p compute evaluate_groundwater_depths.sh`
6. `sbatch -p compute evaluate_indirect_recharge.sh`

## Stress test scenarios
Workflow on local computer:
1. Run the [RoGeR-ONED model](https://github.com/Hydrology-IFH/roger/tree/main/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed)
2. `python roger_modflow6.py --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction soil-compaction --irrigation irrigation --stress-test-well-extraction stress`
3. `python write_binary_to_netcdf_transient.py --stress-test-meteo summer-drought --stress-test-meteo-magnitude 2 --stress-test-meteo-duration 3 --soil-compaction soil-compaction --irrigation irrigation --stress-test-well-extraction stress`
4. `python calculate_gw_anomalies.py`
5. `python calculate_recharge_anomalies.py`

Workflow on BinAC2-cluster: 
1. Run the [RoGeR-ONED model](https://github.com/Hydrology-IFH/roger/tree/main/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed)
2. `python write_job_scripts_slurm.py`
3. `./submit_jobs.sh`
4. `./submit_data_jobs.sh`
From here on the order does not matter.
5. `sbatch -p compute calculate_anomaly_metrics.sh`
6. `sbatch -p compute calculate_gw_anomalies.sh`
7. `sbatch -p compute calculate_recharge_anomalies.sh`
8. `sbatch -p compute calculate_gw_extraction_balance.sh`