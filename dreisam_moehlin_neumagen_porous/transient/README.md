# Porous Aquifer of the Dreisam-Moehlin-Neumagen catchment

- `offline_coupling/`: Offline coupling between RoGeR and MODFLOW6
- `online_coupling/`: Online coupling between RoGeR and MODFLOW6

See READMEs in the subfolders for more information.

Short description of the files and folders:
- `input/`: contains input data and data to parameterise the models
- `output/`: contains the model output
- `figures/`: contains the figures
- `config_roger.yml`: Configuration file of RoGeR
- `config_modflow.yml`: Configuration file of RoGeR
- `make_maps.py`: Plot catchment, water protection areas and well locations as maps
- `plot_drinking_water_extraction.py`: Plot time series of the drinking water extraction volumes
- `prepare_groundwater_head_time_series.py`: Remove outliers from the available groundwater head time series
- `spatio_temporal_universal_kriging.py`: Interpolate the groundwater heads using a spatio-temporal universal kriging
- `write_initial_conditions.py`: Generates initial groundwater heads using universal kriging at 1st January 20213
- `write_stress_tests_lateral_recharge.py`: If PET > PRECIP for future seasons, lateral recharge is decreased. 
- `write_stress_tests_sfr_inflow.py`: If PET > PRECIP for future seasons, river inflow at the cacthemnt boundary is decreased. 
- `write_stress_tests_well_extraction.py`: Increase well extraction by 10%.

Workflow
