# Porous Aquifer of the Dreisam-Moehlin-Neumagen catchment

- `offline_coupling/`: Offline coupling between RoGeR and MODFLOW6
- `online_coupling/`: Online coupling between RoGeR and MODFLOW6

See READMEs in the subfolders for more information.

Short description of the files and folders:
- `input/`: contains input data and data to parameterise the models
- `config_roger.yml`: Configuration file of RoGeR
- `config_modflow.yml`: Configuration file of RoGeR
- `calculate_groundwater_anomalies.py`: Plot anomalies using observed ground heads and results from spatio-temporal kriging of groundwater heads
- `make_maps.py`: Plot catchment, water protection areas and well locations as maps
- `plot_drinking_water_extraction.py`: Plot time series of the drinking water extraction volumes
- `prepare_groundwater_head_time_series.py`: Remove outliers from the available groundwater head time series
- `prepare_well_extraction_data.py`: Prepare well extraction data.
- `prepare_drinking_water_well_extraction_data.py`: Prepare well extraction data of drinking water wells
- `spatio_temporal_universal_kriging.py`: Interpolate the groundwater heads using a spatio-temporal universal kriging
- `write_initial_conditions.py`: Generates initial groundwater heads using universal kriging at 1st January 20213
- `write_stress_tests_lateral_recharge.py`: If PET > PRECIP for future seasons, lateral recharge is decreased. 
- `write_stress_tests_sfr_inflow.py`: If PET > PRECIP for future seasons, river inflow at the cacthment boundary is decreased. 
- `write_stress_tests_well_extraction.py`: Increase well extraction by 15% or 30%.

Workflow:
1. Optimise the steady-state groundwater model --> `../steady-state/sfr-opt-kf-riv_drainage_layers_gaussian-filter/`. Model run 1806 provided the best fit between observations and simulations (see `fudge_parameters_metrics_porous.csv`).
2. `python prepare_groundwater_head_time_series.py`
3. `python prepare_well_extraction_data.py`
4. `python prepare_drinking_water_well_extraction_data.py`
5. `python write_stress_tests_lateral_recharge.py`
6. `python write_stress_tests_sfr_inflow.py`
7. `python write_stress_tests_well_extraction.py`
8. Continue with either `offline_coupling/` or `online_coupling/`

## Stress-Test scenarios

### Climate stress
summer-drought:
- duration3-magnitude0: Summer drought of 2018 is repeated and occurs in 2016, 2017 and 2018 in current climate
- duration3-magnitude2: Summer drought of 2018 is repeated and occurs in 2016, 2017 and 2018 in future climate

long-term:
- duration0-magnitude2: Far future climate (2070 - 2099)

durationx: event is x years repeated.
magnitude1: using seasonal delta values of [RheiKlim](https://apps.hydro.uni-freiburg.de/de/RheiKlim/) for the near future (2040 - 2069)
magnitude2: using seasonale delta values of [RheiKlim](https://apps.hydro.uni-freiburg.de/de/RheiKlim/) for the far future (2070 - 2099)

### Agricultural management
- no-irrigation: no irrigation is applied on agricultural areas
- irrigation: irrigation is applied on agricultural areas
- no-yellow-mustard: No catch crop is considered before summer crops.
- yellow-mustard: Yellow mustard is cultivated before summer crops.
- soil-compaction: Soil compaction on agricultural areas is considered by decreasing air capacity and hydraulic conductivity.

### Drinking water management
- well-extraction-stress: Drinking water well extraction is increased by 30 % and if Ebnet waterworks exceeds 1000 m3/hour in stress periods, extraction volumes are redistributed to the Hausen waterworks.

## Files in output/
File names contain a combination of the stress test scenarios. For example:
- modflow_long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress.nc: MODFLOW6 simulation result of the period 2013-2023 with a future climate, agricultural irrigation, soil compaction of agricultural areas and increased extraction of drinking water wells.

If soil-compaction or well-extraction-stress does not occur in the file name, the stress test scenario is not applied.


## Spatio-temporal kriging of groundwater heads
An universal kriging is applied to monthly values of groundwater heads.

Workflow:
1. `python prepare_groundwater_head_time_series.py`
2. `python spatio_temporal_universal_kriging.py`
3. `python calculate_groundwater_anomalies.py`
