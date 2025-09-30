# Dreisam-Moehlin-Neumagen catchment

Steady-state optimisation of different MODFLOW6 setups (i.e. different combinations of MODFLOW6 packages) of the Dreisam-Moehlin-Neumagen catchment.

## TODO

- Description of the preprocessing steps to derive the hydrogeologic data (mainly kf and layer thickness) --> Julian
- Recharge data --> Hannes: Period of simulation? Which model and assumptions?
- Interpolated groundwater heads: Andreas S./Markus --> Reference to the map


## Files
- `config.yaml`: Configuration file (see in-file comments for more information)
- `add_masks.py`: Add the masks (e.g. catchment mask) to the MODFLOW6 parameter file
- `modify_elevations.py`: Ensure that bottom elevations of the layers do not overlap. Bottom elevations are adjusted in case of intersections. 
- `modify_hydraulic_conductivity_and_specific_yield_of_layer1.py`: Replace hydarulic conductivities by hydraulic conductivities of the BK50 soil map.
- `generate_boundary_conditions.py`: Generate the boundary conditions file and add the no flow, constant head and recharge boundary condition.
- `define_drainage_area.py`: Define the drainage area of the former Tuniberg wetland and the mask to `modflow_parameters.nc`. 
- `make_stream_segment_routing.py`: Identify the downstream segment for each stream segment.
- `repair_geometries_of_stream_segments.py`: Repair the geometries of the stream segments by reversing the order of the vertices.
- `write_sfr_data.py`: Run the `sfrmaker` tool to write the data required by SFR package of MODFLOW6
- `assign_streambed_conductivity_and_manning_to_sfr_packagedata.py`: Assign values from `input/streambed_conductivity.tif` and `input/manning.tif` to `input/sfr_packagedata.csv`
- `make_diversions.py`: Write file to consider diversions in SFR package.
- `add_streamflow_observation_points_to_sfr.py`: Add observation points to collect the water depth and downstream flow of surface waters.
- `plot_input_data.py`: Plot maps of the input data (topography, hydraulic conductivities, groundwater extraction, observation wells)
- `input/modflow_parameters.nc`: Contains hydrogeologic parameters as rasters (Topography, hydraulic conductivities, masks of specific regions)
- `input/boundary_conditions.nc`: Contains boundary conditions as rasters

## Model structures
The model structures are organised in folders (ordered by ascending complexity):
- `schoenberg-no-flow_no-wells_layers_gaussian-filter`: Schoenberg is represented by no flow boundary. No wells are considered and tranisitions between hydraulic conductivities are smoothed by gaussian filter
- `schoenberg-no-flow_layers_gaussian-filter`: Schoenberg is represented by no flow boundary and tranisitions between hydraulic conductivities are smoothed by gaussian filter
- `drainage_schoenberg-no-flow_layers_gaussian-filter`: Drainage pipes are considered. Schoenberg is represented by no flow boundary and tranisitions between hydraulic conductivities are smoothed by gaussian filter.
- `sfr_schoenberg-no-flow_layers_gaussian-filter`: Interaction between rivers and groundwater is enabled. Drainage pipes are considered. Schoenberg is represented by no flow boundary and tranisitions between hydraulic conductivities are smoothed by gaussian filter.
- `sfr-opt_drainage_schoenberg-no-flow_layers_gaussian-filter`: Interaction between rivers and groundwater is enabled and tuning of the streambed conductivity is conducted. Drainage pipes are considered. Schoenberg is represented by no flow boundary and tranisitions between hydraulic conductivities are smoothed by gaussian filter.
- `sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers_gaussian-filter`: Interaction between rivers and groundwater is enabled. Tuning of the streambed conductivity and the hydraulic conductivity of fluvial sediments is conducted. Drainage pipes are considered. Schoenberg is represented by no flow boundary and tranisitions between hydraulic conductivities are smoothed by gaussian filter.
- `sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers`: Interaction between rivers and groundwater is enabled. Tuning of the streambed conductivity and the hydraulic conductivity of fluvial sediments is conducted. Drainage pipes are considered. Schoenberg is represented by no flow boundary.

## Topography
A 50 m x 50 m digital elevation model is used to describe the surface topography. A 5 m x 5 m digital elevation model is used to describe the riverbed. 

- `input/dem_5m.tif`: 5 m x 5 m digital elevation model of the Dreisam-Moehlin-Neumagen catchment as raster file
- `input/dem_50m.tif`: 50 m x 50 m digital elevation model of the Dreisam-Moehlin-Neumagen catchment as raster file

## Hydrogeology, hydraulic conductivities and specific yield
The Hydrogeology is described by 4 layers using a raster format with 50 m x 50 m spatial resolution. Thickness and hydraulic conductivities are derived from the LGRB geodataset "Hydrogeologischer Bau und Aquifereigenschaften der Lockergesteine im Oberrheingraben (Baden-Wuerttemberg)". Specific yield is calculated from the hydraulic conductivities using the formula of Marotz (1968; $n_a = 0.462 + 0.045 \times \ln(k_f)$). 

- `input/parameters_modflow.nc`: Variable `kf` contains the hydraulic conductivities derived from the LGRB geodataset
- `input/parameters_modflow.nc`: Variable `elevations` describes the surface topography and the bottom elevation of the four layers

## Recharge
The annual average recharge is derived by RoGeR simulations.

- `input/recharge_roger_50m.tif`: Annual average groundwater recharge of the period 2013-2022 simulated by RoGeR using a 50 m x 50 m resolution
- `input/boundary_conditions.nc`: Variable `recharge` contains annual average groundwater recharge of the period 2013-2022 simulated by RoGeR 

## Constant head
Constant head is set at the boundary to the upper rhine aquifer. To define the depth of the constant head, we used a map with interpolated groundwater heads of Baden-Wuerttemberg. The interpolation is based on measured groundwater heads.

- `input/groundwater_heads_interpolated_50m.tif`: Interpolated groundwater heads of Dreisam-Moehlin-Neumagen catchment
- `input/boundary_conditions.nc`: Variable `constant_head_porous_aquifer` contains the groundwater heads of the constant head boundary condition derived by the interpolated groundwater heads.

## No flow boundaries
No flow boundary is set at the boundary to the black forest and Schoenberg. We assigned a no flow boundary to the Schoenberg due to very complex hydrogeologic conditions (e.g. layers of opalinus clay).

- `input/parameters_modflow.nc`: Variable `mask_schoenberg` defines the no flow boundary of the Schoenberg. 

## Wells
Drinking water wells are considered for groundwater extraction. We use average annual extraction rates based on the data provided by the drinking water suppliers

- `input/groundwater_extraction.gpkg`: Contains locations of groundwater extraction and annual extraction volumes

## Drainage area
Drainge is considered for former wetlands close to Tuniberg. We assumed a single drainage pipe pipe per grid cell. The drainge pipe have the following properties: gradient of 1%, 50 m length, 0.3 diameter and a hydraulic conductivity of $10^{-1}$ m/s.

- `input/mask_former_tuniberg_wetland.tif`: Area of the former Tuniberg wetland (The area was digitzed manually from OSM)
- `input/land_use.tif`: Raster with RoGeR land use ID 

## Groundwater - surface water interaction
Surface waters are described by river network using Amtliche Digitale Wasserwirtschaftliches Gewässernetz (AWGN) geodataset provided by Landesanstalt fuer Umwelt Baden-Wuerttemberg (LUBW).

- `input/streambed_conductivity.tif`: Streambed conductivity (m/s) as raster
- `input/streambed_structure.tif`: Streambed structure as raster (4= ; 5= ; 6= ; 7=)
- `input/fraction_of_channelisation.tif`: Fraction of channelisation as raster
- `input/manning.tif`: Manning coefficient as raster
- `input/streamflow_observation_points.csv`: Coordniates and raster cells to collect simulated streamflow data in MODFLOW6.

### Prepare data for streamflow routing (SFR package of MODFLOW)
MODFLOW requires river reaches. However, river network data is not divided into reaches. In order to provide the required reach data the `sfrmaker` tool can be used. To calculate the river reach data, the `sfrmaker` tool requires the river network to divided into river segments and width and elevation information at the downstream and upstream nodes. The following steps describe the preparation of the data required by of SFR package of MODFLOW:

1. Use `Processing toolbox --> Vector overlay --> Split with lines` in QGIS to split the river network into river segments
2. Extract the downstream and upstream nodes from the river segments.
3. Assign the elevations and streambed width to the downstream and upstream nodes and join the data to the river segments.
4. Assign to each river segment the downstream connected segment.
5. Repair the geometries of the river segments.
6. Write the data for the SFR package of MODFLOW using the `sfrmaker` tool.
7. Check `input/model_SFR.chk` for errors and load `input/sfr_routing.shp` and `input/sfr_outlets.shp` in QGIS to make visual check.
8. Assign the hydraulic conductivity of the streambed and Manning's coeffecient to the SFR packagedata.
9. Identify the diversions (i.e. river reaches with with downstream connections).
10. Identify the reach IDs to collect the water depth and downstream flow.

## Observations

- `observed_groundwater_heads_avg.csv`: Average groundwater heads of the period 2013 - 2023
- `observed_streamflow.csv`: Average streamflow and water depth of the period 2013 - 2023

Workflow:
1. Run `preprocess_data_for_modflow_state_optimisation.sh`
2. Run `run_optimisation.sh`
3. Run `run_evaluation.sh`

Coordinate system:
EPSG: 25832
proj=utm +zone=32 +ellps=GRS80 +units=m +no_defs
