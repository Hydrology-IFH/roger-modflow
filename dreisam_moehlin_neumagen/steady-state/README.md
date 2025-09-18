# Dreisam-Moehlin-Neumagen catchment

Steady-state optimisation of different MODFLOW6 setups (i.e. different combinations of MODFLOW6 packages) of the Dreisam-Moehlin-Neumagen catchment.

## Topography

- 50 m x 50 m digital elevation model is used to describe the surface topography
- 5 m x 5 m digital elevation model is used to describe the riverbed topography

## Hydrogeology and hydraulic conductivities
- The Hydrogeology is described by 4 layers. Thickness and hydraulic conductivities are derived from the LGRB geodataset "Hydrogeologischer Bau und Aquifereigenschaften der Lockergesteine im Oberrheingraben (Baden-Wuerttemberg)"
#TODO: Further description by Julian 

## Recharge
- The annual average recharge is derived by RoGeR simulations
#TODO: Further informations from Hannes are required (Period of simulation? Which model and assumptions?)

## Constant head
- Constant head is set at the boundary to the upper rhine aquifer. To define the depth of the constant head, we used a map with interpolated grounwater depths of Baden-Wuerttemberg 
#TODO: Further informations from Andreas S./Markus (Reference of the map)

## No flow boundaries
- No flow boundary is set at the boundary to the black forest and Schoenberg. We assigned a no flow boundary to the Schoenberg due to very complex hydrogeologic conditions (e.g. layers of opalinus clay)

## Wells
- Drinking water wells are considered for groundwater extraction. We use average annual extraction rates based on the data provided by the drinking water suppliers

## Drainage area
- Drainge is considered for former wetlands close to Tuniberg. We assumed a single drainage pipe pipe per grid cell. The drainge pipe have the following properties: gradient of 1%, 50 m length, 0.3 diameter and a hydraulic conductivity of $10^{-1}$ m/s.

## Groundwater - surface water interaction

## Observations


Workflow:
1. Run `plot_input_data.py`
2. Run the steady-state simulation `python modflow6_steady-state.py`. Steady-state simulation is required to set the boundary and initial conditions of the transient simulation. 
3. Run the transient simulation `python roger_modflow6_transient.py`
4. Run `python export_binary_output_to_netcdf.py`
5. Run `python plot_groundwater_heads.py`
6. Run `python plot_time_series.py`
7. Run `python make_gif.py`

Coordinate system:
proj=utm +zone=32 +ellps=GRS80 +units=m +no_defs
