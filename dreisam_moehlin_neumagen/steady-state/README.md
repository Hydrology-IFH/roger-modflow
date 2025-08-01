# Dreisam-Moehlin-Neumagen catchment

Steady-state optimazation of different MODFLOW6 setups (i.e. different combinations of MODFLOW6 packages) of the Dreisam-Moehlin-Neumagen catchment.

## Topography

## Hydraulic conductivities

## Reacharge

## Constant head

## Wells

## Drainage area

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
