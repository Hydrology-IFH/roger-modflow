# Backward particle tracking using the steady-state groundwater model of porous Aquifer in the Dreisam-Moehlin-Neumagen catchment

A backward particle tracking using MODPATH 7 is applied to every groundwater extraction well. The particles are released in circles with  radii of 1m, 5m, 10m and 25m. The following three release scenarios are considered:
- near-surface: particles released within the upper 10% of the screen length
- pump_installation_depth: particles from top of the screen until installation depth of the extraction pump
- deep: particles released over the entire screen length


## Files and folders
- `modflow6_modpath7_steady-state.py`: Backward particle tracking using MODPATH 7 for each groundwater extraction well
- `define_well_catchments.py`: Convert pathlines of the particles to polygons
- `write_backward_travel_times.py`: Convert pathlines of the particles to polygons
- `output/near_surface`: Results of the near-surface release scenario
- `output/pump_installation_depth`: Results of the pump-installation-depth release scenario
- `output/deep`: Results of the deep release scenario
- `*_catchment.gpkg`: Entire catchment of the well
- `*_zone2.gpkg`: Zone with less than 50 days travel time
- `*_zone3.gpkg`: Zone with greater than 50 days travel time
- `*_release_points.gpkg`: Coordinates of the particle release points 
- `*_backward_travel_time.csv`: Travel times and coordinates of the end points of the released particles
- `../input/groundwater_extraction.csv`: Well locations and average extraction rates
- `../input/parameters_modflow.nc`: Hydraulic conductivities and elevations of the 4 model layers


## Workflow
1. `mamba env create --file=conda-environment.yml`: Install the required Python packages into a anaconda environment
2. `conda activate roger-modflow`: Activate the anaconda environment
3. `python modflow6_modpath7_steady-state.py`: Run the backward particle tracking
4. `python define_well_catchments.py`: Define the well catchments by using a convex hull og the pathlines of the backward particle tracking
5. `python write_backward_travel_times.py`: Collect the backward travel times of each released particle. 
