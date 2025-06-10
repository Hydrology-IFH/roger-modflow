import numpy as np
from pathlib import Path
import rasterio
import flopy
import sfrmaker

base_path = Path(__file__).parent

# load shapefile with river segments
custom_segments = sfrmaker.Lines.from_shapefile(shapefile=base_path.parent / 'input' / 'streamflow_routing' / 'awgn_gew.shp',
                                             id_column='GEW_ID',  # arguments to sfrmaker.Lines.from_shapefile
                                             routing_column='VOR_GEW_ID',
                                             width1_column='width_up',
                                             width2_column='width_dn',
                                             up_elevation_column='elev_up',
                                             dn_elevation_column='elev_dn',
                                             name_column='GEW_NAME',
                                             )

# set 0 width to 1 m
cond1 = custom_segments.df.width1 == 0
cond2 = custom_segments.df.width2 == 0
custom_segments.df.loc[cond1, 'width1'] = 1
custom_segments.df.loc[cond2, 'width2'] = 1

# load digital elevation model 50 m x 50 m (DEM) 
with rasterio.open(base_path.parent / 'input' / 'dem_dreisam_moehlin_neumagen_50m.tif') as dataset:
    xoff = dataset.bounds.left
    yoff = dataset.bounds.bottom
    dem = dataset.read(1)  # read the first band
    mask = np.isfinite(dem)

# define the domain for the model
domain = np.empty_like(dem)
domain[mask] = 1
domain[~mask] = -1
domain_layers = np.stack([domain, domain, domain, domain], axis=0)

# generate the model grid
sim = flopy.mf6.MFSimulation(version='mf6', exe_name='mf6',
                                         sim_ws=str(base_path))
m = flopy.mf6.ModflowGwf(sim)
m.set_model_relative_path('output')
# define the model grid
delr = np.array([50.] * 777)  # cell spacing along a row
delc = np.array([50.] * 621)  # cell spacing along a column
flopy_grid = flopy.discretization.StructuredGrid(delr=delr, delc=delc,
                                                 xoff=xoff, yoff=yoff,  # lower left corner of model grid
                                                 angrot=0,  # grid is unrotated
                                                 crs=25832
                                                 )

# make the data for the SFR package
file_active_area = base_path.parent / 'input' / 'streamflow_routing' / 'active_area.shp'
sfrdata = custom_segments.to_sfr(grid=flopy_grid, model=m, model_length_units='meters', active_area=file_active_area, consolidate_conductance=True)

# set the streambed top elevations from the 5 m x 5 m DEM 
dem_file = base_path.parent / 'input' / 'streamflow_routing' / 'dem_5m.tif'
sfrdata.set_streambed_top_elevations_from_dem(dem_file,
                                              elevation_units='meters',
                                              method='buffers',
                                              smooth=True,
                                              buffer_distance=100)

# modify reach data
cond = np.isnan(sfrdata.reach_data['width'])
sfrdata.reach_data.loc[cond, 'width'] = 1.0  # set width to 1 m where it is NaN
sfrdata.reach_data.loc[:, 'k'] = 2
sfrdata.reach_data.loc[:, 'strthick'] = 1  # set the stream thickness (in meters)
sfrdata.reach_data.loc[:, 'strhc1'] = 1.44  # set the streambed hydraulic conductivity (in meters per day)
sfrdata.reach_data.loc[:, 'thts'] = 0.035  # set the Manning's roughness coefficient (dimensionless)

# write the SFR package  
sfrdata.write_package(version='mf6', idomain=domain_layers)
sfrdata.write_tables(str(base_path / "output" / "output"))
sfrdata.write_shapefiles(str(base_path / "output" / "output"))

# add diversion data to the SFR package
# diversions are manually defined
with open(base_path / "output" / "model.sfr", "r") as f:
    lines = [line for line in f]

lines.append("\n")
lines.append("BEGIN diversions\n")
with open(base_path.parent / "input" / "streamflow_routing" / "diversions.txt") as f:
    for i, line in enumerate(f):
        if i >= 1:
            lines.append(line)
lines.append("\n")
lines.append("END diversions\n")

with open(base_path / "output" / "model.sfr", "w") as f:
    f.writelines(lines)