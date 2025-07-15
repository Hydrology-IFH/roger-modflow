import numpy as np
import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.ops import unary_union
import shapely
from shapely.geometry import LineString
from shapely.geometry import box
import xarray as xr
import flopy
import sfrmaker
from sfrmaker import StructuredGrid

base_path = Path(__file__).parent

path = Path(__file__).parent / "boundary_conditions.nc"
ds_bc = xr.open_dataset(path, engine="h5netcdf")

path = Path(__file__).parent / "parameters_modflow.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")
xoff = float(ds_params['spatial_ref'].GeoTransform.split(' ')[0])
yoff = float(ds_params['spatial_ref'].GeoTransform.split(' ')[3])

topography = ds_params['elevations'].isel(z=0).values
elevation_bottom_layer1 = ds_params['elevations'].isel(z=1).values
elevation_bottom_layer2 = ds_params['elevations'].isel(z=2).values
elevation_bottom_layer3 = ds_params['elevations'].isel(z=3).values
elevation_bottom_layer4 = ds_params['elevations'].isel(z=4).values
elevation_bottom_layers = np.stack([elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4], axis=0)

mask = np.isfinite(topography)
# set Schoenberg to inactive
mask_schoenberg = (ds_params['mask_schoenberg'].values == 1)
mask = np.where(mask_schoenberg, False, mask)
mask_boundary_condition_schoenberg = ds_bc['mask_schoenberg_bc'].values
mask = np.where(mask_boundary_condition_schoenberg, True, mask)

# define the domain for the model
domain = np.empty_like(mask)
domain[mask] = 1
domain[~mask] = -1
domain_layers = np.stack([domain, domain, domain, domain], axis=0)


# Create the Flopy simulation object
sim = flopy.mf6.MFSimulation(
    sim_name='model', exe_name="mf6", version="mf6", sim_ws=str(base_path / "output"),
)

# Create the Flopy temporal discretization object
tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
    sim, pname="tdis", time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1)]
)

# Create the Flopy groundwater flow (gwf) model object
model_nam_file = "model.nam"
gwf = flopy.mf6.ModflowGwf(sim, modelname="model", model_nam_file=model_nam_file, newtonoptions="NEWTON UNDER_RELAXATION", save_flows=True)

dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
    gwf,
    pname="dis",
    nlay=4,
    nrow=topography.shape[0],
    ncol=topography.shape[1],
    delr=50, 
    delc=50,
    length_units="METERS",
    top=topography,
    botm=elevation_bottom_layers,
    idomain=domain_layers,
    xorigin=xoff,
    yorigin=yoff - (50 * topography.shape[0]),  
)

# define the model grid
delr = np.array([50.] * 777)  # cell spacing along a row
delc = np.array([50.] * 621)  # cell spacing along a column
flopy_grid = flopy.discretization.StructuredGrid(delr=delr, delc=delc,
                                                 xoff=xoff, yoff=yoff - (50 * 621),  # lower left corner of model grid
                                                 angrot=0,  # grid is unrotated
                                                 crs=25832,
                                                 idomain=domain_layers,
                                                 lenuni='METERS',
                                                 top=topography,
                                                 botm=elevation_bottom_layers
                                                 )

cellsize = 50  # Breite/Höhe einer Rasterzelle (z.B. Meter)

data = []
for row in range(mask.shape[0]):
    for col in range(mask.shape[1]):
        value = mask[row, col]
        if value:
            minx = xoff + col * cellsize
            maxx = xoff + (col + 1) * cellsize
            miny = yoff - (row + 1) * cellsize
            maxy = yoff - row * cellsize

            geom = box(minx, miny, maxx, maxy)
            data.append(geom)

merged = unary_union(data) 
gdf = gpd.GeoDataFrame([{'geometry': merged}], crs="EPSG:25832")
file = base_path.parent / 'input' / 'streamflow_routing' / 'active_area_grid.shp'
gdf.to_file(file)

_domain = np.empty_like(mask)
_domain[mask] = 1
_domain[~mask] = 0

# repair the geometries of the shapefile with river segment
df = gpd.read_file(base_path.parent / 'input' / 'streamflow_routing' / 'awgn_gew.shp')
for i in range(len(df)):
    geom = df.geometry[i]
    if geom:
        if geom.geom_type == 'MultiLineString':
                df.at[i, 'geometry'] = LineString(gpd.GeoSeries(geom).get_coordinates())
cond = df.geometry.geom_type == 'LineString'
df = df[cond]  # filter out None geometries
df.index = range(len(df))  # reset the index
for i in range(len(df)):
    geom = df.geometry[i]
    if shapely.is_valid(geom):  # check if the geometry is valid
        line = shapely.wkt.loads(str(geom)).reverse()  # reverse the line direction
        df.at[i, 'geometry'] = line
    else:
        print(f"Invalid geometry at index {i}: {geom}")
        geom_repaired = shapely.make_valid(geom, method="structure", keep_collapsed=True)
        line = shapely.wkt.loads(str(geom_repaired)).reverse()  # reverse the line direction
        df.at[i, 'geometry'] = line

# write the modified shapefile with reversed lines
df.to_file(base_path.parent / 'input' / 'streamflow_routing' / 'awgn_gew_repaired.shp', driver='ESRI Shapefile')

# load shapefile with river segments
custom_segments = sfrmaker.Lines.from_shapefile(shapefile=base_path.parent / 'input' / 'streamflow_routing' / 'awgn_gew_repaired.shp',
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
# # remove segments with no geometry
# custom_segments.df = custom_segments.df[custom_segments.df.geometry.notnull()]
# file_active_area = base_path.parent / 'input' / 'streamflow_routing' / 'active_area_grid.shp'
# grid = StructuredGrid.from_modelgrid(flopy_grid, crs=25832, active_area=file_active_area)
# reach_data = custom_segments.intersect(grid)
# custom_segments.make_routing_one_to_one()
# custom_segments.write_shapefile(outshp=base_path / 'output' / 'flowlines.shp')  

# make the data for the SFR package
file_active_area = base_path.parent / 'input' / 'streamflow_routing' / 'active_area_grid.shp'
sfrdata = custom_segments.to_sfr(grid=flopy_grid, model=gwf, active_area=file_active_area, 
                                 model_length_units='meters', consolidate_conductance=True, add_outlets=[4328, 11391, 11157, 3766, 11151, 11279, 3878, 11279, 8118])

# reverse the reach numbers for each river segment
# line_ids = np.unique(sfrdata.reach_data['line_id'].values).tolist()
# for line_id in line_ids:
#     cond = sfrdata.reach_data['line_id'] == line_id
#     ids = sfrdata.reach_data.loc[cond, 'rno'].values
    # values_rev = sfrdata.reach_data[cond].copy().values[::-1, :]
    # values_rev[:, 0] = ids  # reverse the reach numbers
    # sfrdata.reach_data.loc[cond, :] = values_rev
    # sfrdata.reach_data.loc[cond, 'ireach'] = np.arange(1, len(ids) + 1)[::-1]
# sfrdata.reach_data.sort_values(by='rno', ascending=True)
# sfrdata._reset_routing()
# sfrdata.set_outreaches()  # set the next downstream reach for each reach

# modify reach data
cond = np.isnan(sfrdata.reach_data['width'])
sfrdata.reach_data.loc[cond, 'width'] = 1.0  # set width to 1 m where it is NaN
cond_widht0 = (sfrdata.reach_data.loc[:, 'width'] <= 1.0)
sfrdata.reach_data.loc[cond_widht0, 'width'] = 1.0  # set width to 1 m if it is smaller than 1 m
sfrdata.reach_data.loc[:, 'strthick'] = 1  # set the stream thickness (in meters)
sfrdata.reach_data.loc[:, 'strhc1'] = 1.0  # set the streambed hydraulic conductivity (in meters per day)
sfrdata.reach_data.loc[:, 'thts'] = 0.035  # set the Manning's roughness coefficient (dimensionless)

# set the streambed top elevations from the 5 m x 5 m DEM 
dem_file = base_path.parent / 'input' / 'streamflow_routing' / 'dem_5m.tif'
sfrdata.set_streambed_top_elevations_from_dem(dem_file,
                                              elevation_units='meters',
                                              method='buffers',
                                              smooth=False,
                                              buffer_distance=100)
sfrdata.update_slopes(default_slope=0.001, minimum_slope=0.0001, maximum_slope=0.99)  # update slopes based on the new streambed top elevations

# assign the layer
for rno, i, j in zip(sfrdata.reach_data['rno'], sfrdata.reach_data['i'], sfrdata.reach_data['j']):
    cond = (sfrdata.reach_data['rno'] == rno)
    streambed_top = sfrdata.reach_data.loc[cond, 'strtop'].values[0]
    if streambed_top >= topography[i, j]:
        sfrdata.reach_data.loc[cond, 'strtop'] = topography[i, j]  # set the streambed top elevation to the topography elevation
    streambed_bottom = sfrdata.reach_data.loc[cond, 'strtop'].values[0] - sfrdata.reach_data.loc[cond, 'strthick'].values[0]
    if streambed_bottom <= topography[i, j] and streambed_bottom > elevation_bottom_layer1[i, j]:
        k = 0
    elif streambed_bottom <= elevation_bottom_layer1[i, j] and streambed_bottom > elevation_bottom_layer2[i, j]:
        k = 1
    elif streambed_bottom <= elevation_bottom_layer2[i, j] and streambed_bottom > elevation_bottom_layer3[i, j]:
        k = 2
    elif streambed_bottom <= elevation_bottom_layer3[i, j] and streambed_bottom > elevation_bottom_layer4[i, j]:
        k = 3
    sfrdata.reach_data.loc[cond, 'k'] = k  # set the layer index for each reach

# write the SFR package  
sfrdata.write_package(version='mf6', idomain=domain_layers)
sfrdata.write_tables(str(base_path / "output" / "output"))
sfrdata.write_shapefiles(str(base_path / "output" / "output"))