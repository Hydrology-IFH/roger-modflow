import numpy as np
import os
import pandas as pd
from pathlib import Path
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import box
import xarray as xr
import yaml
import flopy
import sfrmaker

base_path = Path(__file__).parent

# load config file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    modflow_config = yaml.safe_load(file)

path = base_path / "input" / "boundary_conditions.nc"
ds_bc = xr.open_dataset(path, engine="h5netcdf")

path = base_path / "input" / "parameters_modflow.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")
xoff = float(ds_params["spatial_ref"].GeoTransform.split(" ")[0])
yoff = float(ds_params["spatial_ref"].GeoTransform.split(" ")[3])

topography = ds_params["elevations"].isel(z=0).values
elevation_bottom_layer1 = ds_params["elevations"].isel(z=1).values
elevation_bottom_layer2 = ds_params["elevations"].isel(z=2).values
elevation_bottom_layer3 = ds_params["elevations"].isel(z=3).values
elevation_bottom_layer4 = ds_params["elevations"].isel(z=4).values
elevation_bottom_layers = np.stack([elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4], axis=0)

mask = (ds_params["mask_porous_aquifer"].values == 1)

# define the domain for the model
domain = np.empty_like(mask)
domain[mask] = 1
domain[~mask] = -1
domain_layers = np.stack([domain, domain, domain, domain], axis=0)


# Create the Flopy simulation object
sim = flopy.mf6.MFSimulation(
    sim_name="model", exe_name="mf6", version="mf6", sim_ws=str(base_path / "input"),
)

# Create the Flopy temporal discretization object
tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
    sim, pname="tdis", time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1)]
)

# Create the Flopy groundwater flow (gwf) model object
model_nam_file = "model.nam"
gwf = flopy.mf6.ModflowGwf(sim, modelname="model", model_nam_file=model_nam_file, newtonoptions="NEWTON", save_flows=True)

dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
    gwf,
    pname="dis",
    nlay=modflow_config["nz"],
    nrow=topography.shape[0],
    ncol=topography.shape[1],
    delr=modflow_config["dy"], 
    delc=modflow_config["dx"],
    length_units="METERS",
    top=topography,
    botm=elevation_bottom_layers,
    idomain=domain_layers,
    xorigin=xoff,
    yorigin=yoff - (modflow_config["dy"] * topography.shape[0]),  
)

# define the model grid
delr = np.array([modflow_config["dy"]] * modflow_config["ny"])  # cell spacing along a row
delc = np.array([modflow_config["dx"]] * modflow_config["nx"])  # cell spacing along a column
flopy_grid = flopy.discretization.StructuredGrid(delr=delr, delc=delc,
                                                 xoff=xoff, yoff=yoff - (modflow_config["dy"] * topography.shape[0]),  # lower left corner of model grid
                                                 angrot=0,  # grid is unrotated
                                                 crs=25832,
                                                 idomain=domain_layers,
                                                 lenuni="METERS",
                                                 top=topography,
                                                 botm=elevation_bottom_layers
                                                 )

cellsize = modflow_config["dx"]  # Breite/Höhe einer Rasterzelle (z.B. Meter)

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
gdf = gpd.GeoDataFrame([{"geometry": merged}], crs="EPSG:25832")
file = base_path / "input" / "active_area_grid.shp"
gdf.to_file(file)

_domain = np.empty_like(mask)
_domain[mask] = 1
_domain[~mask] = 0

# load shapefile with river segments
custom_segments = sfrmaker.Lines.from_shapefile(shapefile=base_path / "input" / "awgn_stream_segments_connected_repaired_corrected-width.shp",
                                             id_column="segment",  # arguments to sfrmaker.Lines.from_shapefile
                                             routing_column="to_segment",
                                             width1_column="width_up",
                                             width2_column="width_dn",
                                             up_elevation_column="elev_up",
                                             dn_elevation_column="elev_dn",
                                             )


# set 0 width to 1 m
cond1 = custom_segments.df.width1 == 0
cond2 = custom_segments.df.width2 == 0
custom_segments.df.loc[cond1, "width1"] = 1
custom_segments.df.loc[cond2, "width2"] = 1

# make the data for the SFR package
file_active_area = base_path / "input" / "active_area_grid.shp"
sfrdata = custom_segments.to_sfr(grid=flopy_grid, model=gwf,
                                 model_length_units="meters", consolidate_conductance=True, one_reach_per_cell=False)

# modify reach data
sfrdata.reach_data.loc[:, "width"] = sfrdata.reach_data.loc[:, "width"]
cond = np.isnan(sfrdata.reach_data["width"])
sfrdata.reach_data.loc[cond, "width"] = 1.0  # set width to 1 m where it is NaN
cond_widht0 = (sfrdata.reach_data.loc[:, "width"] <= 1.0)
sfrdata.reach_data.loc[cond_widht0, "width"] = 1.0  # set width to 1 m if it is smaller than 1 m
cond_widht18 = (sfrdata.reach_data.loc[:, "width"] >= 18.0)
sfrdata.reach_data.loc[cond_widht18, "width"] = 18.0  # set width to 18 m if it is larger than 18 m
sfrdata.reach_data.loc[:, "strthick"] = 1  # set the stream thickness (in meters)
sfrdata.reach_data.loc[:, "strhc1"] = 1.0  # set the streambed hydraulic conductivity (in meters per day)
sfrdata.reach_data.loc[:, "thts"] = 0.035  # set the Manning"s roughness coefficient (dimensionless)

# set the streambed top elevations from the 5 m x 5 m DEM 
dem_file = base_path / "input" / "dem_5m.tif"
sfrdata.set_streambed_top_elevations_from_dem(dem_file,
                                              elevation_units="meters",
                                              method="buffers",
                                              smooth=True,
                                              buffer_distance=100)
sfrdata.update_slopes(default_slope=0.01, minimum_slope=0.001, maximum_slope=0.45)  # update slopes based on the new streambed top elevations

sfrdata.reach_data.loc[:, "width"] = sfrdata.reach_data.loc[:, "width"]
cond = np.isnan(sfrdata.reach_data["width"])
sfrdata.reach_data.loc[cond, "width"] = 1.0  # set width to 1 m where it is NaN
cond_widht0 = (sfrdata.reach_data.loc[:, "width"] <= 1.0)
sfrdata.reach_data.loc[cond_widht0, "width"] = 1.0  # set width to 1 m if it is smaller than 1 m
cond_widht18 = (sfrdata.reach_data.loc[:, "width"] >= 18.0)
sfrdata.reach_data.loc[cond_widht18, "width"] = 18.0  # set width to 18 m if it is larger than 18 m

# assign the layer
for rno, i, j in zip(sfrdata.reach_data["rno"], sfrdata.reach_data["i"], sfrdata.reach_data["j"]):
    cond = (sfrdata.reach_data["rno"] == rno)
    streambed_top = sfrdata.reach_data.loc[cond, "strtop"].values[0]
    if streambed_top >= topography[i, j]:
        sfrdata.reach_data.loc[cond, "strtop"] = topography[i, j]  # set the streambed top elevation to the topography elevation
    streambed_bottom = sfrdata.reach_data.loc[cond, "strtop"].values[0] - sfrdata.reach_data.loc[cond, "strthick"].values[0]
    if streambed_bottom <= topography[i, j] and streambed_bottom > elevation_bottom_layer1[i, j]:
        k = 0
    elif streambed_bottom <= elevation_bottom_layer1[i, j] and streambed_bottom > elevation_bottom_layer2[i, j]:
        k = 1
    elif streambed_bottom <= elevation_bottom_layer2[i, j] and streambed_bottom > elevation_bottom_layer3[i, j]:
        k = 2
    elif streambed_bottom <= elevation_bottom_layer3[i, j] and streambed_bottom > elevation_bottom_layer4[i, j]:
        k = 3
    sfrdata.reach_data.loc[cond, "k"] = k  # set the layer index for each reach

# write the SFR package  
sfrdata.write_package(version="mf6", idomain=domain_layers)
sfrdata.write_tables(str(base_path / "input" / " "))
sfrdata.write_shapefiles(str(base_path / "input" / " "))

# collect files in input folder
input_files = list((base_path / "input").glob("*"))

# if ' _' in file rename file by removing ' _'
for file in input_files:
    if ' _' in file.name:
        new_name = file.name.replace(' _', '')
        os.rename(file, base_path / "input" / new_name)

# load model.sfr as text file
with open(base_path / "input" / "model.sfr", "r") as file:
    lines = file.readlines()
# find beginning and end of the packagedata block
for i, line in enumerate(lines):
    if line.strip().startswith("BEGIN Packagedata"):
        start_index = i
    if line.strip().startswith("END Packagedata"):
        end_index = i
        break

# extract the packagedata block and write to csv file
packagedata_lines = lines[start_index + 1:end_index]
packagedata = []
for line in packagedata_lines:
    if line.strip() and not line.strip().startswith("#"):
        parts = line.split()
        reach = [int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]),  int(parts[8]), float(parts[9]), float(parts[10]), int(parts[11]), float(parts[12]), int(parts[13]), int(parts[14])]
        packagedata.append(reach)
df_packagedata = pd.DataFrame(packagedata, columns=["rno", "k", "i", "j", "rbth", "rhk", "man", "ncon", "ustrf", "ndv", "line_id"])
df_packagedata.index = df_packagedata["rno"]

df = sfrdata.reach_data.loc[:, ["rchlen", "width", "slope", "strtop"]]
df.index = sfrdata.reach_data["rno"]
df.columns = ["rlen", "rwid", "rgrd", "rtp"]
df_packagedata = df_packagedata.join(df, how="left")

# reorder columns
df_packagedata = df_packagedata.loc[:, ["rno", "k", "i", "j", "rlen", "rwid", "rgrd", "rtp", "rbth",  "rhk", "man", "ncon", "ustrf", "ndv", "line_id"]]

# write to csv file
file = base_path / "input" / "sfr_packagedata.csv"
df_packagedata.to_csv(file, index=False, sep=";")

# write the connectiondata to csv file
for i, line in enumerate(lines):
    if line.strip().startswith("BEGIN Connectiondata"):
        start_index = i
    if line.strip().startswith("END Connectiondata"):
        end_index = i
        break

# extract the connectiondata block and write to csv file
connectiondata_lines = lines[start_index + 1:end_index]
connectiondata = []
for line in connectiondata_lines:
    if line.strip() and not line.strip().startswith("#"):
        parts = line.split()
        connectiondata.append(parts)
df_connectiondata = pd.DataFrame(connectiondata)
file = base_path / "input" / "sfr_connectiondata.csv"
df_connectiondata.to_csv(file, index=False, header=False, sep=";")
