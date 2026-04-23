from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
import xarray as xr

base_path = Path(__file__).parent

# load MODFLOW parameters
path = base_path / "input" / "parameters_modflow_.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")

topography = ds_params["elevations"].isel(z=0).values
hydraulic_conductivities_layer1 = ds_params["kf"].isel(layer=0).values
hydraulic_conductivities_layer2 = ds_params["kf"].isel(layer=1).values
hydraulic_conductivities_layer3 = ds_params["kf"].isel(layer=2).values
hydraulic_conductivities_layer4 = ds_params["kf"].isel(layer=3).values

# load the sfr packagedata
gdf_reaches = gpd.read_file(base_path / "input" / "sfr_lines.shp")
df_reaches = pd.read_csv(base_path / "input" / "sfr_packagedata.csv", sep=";")
gdf_reaches = gdf_reaches[["rno", "k", "i", "j", "rchlen", "width", "slope", "strtop", "strthick", "strhc1", "thts", "line_id", "geometry"]]
gdf_reaches.columns = ["rno", "k", "i", "j", "rlen", "rwid", "rgrd", "rtp", "rbth", "rhk", "man", "line_id", "geometry"]
gdf_reaches["ncon"] = df_reaches["ncon"]
gdf_reaches["ustrf"] = df_reaches["ustrf"]
gdf_reaches["ndv"] = df_reaches["ndv"]

gdf_reaches["k"] = gdf_reaches["k"]
gdf_reaches["i"] = gdf_reaches["i"]
gdf_reaches["j"] = gdf_reaches["j"]

# load the manning coefficient raster
base_path = Path(__file__).parent
src = rasterio.open(str(base_path / "input" / "manning.tif"))
manning = src.read(1)

# load the streambed conductivity raster
base_path = Path(__file__).parent
src = rasterio.open(str(base_path / "input" / "streambed_conductivity.tif"))
streambed_conductivity = src.read(1)

# load the fraction of channelisation raster
base_path = Path(__file__).parent
src = rasterio.open(str(base_path / "input" / "fraction_of_channelisation.tif"))
fraction_of_channelisation = src.read(1)

# load the streambed structure raster
base_path = Path(__file__).parent
src = rasterio.open(str(base_path / "input" / "streambed_structure.tif"))
streambed_structure = src.read(1)

gdf_reaches["rhk"] = np.nan
gdf_reaches["man"] = np.nan
gdf_reaches["fc"] = 0.
gdf_reaches["ss"] = 0
gdf_reaches["kf"] = np.nan
gdf_reaches["topo50"] = np.nan
gdf_reaches["topo50-rtp"] = np.nan

for rno, z, y, x in zip(gdf_reaches.loc[:, "rno"] - 1, gdf_reaches.loc[:, "k"], gdf_reaches.loc[:, "i"], gdf_reaches.loc[:, "j"]):
    if z == 0:
        gdf_reaches.loc[rno, "kf"] = hydraulic_conductivities_layer2[y, x] / 86400
    elif z == 1:
        gdf_reaches.loc[rno, "kf"] = hydraulic_conductivities_layer2[y, x] / 86400
    elif z == 2:
        gdf_reaches.loc[rno, "kf"] = hydraulic_conductivities_layer3[y, x] / 86400
    elif z == 3:
        gdf_reaches.loc[rno, "kf"] = hydraulic_conductivities_layer3[y, x] / 86400
    gdf_reaches.loc[rno, "topo50"] = topography[y, x]

gdf_reaches["topo50-rtp"] = gdf_reaches["topo50"] - gdf_reaches["rtp"]

# assign manning coefficient and streambed hydraulic conductivity to each reach based on the rasters
for idx, row in gdf_reaches.iterrows():
    if np.isnan(streambed_structure[int(row["i"]), int(row["j"])]):
        ss = 0
    else:
        ss = streambed_structure[int(row["i"]), int(row["j"])]
    if np.isnan(fraction_of_channelisation[int(row["i"]), int(row["j"])]):
        fc = 0
    else:
        fc = fraction_of_channelisation[int(row["i"]), int(row["j"])]
    gdf_reaches.loc[idx, "rhk"] = streambed_conductivity[int(row["i"]), int(row["j"])]
    gdf_reaches.loc[idx, "man"] = manning[int(row["i"]), int(row["j"])]
    gdf_reaches.loc[idx, "fc"] = fc
    gdf_reaches.loc[idx, "ss"] = ss

gdf_reaches["rbth"] = 1.0
# modify the manning"s n and hydraulic conductivity of the streambed based on the fraction of channelisation
gdf_reaches["man"] = (1 - gdf_reaches["fc"]) * gdf_reaches["man"]
gdf_reaches["rhk"] = (1 - gdf_reaches["fc"]) * gdf_reaches["rhk"]

# modify the manning"s n and hydraulic conductivity of the streambed based on the degree of alteration (5=partly, 6=strongly, 7=very strongly)
cond = (gdf_reaches["ss"] == 5)
gdf_reaches.loc[cond, "rhk"] = 50e-7
cond = (gdf_reaches["ss"] == 6)
gdf_reaches.loc[cond, "rhk"] = 10e-9
cond = (gdf_reaches["ss"] == 7)
gdf_reaches.loc[cond, "rhk"] = 50e-10

# set lower limits for manning"s n and hydraulic conductivity of the streambed
cond = (gdf_reaches["man"] <= 0.0)
gdf_reaches.loc[cond, "man"] = gdf_reaches["man"].median()
cond = (gdf_reaches["rhk"] <= 10e-9)
gdf_reaches.loc[cond, "rhk"] = 10e-9
gdf_reaches.loc[:, "rhk"] = gdf_reaches["rhk"].interpolate()

gdf_reaches["k"] = gdf_reaches["k"] + 1
gdf_reaches["i"] = gdf_reaches["i"] + 1
gdf_reaches["j"] = gdf_reaches["j"] + 1

gdf_reaches = gdf_reaches[["rno", "k", "i", "j", "rlen", "rwid", "rgrd", "rtp", "rbth", "rhk", "man", "ncon", "ustrf", "ndv", "line_id", "fc", "ss", "kf", "topo50", "topo50-rtp", "geometry"]]

file = base_path / "input" / "sfr_packagedata_modified.gpkg"
gdf_reaches.to_file(file, driver="GPKG")
file = base_path / "input" / "sfr_packagedata_modified.csv"
gdf_reaches.drop('geometry', axis=1).to_csv(file, sep=";", index=False)