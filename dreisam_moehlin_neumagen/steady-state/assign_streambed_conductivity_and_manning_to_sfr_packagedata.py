from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio

base_path = Path(__file__).parent

# load the sfr packagedata
df_reaches = pd.read_csv(base_path / "input" / "sfr_packagedata.csv", sep=";")
gdf_reaches = gpd.read_file(base_path / "input" / "sfr_lines.shp")

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

df_reaches["rhk"] = np.nan
df_reaches["man"] = np.nan
df_reaches["fc"] = 0.
df_reaches["ss"] = 0
gdf_reaches["rhk"] = np.nan
gdf_reaches["man"] = np.nan
gdf_reaches["fc"] = 0.
gdf_reaches["ss"] = 0

# assign manning coefficient and streambed hydraulic conductivity to each reach based on the rasters
for idx, row in df_reaches.iterrows():
    if np.isnan(streambed_structure[int(row["i"]) - 1, int(row["j"]) - 1]):
        ss = 0
    else:
        ss = streambed_structure[int(row["i"]) - 1, int(row["j"]) - 1]
    if np.isnan(fraction_of_channelisation[int(row["i"]) - 1, int(row["j"]) - 1]):
        fc = 0
    else:
        fc = fraction_of_channelisation[int(row["i"]) - 1, int(row["j"]) - 1]
    df_reaches.loc[idx, "rhk"] = streambed_conductivity[int(row["i"]) - 1, int(row["j"]) - 1]
    df_reaches.loc[idx, "man"] = manning[int(row["i"]) - 1, int(row["j"]) - 1]
    df_reaches.loc[idx, "fc"] = fc
    df_reaches.loc[idx, "ss"] = ss
    gdf_reaches.loc[idx, "rhk"] = streambed_conductivity[int(row["i"]) - 1, int(row["j"]) - 1]
    gdf_reaches.loc[idx, "man"] = manning[int(row["i"]) - 1, int(row["j"]) - 1]
    gdf_reaches.loc[idx, "fc"] = fc
    gdf_reaches.loc[idx, "ss"] = ss

df_reaches["rbth"] = 1.0

file = base_path / "input" / "sfr_packagedata_modified.csv"
df_reaches.to_csv(file, index=False, sep=";")
file = base_path / "input" / "sfr_packagedata_modified.gpkg"
gdf_reaches.to_file(file, driver="GPKG")