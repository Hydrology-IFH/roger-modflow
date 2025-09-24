from pathlib import Path
import geopandas as gpd
import shapely
from shapely.geometry import LineString

base_path = Path(__file__).parent

# repair the geometries of the shapefile with river segment
df = gpd.read_file(base_path / "input" / "awgn_stream_segments_connected.shp")
for i in range(len(df)):
    geom = df.geometry[i]
    if geom:
        if geom.geom_type == "MultiLineString":
                df.at[i, "geometry"] = LineString(gpd.GeoSeries(geom).get_coordinates())
cond = df.geometry.geom_type == "LineString"
df = df[cond]  # filter out None geometries
df.index = range(len(df))  # reset the index
for i in range(len(df)):
    geom = df.geometry[i]
    if shapely.is_valid(geom):  # check if the geometry is valid
        line = shapely.wkt.loads(str(geom)).reverse()  # reverse the line direction
        # line = shapely.wkt.loads(str(geom))
        df.at[i, "geometry"] = line
    else:
        print(f"Invalid geometry at index {i}: {geom}")
        geom_repaired = shapely.make_valid(geom, method="structure", keep_collapsed=True)
        line = shapely.wkt.loads(str(geom_repaired)).reverse()  # reverse the line direction
        # line = shapely.wkt.loads(str(geom_repaired))
        df.at[i, "geometry"] = line

# write the modified shapefile with reversed lines
df.to_file(base_path / "input" / "awgn_stream_segments_connected_repaired.shp", driver="ESRI Shapefile")
