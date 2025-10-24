from pathlib import Path
import geopandas as gpd

base_path = Path(__file__).parent

outlet_segment_ids = [137, 792, 603, 282, 806, 882, 463]

#TODO: generate stream segments in QGIS ("Split with lines") by splitting the river network at confluences and outlets 
# identify and assign the downstream segments
path = base_path / "input" / "awgn_stream_segments.shp"
stream_segments = gpd.read_file(path)
records = []
for idx, row in stream_segments.iterrows():
    geom = row.geometry
    # Handle both LineString and MultiLineString
    if geom.geom_type == "LineString":
        segs = [geom]
    elif geom.geom_type == "MultiLineString":
        segs = list(geom.geoms)
    else:
        continue

    for seg in segs:
        start = tuple(seg.coords[0])
        end = tuple(seg.coords[-1])
        records.append({"segment": row["id"], "start": start, "end": end, "width_dn": row["width_dn"], "width_up": row["width_up"], "elev_dn": row["elev_dn"], "elev_up": row["elev_up"], "geometry": seg})

gdf_segments = gpd.GeoDataFrame(records, crs=stream_segments.crs)
nodes = set(gdf_segments["start"]).union(set(gdf_segments["end"]))
node_list = list(nodes)
node_id_map = {coord: i for i, coord in enumerate(node_list)}

gdf_segments["to_node"] = gdf_segments["start"].map(node_id_map)
gdf_segments["from_node"] = gdf_segments["end"].map(node_id_map)

from_node_to_segment = (
    gdf_segments
    .sort_values("width_dn")  # if you want a consistent rule for branches (lowest ID, etc.)
    .groupby("from_node")["segment"]
    .apply(list)
    .to_dict()
)
outlet_set = set(outlet_segment_ids)

def find_downstream(row):
    # Force None for outlets
    if row["segment"] in outlet_set:
        return None
    # Candidates: all segments whose from_node is this segment"s to_node
    downstreams = from_node_to_segment.get(row["to_node"], [])
    # Exclude self (in case of self loops)
    downstreams = [sid for sid in downstreams if sid != row["segment"]]
    # Return the first (lowest, due to sort); change rule if needed
    return downstreams[0] if downstreams else None

gdf_segments["to_segment"] = gdf_segments.apply(find_downstream, axis=1)

# enforce Dreisam
cond = (gdf_segments["segment"] == 810)
gdf_segments.loc[cond, "to_segment"] = 809
cond = (gdf_segments["segment"] == 808)
gdf_segments.loc[cond, "to_segment"] = 807
# enforce Brugga
cond = (gdf_segments["segment"] == 641)
gdf_segments.loc[cond, "to_segment"] = 640
# enforce Moehlin
cond = (gdf_segments["segment"] == 144)
gdf_segments.loc[cond, "to_segment"] = 143
cond = (gdf_segments["segment"] == 766)
gdf_segments.loc[cond, "to_segment"] = 97
cond = (gdf_segments["segment"] == 9)
gdf_segments.loc[cond, "to_segment"] = 8

# fill NaN values in "to_segment" with 0 (for outlets)
gdf_segments["to_segment"] = gdf_segments["to_segment"].fillna(0).astype(int)

# drop columns not needed anymore
gdf_segments = gdf_segments.drop(columns=["start", "end", "to_node", "from_node"])
# reorder columns
gdf_segments = gdf_segments[["segment", "to_segment", "width_dn", "width_up", "elev_dn", "elev_up", "geometry"]]
# limit width to a maximum of 20 m
cond1 = (gdf_segments["width_dn"] > 20)
cond2 = (gdf_segments["width_up"] > 20)
gdf_segments.loc[cond1, "width_dn"] = 20
gdf_segments.loc[cond2, "width_up"] = 20
# save the modified shapefile including downstream segments
gdf_segments.to_file(base_path / "input" / "awgn_stream_segments_connected.shp", driver="ESRI Shapefile")
gdf_segments.to_file(base_path / "input" / "awgn_stream_segments_connected.gpkg", driver="GPKG")
