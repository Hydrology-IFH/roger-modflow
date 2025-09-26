from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import yaml

base_path = Path(__file__).parent

file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    modflow_config = yaml.safe_load(file)

reach_outlet_ids = modflow_config["outlet_rnos"]

path = base_path / "input" / "sfr_lines.shp"
gdf_reaches = gpd.read_file(path)

start = []
end = []
for idx, row in gdf_reaches.iterrows():
    geom = row.geometry
    # Handle both LineString and MultiLineString
    if geom.geom_type == "LineString":
        segs = [geom]
    elif geom.geom_type == "MultiLineString":
        segs = list(geom.geoms)
    else:
        continue

    if segs:
        for seg in segs:
            _start = tuple(seg.coords[0])
            _end = tuple(seg.coords[-1])
            start.append(_start)
            end.append(_end)
    else:
        start.append(np.nan)
        end.append(np.nan)

gdf_reaches["start"] = start
gdf_reaches["end"] = end
# drop rows with NaN coordinates
gdf_reaches = gdf_reaches.dropna(subset=["start", "end"])

nodes = set(gdf_reaches["start"]).union(set(gdf_reaches["end"]))
node_list = list(nodes)
node_id_map = {coord: i for i, coord in enumerate(node_list)}

gdf_reaches["to_node"] = gdf_reaches["end"].map(node_id_map)
gdf_reaches["from_node"] = gdf_reaches["start"].map(node_id_map)

outlet_set = set(reach_outlet_ids)

# Create a mapping: from_node -> list of segment_ids starting at that node
from_node_map = (
    gdf_reaches
    .groupby("from_node")["rno"]
    .apply(list)
    .to_dict()
)

# Downstream assignment function for "one-to-multiple"
def find_downstream_reaches(row):
    if row["rno"] in outlet_set:
        return []  # outlets have no downstreams
    # All segments that have from_node == this segment"s to_node
    downstream = from_node_map.get(row["to_node"], [])
    # Remove self in case of loops
    downstream = [sid for sid in downstream if sid != row["rno"]]
    return downstream

# Apply to every row
gdf_reaches["downstream_reaches"] = gdf_reaches.apply(find_downstream_reaches, axis=1)

file = base_path / "input" / "sfr_diversions.gpkg"
gdf_reaches.to_file(file, driver="GPKG")

div_rno = []
div_iconr = []

for idx, row in gdf_reaches.iterrows():
    if len(row["downstream_reaches"]) > 1:
        for reach in row["downstream_reaches"]:
            if reach != row["outreach"]:
                div_rno.append(row["rno"])
                div_iconr.append(reach)

df_diversions = pd.DataFrame({"rno": div_rno, "iconr": div_iconr})
df_diversions["idv"] = 1
df_diversions["cprior"] = "FRACTION"
# reorder columns
df_diversions = df_diversions[["rno", "idv", "iconr", "cprior"]]
file = base_path / "input" / "sfr_diversions.csv"
df_diversions.to_csv(file, index=False, sep=";")