from pathlib import Path
import flopy
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
import pickle
import os

base_path = Path(__file__).parent

# load the MODFLOW 6 model using pickle
with open(base_path / "output" / "dmn_run_1806.pkl", "rb") as f:
    gwfsim = pickle.load(f)
gwf = gwfsim.get_model("dmn_run_1806")
grid = gwf.modelgrid

well_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
well_names = ["HU1", "HU2", "A4", "K5", "HU3", "A2", "K2", "B1", "A3", "S2", "B4", "C1"]
# well_ids = [5]
# well_names = ["A2"]
for well_id, well_name in zip(well_ids, well_names):
    os.remove(base_path / "output" / f"well{well_id}_mp7.timeseries")
    # load mp7 pathline results
    plf = flopy.utils.PathlineFile(base_path / "output" / f"well{well_id}_mp7.mppth")
    pl = pd.DataFrame(
        plf.get_destination_pathline_data(range(grid.nnodes), to_recarray=True)
    )
    pl["x"] = grid.xoffset + pl["x"].values
    pl["y"] = grid.yoffset + pl["y"].values
    # file = base_path / "output" / f"well{well_name}_mp7.csv"
    # pl.to_csv(file, index=False, sep=";")

    coords = pl[["x", "y"]].values
    # remove duplicate coordinates
    coords = np.unique(coords, axis=0)
    polygon = Polygon(coords)
    # make polygon convex hull
    polygon = polygon.convex_hull
    gdf = gpd.GeoDataFrame(
    {"well_catchment": [f"{well_name}"]},
    geometry=[polygon],
    crs="EPSG:25832"
    )
    gdf.to_file(base_path / "output" / f"well{well_name}_catchment.gpkg", layer=f"well{well_name}_catchment", driver="GPKG")

    pl_zone2 = pl[pl["time"] <= 50]
    coords = pl_zone2[["x", "y"]].values
    # remove duplicate coordinates
    coords = np.unique(coords, axis=0)
    polygon_zone2 = Polygon(coords)
    # make polygon convex hull
    polygon_zone2 = polygon_zone2.convex_hull
    gdf = gpd.GeoDataFrame(
    {"zone2": [f"{well_name}"]},
    geometry=[polygon_zone2],
    crs="EPSG:25832"
    )
    gdf.to_file(base_path / "output" / f"well{well_name}_zone2.gpkg", layer=f"well{well_name}_zone2", driver="GPKG")

    # make difference between the two polygons and remove zone2 from catchment
    polygon_zone3 = polygon.difference(polygon_zone2)
    gdf = gpd.GeoDataFrame(
    {"zone3": [f"{well_name}"]},
    geometry=[polygon_zone3],
    crs="EPSG:25832"
    )
    gdf.to_file(base_path / "output" / f"well{well_name}_zone3.gpkg", layer=f"well{well_name}_zone3", driver="GPKG")