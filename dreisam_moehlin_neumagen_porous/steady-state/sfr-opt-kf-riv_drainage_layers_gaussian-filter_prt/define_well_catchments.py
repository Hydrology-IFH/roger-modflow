from pathlib import Path
import flopy
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import shapely
import numpy as np
import pickle
import os

base_path = Path(__file__).parent
base_path_external = Path("/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/steady-state/sfr-opt-kf-riv_drainage_layers_gaussian-filter_prt")

# load the MODFLOW 6 model using pickle
with open(base_path_external / "output" / "dmn_run_1806.pkl", "rb") as f:
    gwfsim = pickle.load(f)
gwf = gwfsim.get_model("dmn_run_1806")
grid = gwf.modelgrid

# load groundwater extraction wells
path = base_path.parent / "input" / "groundwater_extraction.gpkg"
gw_extraction_wells = gpd.read_file(path)

well_ids = list(range(len(gw_extraction_wells["ID"].values)))
well_names = gw_extraction_wells["ID"].values.astype(str).tolist()
# replace / and . in well names with _ and - respectively to avoid issues with file names
well_names = [well_name.replace("/", "_").replace(".", "-") for well_name in well_names]
# well_ids = [5]
# well_names = ["A2"]
release_scenarios = ["near_surface", "pump_installation_depth", "deep"]
for release_scenario in release_scenarios:
    for well_id, well_name in zip(well_ids[:11], well_names[:11]):
        try:
            os.remove(base_path_external / "output" / f"{release_scenario}" / f"well_{well_id}" / f"well{well_id}_mp7.timeseries")
        except FileNotFoundError:
            pass
        # load mp7 pathline results
        plf = flopy.utils.PathlineFile(base_path_external / "output" / f"{release_scenario}" / f"well_{well_id}" / f"well{well_id}_mp7.mppth")
        pl = pd.DataFrame(
            plf.get_destination_pathline_data(range(grid.nnodes), to_recarray=True)
        )
        pl["x"] = grid.xoffset + pl["x"].values
        pl["y"] = grid.yoffset + pl["y"].values
        cond_time = (pl["time"] < (365.25 * 5))
        pl = pl[cond_time]

        coords = pl[["x", "y"]].values
        # remove duplicate coordinates
        coords = np.unique(coords, axis=0)
        polygon_ = Polygon(coords)
        # make envelope of the polygon
        polygon = shapely.concave_hull(polygon_, ratio=0.3)
        gdf = gpd.GeoDataFrame(
        {"well_catchment": [f"{well_name}"]},
        geometry=[polygon],
        crs="EPSG:25832"
        )
        gdf.to_file(base_path_external / "output" / f"{release_scenario}" / f"well_{well_id}" / f"well{well_name}_catchment.gpkg", layer=f"well{well_name}_catchment", driver="GPKG")

        pl_zone2 = pl[pl["time"] <= 50]
        coords = pl_zone2[["x", "y"]].values
        # remove duplicate coordinates
        coords = np.unique(coords, axis=0)
        polygon_zone2_ = Polygon(coords)
        # make polygon convex hull
        polygon_zone2 = polygon_zone2_.convex_hull
        gdf = gpd.GeoDataFrame(
        {"zone2": [f"{well_name}"]},
        geometry=[polygon_zone2],
        crs="EPSG:25832"
        )
        gdf.to_file(base_path_external / "output" / f"{release_scenario}" / f"well_{well_id}" / f"well{well_name}_zone2.gpkg", layer=f"well{well_name}_zone2", driver="GPKG")

        # make difference between the two polygons and remove zone2 from catchment
        polygon_zone3 = polygon.difference(polygon_zone2)
        gdf = gpd.GeoDataFrame(
        {"zone3": [f"{well_name}"]},
        geometry=[polygon_zone3],
        crs="EPSG:25832"
        )
        gdf.to_file(base_path_external / "output" / f"{release_scenario}" / f"well_{well_id}" / f"well{well_name}_zone3.gpkg", layer=f"well{well_name}_zone3", driver="GPKG")