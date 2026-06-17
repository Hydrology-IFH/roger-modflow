from pathlib import Path
import flopy
import pandas as pd
import geopandas as gpd
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
well_ids = well_ids[:11]
well_names = [str(well_id).replace("/", "_").replace(".", "-") for well_id in gw_extraction_wells["ID"][:11]]
# well_ids = [5]
# well_names = ["A2"]
release_scenarios = ["near_surface", "pump_installation_depth", "deep"]
for release_scenario in release_scenarios:
    for well_id, well_name in zip(well_ids[:11], well_names[:11]):
        try:
            os.remove(base_path_external / "output" / f"{release_scenario}" / f"well{well_name}" / f"well{well_id}_mp7.timeseries")
        except FileNotFoundError:
            pass
        # load mp7 pathline results
        plf = flopy.utils.PathlineFile(base_path_external / "output" / f"{release_scenario}" / f"well{well_name}" / f"well{well_id}_mp7.mppth")
        pl = pd.DataFrame(
            plf.get_destination_pathline_data(range(grid.nnodes), to_recarray=True)
        )
        pl["x"] = grid.xoffset + pl["x"].values
        pl["y"] = grid.yoffset + pl["y"].values

        file = base_path_external / "output" / f"{release_scenario}" / f"well{well_name}" / f"well{well_name}_pathlines.csv"
        pl.to_csv(file, index=False, sep=";")
        gdf_pathlines = gpd.GeoDataFrame(pl, geometry=gpd.points_from_xy(pl["x"], pl["y"]), crs="EPSG:25832")
        file = base_path_external / "output" / f"{release_scenario}" / f"well{well_name}" / f"well{well_name}_pathlines.gpkg"
        gdf_pathlines.to_file(file, layer=f"well{well_name}_pathlines", driver="GPKG")

        # get unique particle IDs
        particle_ids = pl["particleid"].unique()

        df_backward_travel_times = pd.DataFrame(index=particle_ids, columns=["particleID", "x", "y", "backward_travel_time"])
        for particle_id in particle_ids[1:]:
            pl_particle = pl[pl["particleid"] == particle_id]
            backward_travel_time = pl_particle["time"].values[-1]
            x = pl_particle["x"].values[-1]
            y = pl_particle["y"].values[-1]
            z = pl_particle["z"].values[-1]
            df_backward_travel_times.loc[particle_id, "particleID"] = particle_id
            df_backward_travel_times.loc[particle_id, "x"] = x
            df_backward_travel_times.loc[particle_id, "y"] = y
            df_backward_travel_times.loc[particle_id, "z"] = z
            df_backward_travel_times.loc[particle_id, "backward_travel_time"] = backward_travel_time

        cond_nozeros = (df_backward_travel_times["backward_travel_time"] > 0)
        df_backward_travel_times = df_backward_travel_times.loc[cond_nozeros, :]
        gdf_backward_travel_times = gpd.GeoDataFrame(df_backward_travel_times, geometry=gpd.points_from_xy(df_backward_travel_times["x"], df_backward_travel_times["y"]), crs="EPSG:25832")

        file = base_path_external / "output" / f"{release_scenario}" / f"well{well_name}" / f"well{well_name}_backward_travel_times.csv"
        # rename index
        df_backward_travel_times.columns = [["", "[m]", "[m]", "[m a.s.l.]", "[days]"], ["particleID", "x", "y", "z", "backward_travel_time"]]
        # remove blank rows
        df_backward_travel_times = df_backward_travel_times.dropna(how="all")
        df_backward_travel_times.to_csv(file, index=False, sep=";")
        file = base_path_external / "output" / f"{release_scenario}" / f"well{well_name}" / f"well{well_name}_backward_travel_times.gpkg"
        gdf_backward_travel_times.to_file(file, layer=f"well{well_name}_backward_travel_times", driver="GPKG")

        df_release_points = pd.DataFrame(index=particle_ids, columns=["particleID", "x-coordinate", "y-coordinate", "elevation"])
        for particle_id in particle_ids[1:]:
            pl_particle = pl[pl["particleid"] == particle_id]
            x = pl_particle["x"].values[0]
            y = pl_particle["y"].values[0]
            elevation = pl_particle["z"].values[0]
            df_release_points.loc[particle_id, "particleID"] = particle_id
            df_release_points.loc[particle_id, "x-coordinate"] = x
            df_release_points.loc[particle_id, "y-coordinate"] = y
            df_release_points.loc[particle_id, "elevation"] = elevation

        # convert to geopandas dataframe
        df_release_points = df_release_points.dropna(how="all")
        df_release_points = df_release_points.loc[cond_nozeros, :]
        gdf_release_points = gpd.GeoDataFrame(df_release_points, geometry=gpd.points_from_xy(df_release_points["x-coordinate"], df_release_points["y-coordinate"]), crs="EPSG:25832")
        file = base_path_external / "output" / f"{release_scenario}" / f"well{well_name}" / f"well{well_name}_release_points.gpkg"
        gdf_release_points.to_file(file, layer=f"well{well_name}_release_points", driver="GPKG")

        file = base_path_external / "output" / f"{release_scenario}" / f"well{well_name}" / f"well{well_name}_release_points.csv"
        # rename index
        df_release_points.columns = [["", "[m]", "[m]", "[m a.s.l.]"], ["particleID", "x-coordinate", "y-coordinate", "elevation"]]
        # remove blank rows
        df_release_points = df_release_points.dropna(how="all")
        df_release_points.to_csv(file, index=False, sep=";")
