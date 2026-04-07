from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

base_path = Path(__file__).parent

# observations wells in the prous aquifer close to the fissured aquifer
observation_well_ids_porous = ['0160_070-1', '0113_071-9', '2317_071-5', '2064_120-9', '0101_120-2', '2027_120-2', '0107_119-3', '0104_071-8', '2047_120-2', '0109_119-2', 'PE 01', 'PE 39', 'PE 41']
ids_to_remove = ['0190_069-0', '0118_070-0', '0122_069-1', '0831_018-1', '0131_069-2', '2311_120-2']  # wells with inconsistent data

# Load MODFLOW parameters
path = base_path / "input" / "parameters_modflow.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")
spatial_ref = ds_params.spatial_ref
xcoords = ds_params.x.values
ycoords = ds_params.y.values
topography = ds_params['topography'].values
mask_combined = np.isfinite(topography)
mask_porous = ds_params['mask_porous_aquifer'].values == 1
mask_fissured = (ds_params['mask_black_forest'].values == 1) & np.isfinite(topography)
grid_extent = (xcoords[0], xcoords[-1], ycoords[0], ycoords[-1])

# Load observation wells
file = base_path / "observations" / "groundwater_observation_wells.gpkg"
groundwater_observation_wells = gpd.read_file(file)
groundwater_observation_wells['station_id'] = groundwater_observation_wells['station_id'].str.replace('/', '_')
# remove wells outside grid_extent
groundwater_observation_wells = groundwater_observation_wells.cx[
    grid_extent[0]:grid_extent[1], grid_extent[2]:grid_extent[3]
]
# remove wells with inconsistent data
groundwater_observation_wells = groundwater_observation_wells[
    ~groundwater_observation_wells['station_id'].isin(ids_to_remove)
]

groundwater_observation_wells = groundwater_observation_wells[
    groundwater_observation_wells['station_id'].isin(observation_well_ids_porous)
]


# Load groundwater head time series
file = base_path / "observations" / "groundwater_head_time_series_filled.csv"
_df_gw_heads = pd.read_csv(file, sep=";", index_col=0)
_df_gw_heads.index = pd.to_datetime(_df_gw_heads.index, format="%Y-%m-%d")
date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
df_gw_heads = pd.DataFrame(index=date_time)
df_gw_heads = df_gw_heads.join(_df_gw_heads, how="inner")
df_gw_heads_bc = pd.DataFrame(index=date_time, columns=groundwater_observation_wells['station_id'])
for station_id in groundwater_observation_wells['station_id']:
    if station_id in df_gw_heads.columns:
        df_gw_heads_bc.loc[:, station_id] = df_gw_heads.loc[:, station_id].values
# drop columns with all NaN values
df_gw_heads_bc = df_gw_heads_bc.dropna(axis=1, how='all')

# calculate data coverage
print(len(df_gw_heads_bc.columns), "observation wells loaded.")
data_coverage = df_gw_heads_bc.notnull().sum() / len(df_gw_heads) * 100
print("Data coverage of observation wells (%):")
for station_id, coverage in data_coverage.items():
    print(f"  {station_id}: {coverage:.2f}%")

# get IDs of observation wells with data coverage > 80
ids_high_coverage = data_coverage[data_coverage > 80].index.tolist()
df_gw_heads_bc = df_gw_heads_bc[ids_high_coverage]

groundwater_observation_wells = groundwater_observation_wells[
    groundwater_observation_wells['station_id'].isin(ids_high_coverage)
]

df_gw_heads_topo = pd.DataFrame(index=groundwater_observation_wells['station_id'], columns=['x', 'y', 'topography'])
for idx, row in groundwater_observation_wells.iterrows():
    x = row.geometry.x
    y = row.geometry.y
    col_idx = np.argmin(np.abs(xcoords - x))
    row_idx = np.argmin(np.abs(ycoords - y))
    topo = topography[row_idx, col_idx]
    df_gw_heads_topo.loc[row['station_id'], 'x'] = x
    df_gw_heads_topo.loc[row['station_id'], 'y'] = y
    df_gw_heads_topo.loc[row['station_id'], 'topography'] = topo

df_gw_depths_bc = pd.DataFrame(index=df_gw_heads_bc.index, columns=df_gw_heads_bc.columns)
# calculate groundwater depths
for well_id in df_gw_heads_bc.columns:
    if well_id in df_gw_heads_topo.index:
        topo = df_gw_heads_topo.loc[well_id, 'topography']
        df_gw_depths_bc[well_id] = topo - df_gw_heads_bc[well_id]

# plot time series of all observation wells
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
for well_id in df_gw_depths_bc.columns:
    ax.plot(df_gw_depths_bc.index, df_gw_depths_bc[well_id], label=well_id)
ax.set_xlabel("Date")
ax.set_ylabel("Groundwater Depth (m)")
ax.set_xlim([df_gw_depths_bc.index.min(), df_gw_depths_bc.index.max()])
ax.set_ylim([0, df_gw_depths_bc.max().max() + 1])
ax.invert_yaxis()
file = base_path / "figures" / "groundwater_depth_observation_wells_bc.png"
fig.savefig(file, dpi=250, bbox_inches='tight')
plt.close(fig)

# calculate anomalies
df_gw_depths_bc_anom = ((df_gw_depths_bc.copy() - df_gw_depths_bc.mean()) / df_gw_depths_bc.mean()) * (-1) * 100

# plot time series of all observation wells (anomalies)
fig, ax = plt.subplots(1, 1, figsize=(6, 3))
for well_id in df_gw_depths_bc_anom.columns:
    ax.plot(df_gw_depths_bc_anom.index, df_gw_depths_bc_anom[well_id], label=well_id, color='black', alpha=0.3)
# plot meadian
ax.plot(df_gw_depths_bc_anom.index, df_gw_depths_bc_anom.median(axis=1), color='black', linewidth=1.5, label='Median')
# plot average
ax.plot(df_gw_depths_bc_anom.index, df_gw_depths_bc_anom.mean(axis=1), color='orange', linewidth=2, label='Mean')
ax.hlines(0, df_gw_depths_bc_anom.index.min(), df_gw_depths_bc_anom.index.max(), colors='gray', linestyles='dashed')
ax.set_xlabel("Date")
ax.set_ylabel("Groundwater Depth Anomaly [%]")
ax.set_xlim([df_gw_depths_bc_anom.index.min(), df_gw_depths_bc_anom.index.max()])
ax.set_ylim(-100, 100)
file = base_path / "figures" / "groundwater_depth_anomalies_bc.png"
fig.savefig(file, dpi=250, bbox_inches='tight')
plt.close(fig)

df_lateral_recharge_anomaly = pd.DataFrame(index=df_gw_depths_bc_anom.index, columns=['anomaly'])
df_lateral_recharge_anomaly['anomaly'] = df_gw_depths_bc_anom.mean(axis=1) / 100

_df_lateral_recharge_anomaly = df_lateral_recharge_anomaly.copy()
_df_lateral_recharge_anomaly.columns = [['[-]'], ['anomaly']]
file = base_path / "input" / "2013-2023" / "lateral_recharge_anomaly.csv"
_df_lateral_recharge_anomaly.to_csv(file, index=True, sep=';')

# load stress magnitude data
file = base_path / "input" / "prec_spring_stress_magnitude.csv"
df_prec_spring_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
file = base_path / "input" / "prec_summer_stress_magnitude.csv"
df_prec_summer_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
file = base_path / "input" / "prec_autumn_stress_magnitude.csv"
df_prec_autumn_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
file = base_path / "input" / "prec_winter_stress_magnitude.csv"
df_prec_winter_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
file = base_path / "input" / "pet_spring_stress_magnitude.csv"
df_pet_spring_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
file = base_path / "input" / "pet_summer_stress_magnitude.csv"
df_pet_summer_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
file = base_path / "input" / "pet_autumn_stress_magnitude.csv"
df_pet_autumn_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1, index_col=0)
file = base_path / "input" / "pet_winter_stress_magnitude.csv"
df_pet_winter_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1, index_col=0)

durations = [0, 2, 3]
magnitudes = [0, 1, 2]

lateral_recharge_anomaly = df_lateral_recharge_anomaly.copy()
# select spring period of 2020
lateral_recharge_anomaly_spring_2020 = lateral_recharge_anomaly.loc["2020-06-01":"2020-08-31"]
lateral_recharge_anomaly_spring_2020.loc[lateral_recharge_anomaly_spring_2020['anomaly'].values > 0, 'anomaly'] = 0
# select summer period of 2018
lateral_recharge_anomaly_summer_2018 = lateral_recharge_anomaly.loc["2018-06-01":"2018-08-31"]
lateral_recharge_anomaly_summer_2018.loc[lateral_recharge_anomaly_summer_2018['anomaly'].values < 0, 'anomaly'] = 0
for duration in durations:
    for magnitude in magnitudes:
        if magnitude == 0 and duration == 3:
            path_to_dir = base_path / "input" / "stress_tests_lateral_recharge" / "spring-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            lateral_recharge_anomaly = df_lateral_recharge_anomaly.copy()

            # insert spring period of 2020 in spring period of 2019
            lateral_recharge_anomaly.loc["2018-03-01":"2018-05-31"] = lateral_recharge_anomaly_spring_2020.values
            # insert spring period of 2020 in spring period of 2018
            lateral_recharge_anomaly.loc["2017-03-01":"2017-05-31"] = lateral_recharge_anomaly_spring_2020.values

            Q_path = path_to_dir / "lateral_recharge_anomaly.csv"
            lateral_recharge_anomaly.columns = [['[-]'], ['anomaly']]
            lateral_recharge_anomaly.to_csv(Q_path, header=True, index=True, sep=";")

            path_to_dir = base_path / "input" / "stress_tests_lateral_recharge" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            lateral_recharge_anomaly = df_lateral_recharge_anomaly.copy()
            # insert summer period of 2018 in summer period of 2017
            lateral_recharge_anomaly.loc["2017-06-01":"2017-08-31"] = lateral_recharge_anomaly_summer_2018.values
            # insert summer period of 2018 in summer period of 2016
            lateral_recharge_anomaly.loc["2016-06-01":"2016-08-31"] = lateral_recharge_anomaly_summer_2018.values

            Q_path = path_to_dir / "lateral_recharge_anomaly.csv"
            lateral_recharge_anomaly.columns = [['[-]'], ['anomaly']]
            lateral_recharge_anomaly.to_csv(Q_path, header=True, index=True, sep=";")

            path_to_dir = base_path / "input" / "stress_tests_lateral_recharge" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            lateral_recharge_anomaly = df_lateral_recharge_anomaly.copy()
            # insert spring period of 2019 in spring period of 2018
            lateral_recharge_anomaly.loc["2018-03-01":"2018-05-31"] = lateral_recharge_anomaly_spring_2020.values
            # insert spring period of 2019 in spring period of 2017
            lateral_recharge_anomaly.loc["2017-03-01":"2017-05-31"] = lateral_recharge_anomaly_spring_2020.values
            # insert summer period of 2020 in summer period of 2019
            lateral_recharge_anomaly.loc["2019-06-01":"2019-08-31"] = lateral_recharge_anomaly_summer_2018.values
            # insert summer period of 2020 in summer period of 2018
            lateral_recharge_anomaly.loc["2018-06-01":"2018-08-31"] = lateral_recharge_anomaly_summer_2018.values

            Q_path = path_to_dir / "lateral_recharge_anomaly.csv"
            lateral_recharge_anomaly.columns = [['[-]'], ['anomaly']]
            lateral_recharge_anomaly.to_csv(Q_path, header=True, index=True, sep=";")

        elif magnitude == 2 and duration == 3:
            path_to_dir = base_path / "input" / "stress_tests_lateral_recharge" / "spring-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            lateral_recharge_anomaly = df_lateral_recharge_anomaly.copy()
            q_magnitude_spring = (df_prec_spring_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_spring_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_summer = (df_prec_summer_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_summer_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_autumn = (df_prec_autumn_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_autumn_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_winter = (df_prec_winter_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_winter_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
    
            # insert spring period of 2020 in spring period of 2019
            lateral_recharge_anomaly.loc["2018-03-01":"2018-05-31"] = lateral_recharge_anomaly_spring_2020.values
            # insert spring period of 2020 in spring period of 2018
            lateral_recharge_anomaly.loc["2017-03-01":"2017-05-31"] = lateral_recharge_anomaly_spring_2020.values

            # select only spring periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([3, 4, 5]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_spring / 100))
            # select only summer periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([6, 7, 8]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_summer / 100))
            # select only autumn periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([9, 10, 11]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_autumn / 100))
            # select only winter periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([12, 1, 2]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_winter / 100))

            Q_path = path_to_dir / "lateral_recharge_anomaly.csv"
            lateral_recharge_anomaly.columns = [['[-]'], ['anomaly']]
            lateral_recharge_anomaly.to_csv(Q_path, header=True, index=True, sep=";")

            path_to_dir = base_path / "input" / "stress_tests_lateral_recharge" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            lateral_recharge_anomaly = df_lateral_recharge_anomaly.copy()
            q_magnitude_spring = (df_prec_spring_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_spring_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_summer = (df_prec_summer_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_summer_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_autumn = (df_prec_autumn_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_autumn_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_winter = (df_prec_winter_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_winter_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
    
            # insert summer period of 2018 in summer period of 2017
            lateral_recharge_anomaly.loc["2017-06-01":"2017-08-31"] = lateral_recharge_anomaly_summer_2018.values
            # insert summer period of 2018 in summer period of 2016
            lateral_recharge_anomaly.loc["2016-06-01":"2016-08-31"] = lateral_recharge_anomaly_summer_2018.values

            # select only spring periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([3, 4, 5]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_spring / 100))
            # select only summer periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([6, 7, 8]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_summer / 100))
            # select only autumn periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([9, 10, 11]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_autumn / 100))
            # select only winter periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([12, 1, 2]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_winter / 100))

            Q_path = path_to_dir / "lateral_recharge_anomaly.csv"
            lateral_recharge_anomaly.columns = [['[-]'], ['anomaly']]
            lateral_recharge_anomaly.to_csv(Q_path, header=True, index=True, sep=";")

            path_to_dir = base_path / "input" / "stress_tests_lateral_recharge" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            lateral_recharge_anomaly = df_lateral_recharge_anomaly.copy()
            q_magnitude_spring = (df_prec_spring_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_spring_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_summer = (df_prec_summer_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_summer_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_autumn = (df_prec_autumn_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_autumn_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_winter = (df_prec_winter_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_winter_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
    
            lateral_recharge_anomaly.loc["2019-03-01":"2019-05-31", "anomaly"] = lateral_recharge_anomaly_spring_2020.loc[:, "anomaly"].values - q_magnitude_spring
            # insert spring period of 2019 in spring period of 2018
            lateral_recharge_anomaly.loc["2018-03-01":"2018-05-31", "anomaly"] = lateral_recharge_anomaly_spring_2020.loc[:, "anomaly"].values - q_magnitude_spring
            # insert spring period of 2019 in spring period of 2017
            lateral_recharge_anomaly.loc["2017-03-01":"2017-05-31", "anomaly"] = lateral_recharge_anomaly_spring_2020.loc[:, "anomaly"].values - q_magnitude_spring
            
            lateral_recharge_anomaly.loc["2020-06-01":"2020-08-31", "anomaly"] = lateral_recharge_anomaly_summer_2018.loc[:, "anomaly"].values - q_magnitude_summer
            # insert summer period of 2020 in summer period of 2019
            lateral_recharge_anomaly.loc["2019-06-01":"2019-08-31", "anomaly"] = lateral_recharge_anomaly_summer_2018.loc[:, "anomaly"].values - q_magnitude_summer
            # insert summer period of 2020 in summer period of 2018
            lateral_recharge_anomaly.loc["2018-06-01":"2018-08-31", "anomaly"] = lateral_recharge_anomaly_summer_2018.loc[:, "anomaly"].values - q_magnitude_summer

            Q_path = path_to_dir / "lateral_recharge_anomaly.csv"
            lateral_recharge_anomaly.columns = [['[-]'], ['anomaly']]
            lateral_recharge_anomaly.to_csv(Q_path, header=True, index=True, sep=";")

        elif magnitude == 2 and duration == 0:
            path_to_dir = base_path / "input" / "stress_tests_lateral_recharge" / "long-term" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            lateral_recharge_anomaly = df_lateral_recharge_anomaly.copy()
            q_magnitude_spring = (df_prec_spring_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_spring_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_summer = (df_prec_summer_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_summer_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_autumn = (df_prec_autumn_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_autumn_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
            q_magnitude_winter = (df_prec_winter_stress_magnitude.loc["Breitnau", "magnitude2"] - df_pet_winter_stress_magnitude.loc["Breitnau", "magnitude2"]) * 100
    
            # select only spring periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([3, 4, 5]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_spring / 100))
            # select only summer periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([6, 7, 8]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_summer / 100))
            # select only autumn periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([9, 10, 11]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_autumn / 100))
            # select only winter periods and modify them according to the stress magnitude
            cond = (lateral_recharge_anomaly.index.month.isin([12, 1, 2]))
            lateral_recharge_anomaly.loc[cond, "anomaly"] = lateral_recharge_anomaly.loc[cond, "anomaly"] * (1 + (q_magnitude_winter / 100))

            Q_path = path_to_dir / "lateral_recharge_anomaly.csv"
            lateral_recharge_anomaly.columns = [['[-]'], ['anomaly']]
            lateral_recharge_anomaly.to_csv(Q_path, header=True, index=True, sep=";")

        else:
            path_to_dir = base_path / "input" / "stress_tests_lateral_recharge" / "spring-summer-wet"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            lateral_recharge_anomaly = df_lateral_recharge_anomaly.copy()
            lateral_recharge_anomaly.loc["2019-03-01":"2019-05-31"] = lateral_recharge_anomaly_spring_2020.values
            lateral_recharge_anomaly.loc["2019-06-01":"2019-08-31"] = lateral_recharge_anomaly_summer_2018.values
            lateral_recharge_anomaly.loc["2020-03-01":"2020-05-31"] = lateral_recharge_anomaly_spring_2020.values
            lateral_recharge_anomaly.loc["2020-06-01":"2020-08-31"] = lateral_recharge_anomaly_summer_2018.values
            
            Q_path = path_to_dir / "lateral_recharge_anomaly.csv"
            lateral_recharge_anomaly.columns = [['[-]'], ['anomaly']]
            lateral_recharge_anomaly.to_csv(Q_path, header=True, index=True, sep=";")




