from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import os

base_path = Path(__file__).parent

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

# load groundwater extraction data
df_groundwater_extraction = pd.read_csv(base_path / "input" / "groundwater_extraction.csv", sep=";")
df_groundwater_extraction["cell_y"] = df_groundwater_extraction["cell_y"].astype(int)
df_groundwater_extraction["cell_x"] = df_groundwater_extraction["cell_x"].astype(int)
df_groundwater_extraction["layer"] = df_groundwater_extraction["layer"].astype(int)
df_groundwater_extraction["purpose"] = df_groundwater_extraction["purpose"].astype(str)
df_groundwater_extraction["cell_y"] = df_groundwater_extraction["cell_y"].values - 1
df_groundwater_extraction["cell_x"] = df_groundwater_extraction["cell_x"].values - 1
df_groundwater_extraction["layer"] = df_groundwater_extraction["layer"].values - 1
n_wells = len(df_groundwater_extraction)
cond_drinking_water_supply = df_groundwater_extraction["purpose"].isin(['Badenova WW Ebnet', 'Badenova WW Hausen', 'Eigenwasserversorgung', 'oeffentliche Wasserversorgung']).values

# load daily weights for drinking water supply wells to scale the pumping rates of the drinking water supply wells in the well package
_daily_weights_drinking_water_supply = pd.read_csv(base_path / "input" / "daily_weights_drinking_water_supply.csv", sep=";", index_col=0)

date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
NDAYS = len(date_time)
doys = date_time.dayofyear.values
years = date_time.year.values

daily_weights_drinking_water_supply = pd.DataFrame(index=date_time, columns=['weights'])
for i in range(NDAYS):
    year = years[i]
    doy = doys[i]
    daily_weights_drinking_water_supply.iloc[i, 0] = _daily_weights_drinking_water_supply.loc[int(year), f"{int(doy)}"]


df_daily_weights_drinking_water_supply = daily_weights_drinking_water_supply.copy()
# select 2018
daily_weights_drinking_water_supply_2018 = daily_weights_drinking_water_supply.loc["2018-01-01":"2018-12-31"]

durations = [0, 2, 3]
magnitudes = [0, 1, 2]

for duration in durations:
    for magnitude in magnitudes:
        if magnitude == 0 and duration == 3:
            path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            groundwater_extraction = df_groundwater_extraction.copy()
            daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
            # insert 2018 in 2017
            daily_weights_drinking_water_supply.loc["2017-01-01":"2017-12-31"] = daily_weights_drinking_water_supply_2018.values
            groundwater_extraction.loc[:, "2017"] = groundwater_extraction.loc[:, "2018"].values
            # insert 2018 in 2016
            daily_weights_drinking_water_supply.loc["2016-01-01":"2016-12-30"] = daily_weights_drinking_water_supply_2018.values
            groundwater_extraction.loc[:, "2016"] = groundwater_extraction.loc[:, "2018"].values

            path = path_to_dir / "daily_weights_drinking_water_supply.csv"
            daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
            daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

            path = path_to_dir / "groundwater_extraction.csv"
            groundwater_extraction.to_csv(path, header=True, index=False, sep=";")

            path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            groundwater_extraction = df_groundwater_extraction.copy()
            daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
            # insert 2018 in 2017
            daily_weights_drinking_water_supply.loc["2017-01-01":"2017-12-31"] = daily_weights_drinking_water_supply_2018.values
            groundwater_extraction.loc[:, "2017"] = groundwater_extraction.loc[:, "2018"].values
            # insert 2018 in 2016
            daily_weights_drinking_water_supply.loc["2016-01-01":"2016-12-30"] = daily_weights_drinking_water_supply_2018.values
            groundwater_extraction.loc[:, "2016"] = groundwater_extraction.loc[:, "2018"].values

            path = path_to_dir / "daily_weights_drinking_water_supply.csv"
            daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
            daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

            path = path_to_dir / "groundwater_extraction.csv"
            groundwater_extraction.to_csv(path, header=True, index=False, sep=";")
            
        elif magnitude == 1 and duration == 2:
            path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            if magnitude == 1:
                q_magnitude = 0.2
            elif magnitude == 2:
                q_magnitude = 0.4
            
            groundwater_extraction = df_groundwater_extraction.copy()
            daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
            # insert 2018 in 2017
            daily_weights_drinking_water_supply.loc["2017-01-01":"2017-12-31"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2017-06-01":"2017-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2017"] = groundwater_extraction.loc[:, "2018"].values
            # insert 2018 in 2016
            daily_weights_drinking_water_supply.loc["2016-01-01":"2016-12-30"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2016-06-01":"2016-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2016"] = groundwater_extraction.loc[:, "2018"].values

            path = path_to_dir / "daily_weights_drinking_water_supply.csv"
            daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
            daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

            path = path_to_dir / "groundwater_extraction.csv"
            groundwater_extraction.to_csv(path, header=True, index=False, sep=";")

            path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            if magnitude == 1:
                q_magnitude = 0.2
            elif magnitude == 2:
                q_magnitude = 0.4

            groundwater_extraction = df_groundwater_extraction.copy()
            daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
            # insert 2018 in 2017
            daily_weights_drinking_water_supply.loc["2017-01-01":"2017-12-31"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2017-06-01":"2017-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2017"] = groundwater_extraction.loc[:, "2018"].values
            # insert 2018 in 2016
            daily_weights_drinking_water_supply.loc["2016-01-01":"2016-12-30"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2016-06-01":"2016-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2016"] = groundwater_extraction.loc[:, "2018"].values

            path = path_to_dir / "daily_weights_drinking_water_supply.csv"
            daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
            daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

            path = path_to_dir / "groundwater_extraction.csv"
            groundwater_extraction.to_csv(path, header=True, index=False, sep=";")

        elif magnitude == 2 and duration == 3:
            path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            if magnitude == 1:
                q_magnitude = 0.2
            elif magnitude == 2:
                q_magnitude = 0.4

            groundwater_extraction = df_groundwater_extraction.copy()
            daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
            # insert 2018 in 2017
            daily_weights_drinking_water_supply.loc["2017-01-01":"2017-12-31"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2017-06-01":"2017-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2017"] = groundwater_extraction.loc[:, "2018"].values
            # insert 2018 in 2016
            daily_weights_drinking_water_supply.loc["2016-01-01":"2016-12-30"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2016-06-01":"2016-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2016"] = groundwater_extraction.loc[:, "2018"].values

            path = path_to_dir / "daily_weights_drinking_water_supply.csv"
            daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
            daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

            path = path_to_dir / "groundwater_extraction.csv"
            groundwater_extraction.to_csv(path, header=True, index=False, sep=";")

            path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            if magnitude == 1:
                q_magnitude = 0.2
            elif magnitude == 2:
                q_magnitude = 0.4

            groundwater_extraction = df_groundwater_extraction.copy()
            daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
            # insert 2018 in 2017
            daily_weights_drinking_water_supply.loc["2017-01-01":"2017-12-31"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2017-06-01":"2017-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2017"] = groundwater_extraction.loc[:, "2018"].values
            # insert 2018 in 2016
            daily_weights_drinking_water_supply.loc["2016-01-01":"2016-12-30"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2016-06-01":"2016-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2016"] = groundwater_extraction.loc[:, "2018"].values

            path = path_to_dir / "daily_weights_drinking_water_supply.csv"
            daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
            daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

            path = path_to_dir / "groundwater_extraction.csv"
            groundwater_extraction.to_csv(path, header=True, index=False, sep=";")

        elif magnitude == 2 and duration == 0:
            path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            if magnitude == 1:
                q_magnitude = 0.2
            elif magnitude == 2:
                q_magnitude = 0.4

            groundwater_extraction = df_groundwater_extraction.copy()
            daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
            # insert 2018 in 2017
            daily_weights_drinking_water_supply.loc["2017-01-01":"2017-12-31"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2017-06-01":"2017-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2017"] = groundwater_extraction.loc[:, "2018"].values
            # insert 2018 in 2016
            daily_weights_drinking_water_supply.loc["2016-01-01":"2016-12-30"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2016-06-01":"2016-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2016"] = groundwater_extraction.loc[:, "2018"].values

            path = path_to_dir / "daily_weights_drinking_water_supply.csv"
            daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
            daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

            path = path_to_dir / "groundwater_extraction.csv"
            groundwater_extraction.to_csv(path, header=True, index=False, sep=";")

            path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}"
            if not os.path.exists(path_to_dir):
                os.makedirs(path_to_dir)  

            if magnitude == 1:
                q_magnitude = 0.2
            elif magnitude == 2:
                q_magnitude = 0.4

            groundwater_extraction = df_groundwater_extraction.copy()
            daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
            # insert 2018 in 2017
            daily_weights_drinking_water_supply.loc["2017-01-01":"2017-12-31"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2017-06-01":"2017-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2017"] = groundwater_extraction.loc[:, "2018"].values
            # insert 2018 in 2016
            daily_weights_drinking_water_supply.loc["2016-01-01":"2016-12-30"] = daily_weights_drinking_water_supply_2018.values
            daily_weights_drinking_water_supply.loc["2016-06-01":"2016-08-31"] *= (1 + q_magnitude)
            groundwater_extraction.loc[:, "2016"] = groundwater_extraction.loc[:, "2018"].values

            path = path_to_dir / "daily_weights_drinking_water_supply.csv"
            daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
            daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

            path = path_to_dir / "groundwater_extraction.csv"
            groundwater_extraction.to_csv(path, header=True, index=False, sep=";")



