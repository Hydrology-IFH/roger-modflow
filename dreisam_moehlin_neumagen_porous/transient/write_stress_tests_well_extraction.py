from pathlib import Path
import pandas as pd
import os

base_path = Path(__file__).parent

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

date_time = pd.date_range(start="2013-07-01", end="2023-08-31", freq="D")
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
daily_weights_drinking_water_supply_2018 = daily_weights_drinking_water_supply.loc["2018-07-01":"2018-08-31"]

durations = [0, 3]
magnitudes = [2, 2]

for duration, magnitude in zip(durations, magnitudes):
    if magnitude == 0 and duration == 3:
        path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)  

        groundwater_extraction = df_groundwater_extraction.copy()
        daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
        # insert 2018 in 2017
        daily_weights_drinking_water_supply.loc["2017-07-01":"2017-08-31"] = daily_weights_drinking_water_supply_2018.values
        # insert 2018 in 2016
        daily_weights_drinking_water_supply.loc["2016-07-01":"2016-08-31"] = daily_weights_drinking_water_supply_2018.values

        path = path_to_dir / "daily_weights_drinking_water_supply.csv"
        daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
        daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

        path = path_to_dir / "groundwater_extraction.csv"
        groundwater_extraction.to_csv(path, header=True, index=False, sep=";")

        df_drinking_water_well_extraction_daily = pd.DataFrame(index=date_time, columns=['well_extraction'])
        for i in range(NDAYS):
            year = years[i]
            extraction_year = groundwater_extraction.loc[cond_drinking_water_supply, f"{year}"].values.sum()
            df_drinking_water_well_extraction_daily.iloc[i, 0] = extraction_year * daily_weights_drinking_water_supply.iloc[i, 0]

        # save daily drinking water well extraction to csv
        path = path_to_dir / "drinking_water_well_extraction_daily.csv"
        df_drinking_water_well_extraction_daily.to_csv(path, sep=";")
        
    elif magnitude == 2 and duration == 3:
        path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)  

        if magnitude == 1:
            q_magnitude = 0.1
        elif magnitude == 2:
            q_magnitude = 0.2

        groundwater_extraction = df_groundwater_extraction.copy()
        daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
        # insert 2018 in 2017
        daily_weights_drinking_water_supply.loc["2017-07-01":"2017-08-31"] = daily_weights_drinking_water_supply_2018.values
        # insert 2018 in 2016
        daily_weights_drinking_water_supply.loc["2016-07-01":"2016-08-31"] = daily_weights_drinking_water_supply_2018.values
        for year in range(2016, 2019):
            daily_weights_drinking_water_supply.loc[f"{year}-07-01":f"{year}-08-31"] = daily_weights_drinking_water_supply.loc[f"{year}-07-01":f"{year}-08-31"].values * (1 + q_magnitude)

        path = path_to_dir / "daily_weights_drinking_water_supply.csv"
        daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
        daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

        path = path_to_dir / "groundwater_extraction.csv"
        groundwater_extraction.to_csv(path, header=True, index=False, sep=";")

        df_drinking_water_well_extraction_daily = pd.DataFrame(index=date_time, columns=['well_extraction'])
        for i in range(NDAYS):
            year = years[i]
            extraction_year = groundwater_extraction.loc[cond_drinking_water_supply, f"{year}"].values.sum()
            df_drinking_water_well_extraction_daily.iloc[i, 0] = extraction_year * daily_weights_drinking_water_supply.iloc[i, 0]

        # save daily drinking water well extraction to csv
        df_drinking_water_well_extraction_daily.columns = [['[m3]'], ['well_extraction']]
        path = path_to_dir / "drinking_water_well_extraction_daily.csv"
        df_drinking_water_well_extraction_daily.to_csv(path, sep=";")

    elif magnitude == 2 and duration == 0:
        path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "long-term" / f"duration{duration}_magnitude{magnitude}"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)  

        if magnitude == 1:
            q_magnitude = 0.1
        elif magnitude == 2:
            q_magnitude = 0.2

        groundwater_extraction = df_groundwater_extraction.copy()
        daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
        daily_weights_drinking_water_supply.loc[:] = daily_weights_drinking_water_supply.values * (1 + q_magnitude)

        path = path_to_dir / "daily_weights_drinking_water_supply.csv"
        daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
        daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

        path = path_to_dir / "groundwater_extraction.csv"
        groundwater_extraction.to_csv(path, header=True, index=False, sep=";")

        df_drinking_water_well_extraction_daily = pd.DataFrame(index=date_time, columns=['well_extraction'])
        for i in range(NDAYS):
            year = years[i]
            extraction_year = groundwater_extraction.loc[cond_drinking_water_supply, f"{year}"].values.sum()
            df_drinking_water_well_extraction_daily.iloc[i, 0] = extraction_year * daily_weights_drinking_water_supply.iloc[i, 0]

        # save daily drinking water well extraction to csv
        df_drinking_water_well_extraction_daily.columns = [['[m3]'], ['well_extraction']]
        path = path_to_dir / "drinking_water_well_extraction_daily.csv"
        df_drinking_water_well_extraction_daily.to_csv(path, sep=";")

        path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)  

        if magnitude == 1:
            q_magnitude = 0.1
        elif magnitude == 2:
            q_magnitude = 0.2

        groundwater_extraction = df_groundwater_extraction.copy()
        daily_weights_drinking_water_supply = df_daily_weights_drinking_water_supply.copy()
        daily_weights_drinking_water_supply.loc["2018-07-01":"2018-08-31"] *= (1 + q_magnitude)

        path = path_to_dir / "daily_weights_drinking_water_supply.csv"
        daily_weights_drinking_water_supply.columns = [['[-]'], ['weights']]
        daily_weights_drinking_water_supply.to_csv(path, header=True, index=True, sep=";")

        path = path_to_dir / "groundwater_extraction.csv"
        groundwater_extraction.to_csv(path, header=True, index=False, sep=";")

        df_drinking_water_well_extraction_daily = pd.DataFrame(index=date_time, columns=['well_extraction'])
        for i in range(NDAYS):
            year = years[i]
            extraction_year = groundwater_extraction.loc[cond_drinking_water_supply, f"{year}"].values.sum()
            df_drinking_water_well_extraction_daily.iloc[i, 0] = extraction_year * daily_weights_drinking_water_supply.iloc[i, 0]

        # save daily drinking water well extraction to csv
        df_drinking_water_well_extraction_daily.columns = [['[m3]'], ['well_extraction']]
        path = path_to_dir / "drinking_water_well_extraction_daily.csv"
        df_drinking_water_well_extraction_daily.to_csv(path, sep=";")
