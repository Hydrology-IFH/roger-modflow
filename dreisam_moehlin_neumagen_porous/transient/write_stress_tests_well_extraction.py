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

# load daily well extraction data
df_well_extraction_daily = pd.read_csv(base_path / "input" / "well_extraction_daily.csv", sep=";", index_col=0)
df_well_extraction_daily.index = pd.to_datetime(df_well_extraction_daily.index, format="%Y-%m-%d")

wells_wsg_hausen = ["A2", "A3", "A4", "B1", "B4", "C1"]
wells_wsg_zartener_becken = ["HU1", "HU2", "HU3", "K2", "K5", "S2"]
wells_badenova = wells_wsg_hausen + wells_wsg_zartener_becken
cond = df_groundwater_extraction["purpose"].isin(['Eigenwasserversorgung', 'oeffentliche Wasserversorgung'])
wells_local_water_supply = df_groundwater_extraction.loc[cond, "ID"].values.tolist()
wells_water_supply = wells_badenova + wells_local_water_supply


date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
NDAYS = len(date_time)
doys = date_time.dayofyear.values
years = date_time.year.values

durations = [0, 3, 3]
magnitudes = [2, 2, 0]

for duration, magnitude in zip(durations, magnitudes):
    if magnitude == 0 and duration == 3:
        path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)  

        df_well_extraction_daily_stress_test = df_well_extraction_daily.copy()
        # insert 2018 in 2017
        df_well_extraction_daily_stress_test.loc["2017-07-01":"2017-08-31"] = df_well_extraction_daily.loc["2018-07-01":"2018-08-31"].values
        # insert 2018 in 2016
        df_well_extraction_daily_stress_test.loc["2016-07-01":"2016-08-31"] = df_well_extraction_daily.loc["2018-07-01":"2018-08-31"].values

        df_extraction_shares = pd.DataFrame(index=df_well_extraction_daily_stress_test.index, columns=['wsg_hausen_share', 'wsg_zartener_becken_share'])
        df_extraction_shares.loc[:, 'wsg_hausen_share'] = df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values / (df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values + df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values)
        df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] = df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values / (df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values + df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values)

        # increase water supply wells extraction by q_magnitude
        df_well_extraction_daily_stress_test.loc[:, wells_water_supply] = df_well_extraction_daily_stress_test.loc[:, wells_water_supply] * (1 + q_magnitude)

        # if extraction exceeds 1000 m3/hour, redistribute extraction to Hausen wells
        redistribution_to_hausen = df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values - (1000 * 24)
        cond = (redistribution_to_hausen < 0)
        redistribution_to_hausen[cond] = 0
        cond = (df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] > 0.4) & (df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] < 0.2)
        redistribution_to_hausen[cond] = 0

        cond = (redistribution_to_hausen > 0)
        redistribution_factor_hausen = 1 + (redistribution_to_hausen[cond] / df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen].sum(axis=1))
        redistribution_factor_zartener_becken = 1 - (redistribution_to_hausen[cond] / df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken].sum(axis=1))

        df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen] = df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen].multiply(redistribution_factor_hausen, axis=0)
        df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken] = df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken].multiply(redistribution_factor_zartener_becken, axis=0)

        # save daily drinking water well extraction to csv
        path = path_to_dir / "well_extraction_daily.csv"
        df_well_extraction_daily_stress_test.to_csv(path, sep=";")

        # save redistribution factors to csv
        df_redistribution_factors = pd.DataFrame(index=df_well_extraction_daily_stress_test.index, columns=['redistribution_factor_hausen', 'redistribution_factor_zartener_becken'])
        df_redistribution_factors.loc[cond, 'redistribution_factor_hausen'] = redistribution_factor_hausen
        df_redistribution_factors.loc[cond, 'redistribution_factor_zartener_becken'] = redistribution_factor_zartener_becken
        df_redistribution_factors.to_csv(path_to_dir / "redistribution_factors.csv", sep=";")
        
    elif magnitude == 2 and duration == 3:
        path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)  

        if magnitude == 1:
            q_magnitude = 0.15
        elif magnitude == 2:
            q_magnitude = 0.3

        df_well_extraction_daily_stress_test = df_well_extraction_daily.copy()
        # insert 2018 in 2017
        df_well_extraction_daily_stress_test.loc["2017-07-01":"2017-08-31"] = df_well_extraction_daily.loc["2018-07-01":"2018-08-31"].values
        # insert 2018 in 2016
        df_well_extraction_daily_stress_test.loc["2016-07-01":"2016-08-31"] = df_well_extraction_daily.loc["2018-07-01":"2018-08-31"].values

        df_extraction_shares = pd.DataFrame(index=df_well_extraction_daily_stress_test.index, columns=['wsg_hausen_share', 'wsg_zartener_becken_share'])
        df_extraction_shares.loc[:, 'wsg_hausen_share'] = df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values / (df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values + df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values)
        df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] = df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values / (df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values + df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values)

        # increase water supply wells extraction by q_magnitude
        df_well_extraction_daily_stress_test.loc[:, wells_water_supply] = df_well_extraction_daily_stress_test.loc[:, wells_water_supply] * (1 + q_magnitude)

        # if extraction exceeds 1000 m3/hour, redistribute extraction to Hausen wells
        redistribution_to_hausen = df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values - (1000 * 24)
        cond = (redistribution_to_hausen < 0)
        redistribution_to_hausen[cond] = 0
        cond = (df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] > 0.4) & (df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] < 0.2)
        redistribution_to_hausen[cond] = 0

        cond = (redistribution_to_hausen > 0)
        redistribution_factor_hausen = 1 + (redistribution_to_hausen[cond] / df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen].sum(axis=1))
        redistribution_factor_zartener_becken = 1 - (redistribution_to_hausen[cond] / df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken].sum(axis=1))

        df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen] = df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen].multiply(redistribution_factor_hausen, axis=0)
        df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken] = df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken].multiply(redistribution_factor_zartener_becken, axis=0)

        # save daily drinking water well extraction to csv
        path = path_to_dir / "well_extraction_daily.csv"
        df_well_extraction_daily_stress_test.to_csv(path, sep=";")

        # save redistribution factors to csv
        df_redistribution_factors = pd.DataFrame(index=df_well_extraction_daily_stress_test.index, columns=['redistribution_factor_hausen', 'redistribution_factor_zartener_becken'])
        df_redistribution_factors.loc[cond, 'redistribution_factor_hausen'] = redistribution_factor_hausen
        df_redistribution_factors.loc[cond, 'redistribution_factor_zartener_becken'] = redistribution_factor_zartener_becken
        df_redistribution_factors.to_csv(path_to_dir / "redistribution_factors.csv", sep=";")

    elif magnitude == 2 and duration == 0:
        path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "long-term" / f"duration{duration}_magnitude{magnitude}"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)  

        if magnitude == 1:
            q_magnitude = 0.15
        elif magnitude == 2:
            q_magnitude = 0.3

        df_well_extraction_daily_stress_test = df_well_extraction_daily.copy()

        df_extraction_shares = pd.DataFrame(index=df_well_extraction_daily_stress_test.index, columns=['wsg_hausen_share', 'wsg_zartener_becken_share'])
        df_extraction_shares.loc[:, 'wsg_hausen_share'] = df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values / (df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values + df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values)
        df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] = df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values / (df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values + df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values)

        # increase water supply wells extraction by q_magnitude
        df_well_extraction_daily_stress_test.loc[:, wells_water_supply] = df_well_extraction_daily_stress_test.loc[:, wells_water_supply] * (1 + q_magnitude)

        # if extraction exceeds 1000 m3/hour, redistribute extraction to Hausen wells
        redistribution_to_hausen = df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values - (1000 * 24)
        cond = (redistribution_to_hausen < 0)
        redistribution_to_hausen[cond] = 0
        cond = (df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] > 0.4) & (df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] < 0.2)
        redistribution_to_hausen[cond] = 0

        cond = (redistribution_to_hausen > 0)
        redistribution_factor_hausen = 1 + (redistribution_to_hausen[cond] / df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen].sum(axis=1))
        redistribution_factor_zartener_becken = 1 - (redistribution_to_hausen[cond] / df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken].sum(axis=1))

        df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen] = df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen].multiply(redistribution_factor_hausen, axis=0)
        df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken] = df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken].multiply(redistribution_factor_zartener_becken, axis=0)

        # save daily drinking water well extraction to csv
        path = path_to_dir / "well_extraction_daily.csv"
        df_well_extraction_daily_stress_test.to_csv(path, sep=";")

        # save redistribution factors to csv
        df_redistribution_factors = pd.DataFrame(index=df_well_extraction_daily_stress_test.index, columns=['redistribution_factor_hausen', 'redistribution_factor_zartener_becken'])
        df_redistribution_factors.loc[cond, 'redistribution_factor_hausen'] = redistribution_factor_hausen
        df_redistribution_factors.loc[cond, 'redistribution_factor_zartener_becken'] = redistribution_factor_zartener_becken
        df_redistribution_factors.to_csv(path_to_dir / "redistribution_factors.csv", sep=";")

        path_to_dir = base_path / "input" / "stress_tests_well_extraction" / "summer-drought" / f"duration{duration}_magnitude{magnitude}"
        if not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)  

        if magnitude == 1:
            q_magnitude = 0.15
        elif magnitude == 2:
            q_magnitude = 0.3

        df_well_extraction_daily_stress_test = df_well_extraction_daily.copy()

        df_extraction_shares = pd.DataFrame(index=df_well_extraction_daily_stress_test.index, columns=['wsg_hausen_share', 'wsg_zartener_becken_share'])
        df_extraction_shares.loc[:, 'wsg_hausen_share'] = df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values / (df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values + df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values)
        df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] = df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values / (df_well_extraction_daily_stress_test.loc[:, wells_wsg_hausen].sum(axis=1).values + df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values)

        # increase water supply wells extraction by q_magnitude
        df_well_extraction_daily_stress_test.loc[:, wells_water_supply] = df_well_extraction_daily_stress_test.loc[:, wells_water_supply] * (1 + q_magnitude)

        # if extraction exceeds 1000 m3/hour, redistribute extraction to Hausen wells
        redistribution_to_hausen = df_well_extraction_daily_stress_test.loc[:, wells_wsg_zartener_becken].sum(axis=1).values - (1000 * 24)
        cond = (redistribution_to_hausen < 0)
        redistribution_to_hausen[cond] = 0
        cond = (df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] > 0.4) & (df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] < 0.2)
        redistribution_to_hausen[cond] = 0

        cond = (redistribution_to_hausen > 0)
        redistribution_factor_hausen = 1 + (redistribution_to_hausen[cond] / df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen].sum(axis=1))
        redistribution_factor_zartener_becken = 1 - (redistribution_to_hausen[cond] / df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken].sum(axis=1))

        df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen] = df_well_extraction_daily_stress_test.loc[cond, wells_wsg_hausen].multiply(redistribution_factor_hausen, axis=0)
        df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken] = df_well_extraction_daily_stress_test.loc[cond, wells_wsg_zartener_becken].multiply(redistribution_factor_zartener_becken, axis=0)

        # save daily drinking water well extraction to csv
        path = path_to_dir / "well_extraction_daily.csv"
        df_well_extraction_daily_stress_test.to_csv(path, sep=";")

        # save redistribution factors to csv
        df_redistribution_factors = pd.DataFrame(index=df_well_extraction_daily_stress_test.index, columns=['redistribution_factor_hausen', 'redistribution_factor_zartener_becken'])
        df_redistribution_factors.loc[cond, 'redistribution_factor_hausen'] = redistribution_factor_hausen
        df_redistribution_factors.loc[cond, 'redistribution_factor_zartener_becken'] = redistribution_factor_zartener_becken
        df_redistribution_factors.to_csv(path_to_dir / "redistribution_factors.csv", sep=";")
