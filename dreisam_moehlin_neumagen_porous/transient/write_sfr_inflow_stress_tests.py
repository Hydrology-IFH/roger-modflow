from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = Path(__file__).parent

# load discharge data
import pandas as pd

file = base_path / "observations" / "discharge_dreisam_2013_2019.csv"
# Read with custom column names
df_dreisam_2013_2019_raw = pd.read_csv(file,
                 skiprows=14)
df_dreisam_2013_2019_raw['Date'] = pd.to_datetime(df_dreisam_2013_2019_raw['ISO 8601 UTC'])
df_dreisam_2013_2019_raw['Discharge'] = df_dreisam_2013_2019_raw['Value'].astype(float)

# Set date as index for time series analysis
df_dreisam_2013_2019_raw.set_index('Date', inplace=True)

# Resample to daily mean
df_dreisam_2013_2019 = df_dreisam_2013_2019_raw.loc[:, 'Discharge'].to_frame().resample('D').mean()
df_dreisam_2013_2019.index = pd.date_range(start='2013-01-01', end='2020-09-10', freq='D')

file = base_path / "observations" / "discharge_dreisam_2020_2023.csv"
# Read with custom column names
df_dreisam_2020_2023 = pd.read_csv(file,
                 skiprows=8,
                 encoding='latin-1')

# Data preprocessing
df_dreisam_2020_2023.index = pd.to_datetime(df_dreisam_2020_2023['Datum / Uhrzeit'])
df_dreisam_2020_2023['Discharge'] = df_dreisam_2020_2023['Wert'].str.replace(',', '.').astype(float)

df_dreisam_2020_2023 = df_dreisam_2020_2023.loc[:, 'Discharge'].to_frame()

_df_dreisam = pd.concat([df_dreisam_2013_2019, df_dreisam_2020_2023], axis=0, ignore_index=False, sort=False)
idx = pd.date_range(start='2013-01-01', end='2023-12-31', freq='D')
df_dreisam = pd.DataFrame(index=idx)
df_dreisam = df_dreisam.join(_df_dreisam, how='left')
# interpolate missing values
df_dreisam['Discharge'] = df_dreisam['Discharge'].interpolate(method='time')

df_dreisam.columns = ['Q']
_df_dreisam = df_dreisam.copy()
_df_dreisam.columns = [['[m3/s]'], ['Q']]
file = base_path / "input" / "2013-2023" / "discharge_dreisam.csv"
_df_dreisam.to_csv(file, index=True, sep=';')

file = base_path / "observations" / "discharge_moehlin.csv"
# Read with custom column names
df_moehlin = pd.read_csv(file,
                 skiprows=8,
                 encoding='latin-1')

# Data preprocessing
df_moehlin.index = pd.date_range(start='2013-01-01', end='2023-12-31', freq='D')
df_moehlin['Discharge'] = df_moehlin['Wert'].str.replace(',', '.').astype(float)

df_moehlin = df_moehlin.loc[:, 'Discharge'].to_frame()
df_moehlin.columns = ['Q']
_df_moehlin = df_moehlin.copy()
_df_moehlin.columns = [['[m3/s]'], ['Q']]
file = base_path / "input" / "2013-2023" / "discharge_moehlin.csv"
_df_moehlin.to_csv(file, index=True, sep=';')

file = base_path / "observations" / "discharge_neumagen.csv"
# Read with custom column names
df_neumagen = pd.read_csv(file,
                 skiprows=8,
                 encoding='latin-1')

# Data preprocessing
df_neumagen.index = pd.date_range(start='2013-01-01', end='2023-12-31', freq='D')
df_neumagen['Discharge'] = df_neumagen['Wert'].str.replace(',', '.').astype(float)

df_neumagen = df_neumagen.loc[:, 'Discharge'].to_frame()
df_neumagen.columns = ['Q']
_df_neumagen = df_neumagen.copy()
_df_neumagen.columns = [['[m3/s]'], ['Q']]
file = base_path / "input" / "2013-2023" / "discharge_neumagen.csv"
_df_neumagen.to_csv(file, index=True, sep=';')

file = base_path / "observations" / "discharge_rotbach.csv"
# Read with custom column names
df_rotbach = pd.read_csv(file,
                 skiprows=8,
                 encoding='latin-1')

# Data preprocessing
df_rotbach.index = pd.date_range(start='2013-01-01', end='2023-12-31', freq='D')
df_rotbach['Discharge'] = df_rotbach['Wert'].str.replace(',', '.').astype(float)

df_rotbach = df_rotbach.loc[:, 'Discharge'].to_frame()
df_rotbach.columns = ['Q']
_df_rotbach = df_rotbach.copy()
_df_rotbach.columns = [['[m3/s]'], ['Q']]
file = base_path / "input" / "2013-2023" / "discharge_rotbach.csv"
_df_rotbach.to_csv(file, index=True, sep=';')


file = base_path / "observations" / "discharge_brugga.csv"
# Read with custom column names
df_brugga = pd.read_csv(file,
                 skiprows=8,
                 encoding='latin-1')

# Data preprocessing
df_brugga.index = pd.date_range(start='2013-01-01', end='2023-12-31', freq='D')
df_brugga['Discharge'] = df_brugga['Wert'].str.replace(',', '.').astype(float)

df_brugga = df_brugga.loc[:, 'Discharge'].to_frame()
df_brugga.columns = ['Q']
_df_brugga = df_brugga.copy()
_df_brugga.columns = [['[m3/s]'], ['Q']]
file = base_path / "input" / "2013-2023" / "discharge_brugga.csv"
_df_brugga.to_csv(file, index=True, sep=';')


# plot discharge time series for visual inspection
fig, ax = plt.subplots(5, 1, figsize=(8, 8), sharex=True)
ax[0].plot(df_rotbach.index, df_rotbach['Q'], label='Rotbach', color='purple')
ax[0].set_ylabel("Discharge [m³/s]")
ax[0].set_xlim(df_rotbach.index.min(), df_rotbach.index.max())
ax[0].set_ylim(0, )
ax[1].plot(df_brugga.index, df_brugga['Q'], label='Brugga', color='pink')
ax[1].set_ylabel("Discharge [m³/s]")
ax[1].set_xlim(df_brugga.index.min(), df_brugga.index.max())
ax[1].set_ylim(0, )
ax[2].plot(df_dreisam.index, df_dreisam['Q'], label='Dreisam', color='blue')
ax[2].set_ylabel("Discharge [m³/s]")
ax[2].set_xlim(df_dreisam.index.min(), df_dreisam.index.max())
ax[2].set_ylim(0, )
ax[3].plot(df_moehlin.index, df_moehlin['Q'], label='Moehlin', color='orange')
ax[3].set_ylabel("Discharge [m³/s]")
ax[3].set_xlim(df_moehlin.index.min(), df_moehlin.index.max())
ax[3].set_ylim(0, )
ax[4].plot(df_neumagen.index, df_neumagen['Q'], label='Neumagen', color='red')
ax[4].set_xlabel("Date")
ax[4].set_ylabel("Discharge [m³/s]")
ax[4].set_xlim(df_neumagen.index.min(), df_neumagen.index.max())
ax[4].set_ylim(0, )
file = base_path / "figures" / "discharge.png"
fig.savefig(file, dpi=300, bbox_inches='tight')
plt.close(fig)

# calculate average of spring 2020
dreisam_average_spring_2020 = df_dreisam.loc["2020-03-01":"2020-05-31", "Q"].mean()
moehlin_average_spring_2020 = df_moehlin.loc["2020-03-01":"2020-05-31", "Q"].mean()
neumagen_average_spring_2020 = df_neumagen.loc["2020-03-01":"2020-05-31", "Q"].mean()
rotbach_average_spring_2020 = df_rotbach.loc["2020-03-01":"2020-05-31", "Q"].mean()

# calculate average of summer 2018
dreisam_average_summer_2018 = df_dreisam.loc["2018-06-01":"2018-08-31", "Q"].mean()
moehlin_average_summer_2018 = df_moehlin.loc["2018-06-01":"2018-08-31", "Q"].mean()
neumagen_average_summer_2018 = df_neumagen.loc["2018-06-01":"2018-08-31", "Q"].mean()
rotbach_average_summer_2018 = df_rotbach.loc["2018-06-01":"2018-08-31", "Q"].mean()

# q_magnitude_dreisam_spring_2020 = ((dreisam_average_spring_2020 - 5.69) / 5.69) * 100
# q_magnitude_dreisam_summer_2020 = ((dreisam_average_summer_2018 - 5.69) / 5.69) * 100

base_path = Path(__file__).parent  # current directory; change if files are elsewhere
discharge_path = base_path / "input" / "2013-2023"

discharge_stations = ["Dreisam", "Moehlin", "Neumagen", "Rotbach", "Brugga"]
durations = [0, 2, 3]
magnitudes = [0, 1, 2]

for station in discharge_stations:
    if station == "Dreisam":
        discharge = df_dreisam.copy()
    elif station == "Moehlin":
        discharge = df_moehlin.copy()
    elif station == "Neumagen":
        discharge = df_neumagen.copy()
    elif station == "Rotbach":
        discharge = df_rotbach.copy()
    elif station == "Brugga":
        discharge = df_brugga.copy()
    # select summer period of 2018
    discharge_summer_2018 = discharge.loc["2018-06-01":"2018-08-31"]
    # select summer period of 2017
    discharge_summer_2017 = discharge.loc["2017-06-01":"2017-08-31"]
    # select spring period of 2020
    discharge_spring_2020 = discharge.loc["2020-03-01":"2020-05-31"]
    # select spring period of 2015
    discharge_spring_2015 = discharge.loc["2015-03-01":"2015-05-31"]
    # select summer and spring period of 2021
    discharge_spring_summer_2021 = discharge.loc["2021-03-01":"2021-08-31"]
    for duration in durations:
        for magnitude in magnitudes:
            if magnitude == 0 and duration == 3:
                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "spring-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()

                # insert spring period of 2020 in spring period of 2019
                discharge.loc["2019-03-01":"2019-05-31"] = discharge_spring_2020.values
                # insert spring period of 2020 in spring period of 2018
                discharge.loc["2018-03-01":"2018-05-31"] = discharge_spring_2020.values

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                # insert summer period of 2018 in summer period of 2017
                discharge.loc["2017-06-01":"2017-08-31"] = discharge_summer_2018.values
                # insert summer period of 2018 in summer period of 2016
                discharge.loc["2016-06-01":"2016-08-31"] = discharge_summer_2018.values

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                # insert spring period of 2020 in spring period of 2019
                discharge.loc["2019-03-01":"2019-05-31"] = discharge_spring_2020.values
                # insert spring period of 2020 in spring period of 2018
                discharge.loc["2018-03-01":"2018-05-31"] = discharge_spring_2020.values
                # insert summer period of 2018 in summer period of 2019
                discharge.loc["2019-06-01":"2019-08-31"] = discharge_summer_2018.values
                # insert summer period of 2018 in summer period of 2020
                discharge.loc["2020-06-01":"2020-08-31"] = discharge_summer_2018.values

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

            elif magnitude == 1 and duration == 2:
                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "spring-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                if magnitude == 1:
                    q_magnitude = -11.
                elif magnitude == 2:
                    q_magnitude = -25.4

                discharge.loc["2020-03-01":"2020-05-31", "Q"] = discharge_spring_2020.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert spring period of 2020 in spring period of 2019
                discharge.loc["2019-03-01":"2019-05-31", "Q"] = discharge_spring_2020.loc[:, "Q"].values * (1 + (q_magnitude / 100))

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                if magnitude == 1:
                    q_magnitude = -11.
                elif magnitude == 2:
                    q_magnitude = -25.4
                
                discharge.loc["2018-06-01":"2018-08-31", "Q"] = discharge_summer_2018.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert summer period of 2018 in summer period of 2017
                discharge.loc["2017-06-01":"2017-08-31", "Q"] = discharge_summer_2018.loc[:, "Q"].values * (1 + (q_magnitude / 100))

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                if magnitude == 1:
                    q_magnitude = -11.
                elif magnitude == 2:
                    q_magnitude = -25.4

                discharge.loc["2020-03-01":"2020-05-31", "Q"] = discharge_spring_2020.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert spring period of 2020 in spring period of 2019
                discharge.loc["2019-03-01":"2019-05-31", "Q"] = discharge_spring_2020.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert summer period of 2018 in summer period of 2019
                discharge.loc["2019-06-01":"2019-08-31", "Q"] = discharge_summer_2018.loc[:, "Q"].values * (1 + (q_magnitude / 100))

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

            elif magnitude == 2 and duration == 3:
                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "spring-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                if magnitude == 1:
                    q_magnitude = -11.
                elif magnitude == 2:
                    q_magnitude = -25.4
                
                discharge.loc["2020-03-01":"2020-05-31", "Q"] = discharge_spring_2020.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert spring period of 2020 in spring period of 2019
                discharge.loc["2019-03-01":"2019-05-31", "Q"] = discharge_spring_2020.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert spring period of 2020 in spring period of 2018
                discharge.loc["2018-03-01":"2018-05-31", "Q"] = discharge_spring_2020.loc[:, "Q"].values * (1 + (q_magnitude / 100))

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                if magnitude == 1:
                    q_magnitude = -11.
                elif magnitude == 2:
                    q_magnitude = -25.4

                discharge.loc["2018-06-01":"2018-08-31", "Q"] = discharge_summer_2018.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert summer period of 2018 in summer period of 2017
                discharge.loc["2017-06-01":"2017-08-31", "Q"] = discharge_summer_2018.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert summer period of 2018 in summer period of 2016
                discharge.loc["2016-06-01":"2016-08-31", "Q"] = discharge_summer_2018.loc[:, "Q"].values * (1 + (q_magnitude / 100))

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                if magnitude == 1:
                    q_magnitude = -11.
                elif magnitude == 2:
                    q_magnitude = -25.4

                discharge.loc["2020-03-01":"2020-05-31", "Q"] = discharge_spring_2020.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert spring period of 2020 in spring period of 2019
                discharge.loc["2019-03-01":"2019-05-31", "Q"] = discharge_spring_2020.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert spring period of 2020 in spring period of 2018
                discharge.loc["2018-03-01":"2018-05-31", "Q"] = discharge_spring_2020.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                
                discharge.loc["2018-06-01":"2018-08-31", "Q"] = discharge_summer_2018.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert summer period of 2018 in summer period of 2019
                discharge.loc["2019-06-01":"2019-08-31", "Q"] = discharge_summer_2018.loc[:, "Q"].values * (1 + (q_magnitude / 100))
                # insert summer period of 2018 in summer period of 2020
                discharge.loc["2020-06-01":"2020-08-31", "Q"] = discharge_summer_2018.loc[:, "Q"].values * (1 + (q_magnitude / 100))

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

            elif magnitude == 2 and duration == 0:
                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "spring-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                if magnitude == 1:
                    q_magnitude = -11.
                elif magnitude == 2:
                    q_magnitude = -25.4

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                if magnitude == 1:
                    q_magnitude = -11.
                elif magnitude == 2:
                    q_magnitude = -25.4

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                if magnitude == 1:
                    q_magnitude = -11.
                elif magnitude == 2:
                    q_magnitude = -25.4

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")

            else:
                path_to_dir = base_path  / "input" / "stress_tests_discharge" / "spring-summer-wet" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                if station == "Dreisam":
                    discharge = df_dreisam.copy()
                elif station == "Moehlin":
                    discharge = df_moehlin.copy()
                elif station == "Neumagen":
                    discharge = df_neumagen.copy()
                elif station == "Rotbach":
                    discharge = df_rotbach.copy()
                # insert spring period of 2015 in spring period of 2020
                discharge.loc["2020-03-01":"2020-05-31"] = discharge_spring_2015.values
                # insert summer period of 2017 in summer period of 2018
                discharge.loc["2018-06-01":"2018-08-31"] = discharge_summer_2017.values
                # insert summer and spring period of 2021 in spring and summer period of 2020
                discharge.loc["2020-03-01":"2020-08-31"] = discharge_spring_summer_2021.values
                # insert summer and spring period of 2021 in spring and summer period of 2019
                discharge.loc["2019-03-01":"2019-08-31"] = discharge_spring_summer_2021.values

                Q_path = path_to_dir / "Q.csv"
                discharge.columns = [['[m3/s]'], ['Q']]
                discharge.to_csv(Q_path, header=True, index=True, sep=";")
