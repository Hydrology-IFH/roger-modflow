from pathlib import Path
import numpy as np
import pandas as pd
import os

base_path = Path(__file__).parent

def calc_pet_with_makkink(rs, ta, z, c1=0.63, c2=-0.05):
    """Calculate potential evapotranspiration according to Makkink.

    Args
    ----------
    rs : np.ndarray
        solar radiation (in MJ m-2 day-1)

    ta : np.ndarray
        air temperature (in celsius)

    z : float
        elevation above sea level (in m)

    c1 : float, optional
        Makkink coefficient (-)

    c2 : float, optional
        Makkink coefficient (-)

    Reference
    ----------
    Makkink, G. F., Testing the Penman formula by means of lysimeters,
    J. Inst. Wat. Engrs, 11, 277-288, 1957.

    Returns
    ----------
    pet : np.ndarray
        potential evapotranspiration
    """
    # slope of saturation vapour pressure curve (in kPa celsius-1)
    svpc = 4098 * (0.6108 * np.exp((17.27 * ta) / (ta + 237.3))) / (ta + 237.3) ** 2

    # atmospheric pressure (in kPa)
    p = 101.3 * ((293 - 0.0065 * z) / 293) ** 5.26

    # psychometric constant (in kPa celsius-1)
    gam = 0.665 * 1e-3 * p

    # special heat of evaporation (in MJ m-2 mm-1)
    lam = 0.0864 * (28.4 - 0.028 * ta)

    # potential evapotranspiration (in mm)
    pet = (svpc / (svpc + gam)) * ((c1 * rs / lam) + c2)

    return np.where(pet < 0, 0, pet)

climate_scenarios = ["CCCma-CanESM2_CCLM4-8-17", "MPI-M-MPI-ESM-LR_RCA4"]
periods = ["1985-2014", "2030-2059", "2070-2099"]

for climate_scenario in climate_scenarios:
    for period in periods:
        file = base_path / "input" / "meteo" / climate_scenario / period / "TA.txt"
        df_TA = pd.read_csv(file, sep="\t")

        file = base_path / "input" / "meteo" / climate_scenario / period / "RS.txt"
        df_RS = pd.read_csv(file, sep="\t")

        df_PET = pd.DataFrame(index=range(len(df_TA.index)), columns=["YYYY", "MM", "DD", "hh", "mm"])
        df_PET["YYYY"] = df_TA["YYYY"].values
        df_PET["MM"] = df_TA["MM"].values
        df_PET["DD"] = df_TA["DD"].values
        df_PET["hh"] = df_TA["hh"].values
        df_PET["mm"] = df_TA["mm"].values
        df_PET["PET"] = calc_pet_with_makkink(df_RS["RS"].values, df_TA["TA"].values, 236)
        path_txt = base_path / "input" / "meteo" / climate_scenario / period / "PET.txt"
        df_PET.to_csv(path_txt, header=True, index=False, sep="\t")