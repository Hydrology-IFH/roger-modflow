from pathlib import Path
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent

    wells_wsg_hausen = ["A2", "A3", "A4", "B1", "B4", "C1"]
    wells_wsg_zartener_becken = ["HU1", "HU2", "HU3", "K2", "K5", "S2"]

    df_well_extraction_wsg_hausen = pd.DataFrame(index=pd.date_range(start="2008-01-01", end="2024-12-31", freq="D"))
    df_well_extraction_wsg_zartener_becken_ = pd.DataFrame(index=pd.date_range(start="2014-01-01", end="2024-12-31", freq="D"))

    for well in wells_wsg_hausen:
        file = base_path / "input" / "drinking_water_well_extraction_data" / f"PumpingVolume{well}.csv"
        df = pd.read_csv(file, sep=",", index_col=1, skiprows=14)
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
        df = df.loc[:, ["Value"]]
        # values greater than 1200 m³/day are set to 1200 m³/day
        cond = (df["Value"].values > 1200 * 24)
        df.loc[cond, "Value"] = df.loc[cond, "Value"].values / 10
        df.columns = [f"{well}"]
        df_well_extraction_wsg_hausen = df_well_extraction_wsg_hausen.join(df, how="inner")

    # sum up the extraction of all wells for each day
    df_well_extraction_wsg_hausen["total_extraction"] = df_well_extraction_wsg_hausen.sum(axis=1)

    # write to csv file
    df_well_extraction_wsg_hausen.to_csv(base_path / "input" / "drinking_water_well_extraction_data" / "well_extraction_wsg_hausen.csv", sep=";")

    for well in wells_wsg_zartener_becken:
        file = base_path / "input" / "drinking_water_well_extraction_data" / f"PumpingVolume{well}.csv"
        df = pd.read_csv(file, sep=",", index_col=1, skiprows=14)
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
        df = df.loc[:, ["Value"]]
        # aggregate to daily values by summing up the values of each day
        df = df.resample("D").sum()
        df.columns = [f"{well}"]
        df_well_extraction_wsg_zartener_becken_ = df_well_extraction_wsg_zartener_becken_.join(df, how="inner")

    # sum up the extraction of all wells for each day
    df_well_extraction_wsg_zartener_becken_["total_extraction"] = df_well_extraction_wsg_zartener_becken_.sum(axis=1)

    # write to csv file
    df_well_extraction_wsg_zartener_becken_.to_csv(base_path / "input" / "drinking_water_well_extraction_data" / "well_extraction_wsg_zartener_becken_.csv", sep=";")

    # aggregate to annual values by summing up the values of each year
    df_well_extraction_wsg_hausen_annual = df_well_extraction_wsg_hausen.resample("YE").sum()
    df_well_extraction_wsg_zartener_becken_annual = df_well_extraction_wsg_zartener_becken_.resample("YE").sum()

    # write to csv file
    df_well_extraction_wsg_hausen_annual.to_csv(base_path / "input" / "drinking_water_well_extraction_data" / "well_extraction_wsg_hausen_annual.csv", sep=";")
    df_well_extraction_wsg_zartener_becken_annual.to_csv(base_path / "input" / "drinking_water_well_extraction_data" / "well_extraction_wsg_zartener_becken_annual.csv", sep=";")

    # generate daily weights for the drinking water supply wells by dividing the daily extraction of the drinking water supply wells by the total daily extraction of all wells for each year
    daily_weights_wsg_hausen = pd.DataFrame(index=df_well_extraction_wsg_hausen.index, columns=['weights'])
    for year in range(2008, 2025):
        total_extraction_year = df_well_extraction_wsg_hausen_annual.loc[f"{year}", "total_extraction"].values[0]
        extraction_year = df_well_extraction_wsg_hausen.loc[df_well_extraction_wsg_hausen.index.year == year, "total_extraction"].values
        daily_weights_wsg_hausen.loc[daily_weights_wsg_hausen.index.year == year, "weights"] = extraction_year / total_extraction_year

    weights_2013 = daily_weights_wsg_hausen.loc[daily_weights_wsg_hausen.index.year == year, "weights"].values
    df_well_extraction_wsg_zartener_becken_2013 = pd.DataFrame(index=pd.date_range(start="2013-01-01", end="2013-12-31", freq="D"))
    for well in wells_wsg_zartener_becken:
        weight = df_well_extraction_wsg_hausen_annual.loc["2013", "total_extraction"].values[0] / df_well_extraction_wsg_hausen_annual.loc["2014", "total_extraction"].values[0]
        annual_value_2013 = df_well_extraction_wsg_zartener_becken_annual.loc["2014", f"{well}"].values[0] * weight
        daily_weights_hausen_2013 = daily_weights_wsg_hausen.loc[daily_weights_wsg_hausen.index.year == 2013, "weights"].values
        df_well_extraction_wsg_zartener_becken_2013[f"{well}"] = daily_weights_hausen_2013 * annual_value_2013

    df_well_extraction_wsg_zartener_becken = pd.concat([df_well_extraction_wsg_zartener_becken_2013, df_well_extraction_wsg_zartener_becken_], axis=0)
    df_well_extraction_wsg_zartener_becken["total_extraction"] = df_well_extraction_wsg_zartener_becken.loc[:, :"S4"].sum(axis=1)
    # write to csv file
    df_well_extraction_wsg_zartener_becken.to_csv(base_path / "input" / "drinking_water_well_extraction_data" / "well_extraction_wsg_zartener_becken.csv", sep=";")

    # write to csv file
    df_well_extraction_wsg_zartener_becken_annual = df_well_extraction_wsg_zartener_becken.resample("YE").sum()
    df_well_extraction_wsg_zartener_becken_annual.to_csv(base_path / "input" / "drinking_water_well_extraction_data" / "well_extraction_wsg_zartener_becken_annual.csv", sep=";")

    # generate daily weights for the drinking water supply wells by dividing the daily extraction of the drinking water supply wells by the total daily extraction of all wells for each year
    daily_weights_wsg_zartener_becken = pd.DataFrame(index=df_well_extraction_wsg_zartener_becken.index, columns=['weights'])
    for year in range(2013, 2025):
        total_extraction_year = df_well_extraction_wsg_zartener_becken_annual.loc[f"{year}", "total_extraction"].values[0]
        extraction_year = df_well_extraction_wsg_zartener_becken.loc[df_well_extraction_wsg_zartener_becken.index.year == year, "total_extraction"].values
        daily_weights_wsg_zartener_becken.loc[daily_weights_wsg_zartener_becken.index.year == year, "weights"] = extraction_year / total_extraction_year

    df_daily_weights = pd.DataFrame(index=pd.date_range(start="2013-01-01", end="2023-12-31", freq="D"), columns=['weights'])
    for year in range(2013, 2024):
        total_extraction_year_1 = df_well_extraction_wsg_zartener_becken_annual.loc[f"{year}", "total_extraction"].values[0]
        total_extraction_year_2 = df_well_extraction_wsg_hausen_annual.loc[f"{year}", "total_extraction"].values[0]
        total_extraction_year = total_extraction_year_1 + total_extraction_year_2
        extraction_year_1 = df_well_extraction_wsg_zartener_becken.loc[df_well_extraction_wsg_zartener_becken.index.year == year, "total_extraction"].values
        extraction_year_2 = df_well_extraction_wsg_hausen.loc[df_well_extraction_wsg_hausen.index.year == year, "total_extraction"].values
        extraction_year = extraction_year_1 + extraction_year_2
        df_daily_weights.loc[df_daily_weights.index.year == year, "weights"] = extraction_year / total_extraction_year

    df_daily_weights_wsg_hausen = daily_weights_wsg_hausen.loc["2008-01-01":"2023-12-31", :]
    df_daily_weights_wsg_zartener_becken = daily_weights_wsg_zartener_becken.loc["2013-01-01":"2023-12-31", :]
    df_daily_weights_wsg_hausen.columns = [['[-]'], ['weights']]
    df_daily_weights_wsg_zartener_becken.columns = [['[-]'], ['weights']]
    df_daily_weights.columns = [['[-]'], ['weights']]
    df_daily_weights_wsg_hausen.to_csv(base_path / "input" / "drinking_water_well_extraction_data" / "daily_weights_wsg_hausen.csv", header=True, index=True, sep=";")
    df_daily_weights_wsg_zartener_becken.to_csv(base_path / "input" / "drinking_water_well_extraction_data" / "daily_weights_wsg_zartener_becken.csv", header=True, index=True, sep=";")
    df_daily_weights.to_csv(base_path / "input" / "drinking_water_well_extraction_data" / "daily_weights_drinking_water_supply.csv", header=True, index=True, sep=";")

    return


if __name__ == "__main__":
    main()