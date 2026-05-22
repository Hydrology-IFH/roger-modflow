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
    wells_badenova = wells_wsg_hausen + wells_wsg_zartener_becken

    df_well_extraction_wsg_hausen = pd.read_csv(base_path / "input" / "drinking_water_well_extraction_data" / "well_extraction_wsg_hausen.csv", sep=";", index_col=0, skiprows=0)
    df_well_extraction_wsg_hausen.index = pd.to_datetime(df_well_extraction_wsg_hausen.index)
    df_well_extraction_wsg_zartener_becken = pd.read_csv(base_path / "input" / "drinking_water_well_extraction_data" / "well_extraction_wsg_zartener_becken.csv", sep=";", index_col=0, skiprows=0)
    df_well_extraction_wsg_zartener_becken.index = pd.to_datetime(df_well_extraction_wsg_zartener_becken.index)
    df_well_extraction_wsg_hausen = df_well_extraction_wsg_hausen.loc["2013-01-01":"2023-12-31", :]    
    df_well_extraction_wsg_zartener_becken = df_well_extraction_wsg_zartener_becken.loc["2013-01-01":"2023-12-31", :]   


    df_groundwater_extraction = pd.read_csv(base_path / "input" / "groundwater_extraction.csv", sep=";")
    df_groundwater_extraction.index = df_groundwater_extraction["ID"].values
    df_daily_weights_drinking_water_supply = pd.read_csv(base_path / "input" / "drinking_water_well_extraction_data" / "daily_weights_drinking_water_supply.csv", sep=";", index_col=0, skiprows=1)

    cond = df_groundwater_extraction["purpose"].isin(['Eigenwasserversorgung', 'oeffentliche Wasserversorgung'])
    wells_local_water_supply = df_groundwater_extraction.loc[cond, "ID"].values.tolist()

    df_well_extraction_daily = pd.DataFrame(index=df_well_extraction_wsg_hausen.index, columns=df_groundwater_extraction.index)
    for well in wells_wsg_zartener_becken:
        df_well_extraction_daily[well] = df_well_extraction_wsg_zartener_becken[well]
    for well in wells_wsg_hausen:
        df_well_extraction_daily[well] = df_well_extraction_wsg_hausen[well]
    years = df_well_extraction_daily.index.year.unique()
    for well in wells_local_water_supply:
        for year in years:
            value_year = df_groundwater_extraction.loc[well, f"{year}"]
            daily_weights_year = df_daily_weights_drinking_water_supply.loc[f"{year}-01-01":f"{year}-12-31", "weights"]
            df_well_extraction_daily.loc[f"{year}-01-01":f"{year}-12-31", well] = value_year * daily_weights_year.values

    wells_water_supply = wells_badenova + wells_local_water_supply
    for well in df_groundwater_extraction.index:
        if well not in wells_water_supply:
            for year in years:
                value_year = df_groundwater_extraction.loc[well, f"{year}"]
                df_well_extraction_daily.loc[f"{year}-01-01":f"{year}-12-31", well] = value_year/365.25

    #  fill NaN values with 0
    df_well_extraction_daily = df_well_extraction_daily.fillna(0)

    # save well extraction data for all wells
    path = base_path / "input" / "well_extraction_daily.csv"
    df_well_extraction_daily.to_csv(path, header=True, index=True, sep=";")
        
    return


if __name__ == "__main__":
    main()