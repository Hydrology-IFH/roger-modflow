from pathlib import Path
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import datetime
import contextily as ctx
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar
import click


@click.command("main")
def main():
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

    df_drinking_water_well_extraction_daily = pd.DataFrame(index=date_time, columns=['well_extraction'])
    for i in range(NDAYS):
        year = years[i]
        extraction_year = df_groundwater_extraction.loc[cond_drinking_water_supply, f"{year}"].values.sum()
        df_drinking_water_well_extraction_daily.iloc[i, 0] = extraction_year * daily_weights_drinking_water_supply.iloc[i, 0]

    # save daily drinking water well extraction to csv
    df_drinking_water_well_extraction_daily.columns = [['[m3]'], ['well_extraction']]
    file = base_path / "input" / "drinking_water_well_extraction_daily.csv"
    df_drinking_water_well_extraction_daily.to_csv(file, sep=";")
    df_drinking_water_well_extraction_daily.columns = ['well_extraction']

    # aggreate to monthly values
    df_drinking_water_well_extraction_monthly = df_drinking_water_well_extraction_daily.resample('ME').sum()

    # save monthly drinking water well extraction to csv
    df_drinking_water_well_extraction_monthly.columns = [['[m3]'], ['well_extraction']]
    file = base_path / "input" / "drinking_water_well_extraction_monthly.csv"
    df_drinking_water_well_extraction_monthly.to_csv(file, sep=";")
    df_drinking_water_well_extraction_monthly.columns = ['well_extraction']

    # barplot monthly drinking water well extraction    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df_drinking_water_well_extraction_monthly.index, df_drinking_water_well_extraction_monthly['well_extraction']/ 1000000, color='purple', width=20)
    ax.set_xlim(df_drinking_water_well_extraction_monthly.index[0] - pd.Timedelta(days=15), df_drinking_water_well_extraction_monthly.index[-1] + pd.Timedelta(days=15))
    ax.set_xlabel('Zeit')
    ax.set_ylabel('GW-Entnahme\n[Mio. m³/Monat]')
    fig.tight_layout()
    file = base_path / "figures" / "drinking_water_well_extraction.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    # barplot monthly drinking water well extraction    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df_drinking_water_well_extraction_monthly.index, df_drinking_water_well_extraction_monthly['well_extraction']/ 1000000, color='purple', width=20, zorder=1)
    # select July and August of year 2018 and 2020 and increase values by 40% to visualize the effect of increased pumping rates during droughts
    cond = (df_drinking_water_well_extraction_monthly.index.month == 7) & (df_drinking_water_well_extraction_monthly.index.year == 2018)
    df_drinking_water_well_extraction_monthly.loc[cond, 'well_extraction'] = df_drinking_water_well_extraction_monthly.loc[cond, 'well_extraction'] * 1.4
    cond = (df_drinking_water_well_extraction_monthly.index.month == 8) & (df_drinking_water_well_extraction_monthly.index.year == 2018)
    df_drinking_water_well_extraction_monthly.loc[cond, 'well_extraction'] = df_drinking_water_well_extraction_monthly.loc[cond, 'well_extraction'] * 1.4
    cond = (df_drinking_water_well_extraction_monthly.index.month == 7) & (df_drinking_water_well_extraction_monthly.index.year == 2020)
    df_drinking_water_well_extraction_monthly.loc[cond, 'well_extraction'] = df_drinking_water_well_extraction_monthly.loc[cond, 'well_extraction'] * 1.4
    cond = (df_drinking_water_well_extraction_monthly.index.month == 8) & (df_drinking_water_well_extraction_monthly.index.year == 2020)
    df_drinking_water_well_extraction_monthly.loc[cond, 'well_extraction'] = df_drinking_water_well_extraction_monthly.loc[cond, 'well_extraction'] * 1.4
    ax.bar(df_drinking_water_well_extraction_monthly.index, df_drinking_water_well_extraction_monthly['well_extraction']/ 1000000, color='red', width=20, zorder=0)
    ax.set_xlim(df_drinking_water_well_extraction_monthly.index[0] - pd.Timedelta(days=15), df_drinking_water_well_extraction_monthly.index[-1] + pd.Timedelta(days=15))
    ax.set_xlabel('Zeit')
    ax.set_ylabel('GW-Entnahme\n[Mio. m³/Monat]')
    fig.tight_layout()
    file = base_path / "figures" / "drinking_water_well_extraction_stress.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


    return


if __name__ == "__main__":
    main()