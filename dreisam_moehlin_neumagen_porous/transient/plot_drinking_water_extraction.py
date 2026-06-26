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
    df_well_extraction_wsg_hausen = pd.read_csv(base_path / "input" / "drinking_water_well_extraction_data" / "well_extraction_wsg_hausen.csv", sep=";", index_col=0, skiprows=0)
    # convert index to datetime
    df_well_extraction_wsg_hausen.index = pd.to_datetime(df_well_extraction_wsg_hausen.index)
    df_well_extraction_wsg_zartener_becken = pd.read_csv(base_path / "input" / "drinking_water_well_extraction_data" / "well_extraction_wsg_zartener_becken.csv", sep=";", index_col=0, skiprows=0)
    # convert index to datetime
    df_well_extraction_wsg_zartener_becken.index = pd.to_datetime(df_well_extraction_wsg_zartener_becken.index)
    
    df_well_extraction_wsg_hausen = df_well_extraction_wsg_hausen.loc["2013-01-01":"2023-12-31", :]    
    df_well_extraction_wsg_zartener_becken = df_well_extraction_wsg_zartener_becken.loc["2013-01-01":"2023-12-31", :]    

    df_extraction_shares = pd.DataFrame(index=df_well_extraction_wsg_hausen.index, columns=['wsg_hausen_share', 'wsg_zartener_becken_share'])
    df_extraction_shares.loc[:, 'wsg_hausen_share'] = df_well_extraction_wsg_hausen.loc[:, 'total_extraction'].values / (df_well_extraction_wsg_hausen.loc[:, 'total_extraction'].values + df_well_extraction_wsg_zartener_becken.loc[:, 'total_extraction'].values)
    df_extraction_shares.loc[:, 'wsg_zartener_becken_share'] = df_well_extraction_wsg_zartener_becken.loc[:, 'total_extraction'].values / (df_well_extraction_wsg_hausen.loc[:, 'total_extraction'].values + df_well_extraction_wsg_zartener_becken.loc[:, 'total_extraction'].values)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df_well_extraction_wsg_hausen.index, df_well_extraction_wsg_hausen['total_extraction'], color='orange', label='WSG Hausen')
    ax.plot(df_well_extraction_wsg_zartener_becken.index, df_well_extraction_wsg_zartener_becken['total_extraction'], color='blue', label='WSG Zartener Becken')
    ax.set_xlabel('Zeit')
    ax.set_ylabel('GW-Entnahme [m³/Tag]')
    ax.set_xlim(df_well_extraction_wsg_hausen.index[0] - pd.Timedelta(days=15), df_well_extraction_wsg_hausen.index[-1] + pd.Timedelta(days=15))
    ax.set_ylim(0, )
    ax.legend(loc='upper left', frameon=False, ncol=2)
    fig.tight_layout()
    file = base_path / "figures" / "badenova_water_well_extraction_time_series.png"
    fig.savefig(file, dpi=300)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df_extraction_shares.index, df_extraction_shares['wsg_hausen_share'], color='orange', label='WSG Hausen')
    ax.plot(df_extraction_shares.index, df_extraction_shares['wsg_zartener_becken_share'], color='blue', label='WSG Zartener Becken')
    ax.set_xlabel('Zeit')
    ax.set_ylabel('Entnahmefaktor [-]')
    ax.set_xlim(df_well_extraction_wsg_hausen.index[0] - pd.Timedelta(days=15), df_well_extraction_wsg_hausen.index[-1] + pd.Timedelta(days=15))
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', frameon=False, ncol=2)
    fig.tight_layout()
    file = base_path / "figures" / "badenova_water_well_extraction_shares_time_series.png"
    fig.savefig(file, dpi=300)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df_well_extraction_wsg_zartener_becken.index, df_well_extraction_wsg_zartener_becken['total_extraction'], color='blue', label='WSG Zartener Becken')
    ax.set_xlabel('Zeit')
    ax.set_ylabel('GW-Entnahme [m³/Tag]')
    ax.set_xlim(df_well_extraction_wsg_hausen.index[0] - pd.Timedelta(days=15), df_well_extraction_wsg_hausen.index[-1] + pd.Timedelta(days=15))
    ax.set_ylim(0, )
    fig.tight_layout()
    file = base_path / "figures" / "wsg_hausen_water_well_extraction_time_series.png"
    fig.savefig(file, dpi=300)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df_well_extraction_wsg_zartener_becken.index, df_well_extraction_wsg_zartener_becken['total_extraction'], color='blue', label='WSG Zartener Becken')
    ax.axhline(y=900*24, color='black', linestyle='-.', linewidth=0.5)
    ax.axhline(y=1200*24, color='black', linestyle='--', linewidth=0.65)
    ax.axhline(y=1500*24, color='black', linestyle='-', linewidth=0.9)
    ax.set_xlabel('Zeit')
    ax.set_ylabel('GW-Entnahme [m³/Tag]')
    ax.set_xlim(df_well_extraction_wsg_hausen.index[0] - pd.Timedelta(days=15), df_well_extraction_wsg_hausen.index[-1] + pd.Timedelta(days=15))
    ax.set_ylim(0, )
    fig.tight_layout()
    file = base_path / "figures" / "wsg_zartener_becken_water_well_extraction_time_series.png"
    fig.savefig(file, dpi=300)

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


    date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")

    _df_well_extraction_daily = pd.read_csv(base_path / "input" / "well_extraction_daily.csv", sep=";", index_col=0)
    _df_well_extraction_daily.index = pd.to_datetime(_df_well_extraction_daily.index)
    df_well_extraction_daily = _df_well_extraction_daily.sum(axis=1).to_frame(name='well_extraction')

    # aggreate to monthly values
    df_well_extraction_monthly = df_well_extraction_daily.resample("ME").sum()

    # barplot monthly drinking water well extraction    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df_well_extraction_monthly.index, df_well_extraction_monthly['well_extraction']/ 1000000, color='purple', width=20)
    ax.set_xlim(df_well_extraction_monthly.index[0] - pd.Timedelta(days=15), df_well_extraction_monthly.index[-1] + pd.Timedelta(days=15))
    ax.set_xlabel('Zeit')
    ax.set_ylabel('GW-Entnahme\n[Mio. m³/Monat]')
    fig.tight_layout()
    file = base_path / "figures" / "well_extraction_monthly.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    # barplot monthly drinking water well extraction    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(df_well_extraction_monthly.index, df_well_extraction_monthly['well_extraction']/ 1000000, color='purple', width=20, zorder=1)
    df_well_extraction_monthly.loc[:, 'well_extraction'] = df_well_extraction_monthly.loc[:, 'well_extraction'] * 1.3
    ax.bar(df_well_extraction_monthly.index, df_well_extraction_monthly['well_extraction']/ 1000000, color='red', width=20, zorder=0)
    ax.set_xlim(df_well_extraction_monthly.index[0] - pd.Timedelta(days=15), df_well_extraction_monthly.index[-1] + pd.Timedelta(days=15))
    ax.set_xlabel('Zeit')
    ax.set_ylabel('GW-Entnahme\n[Mio. m³/Monat]')
    fig.tight_layout()
    file = base_path / "figures" / "well_extraction_monthly_stress.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


    return


if __name__ == "__main__":
    main()