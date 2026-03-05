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

    # load interpolated groundwater heads
    src = rasterio.open(str(base_path / "input" / "groundwater_heads_interpolated_50m.tif"))
    gw_heads_interpolated_avg = src.read(1)

    # load dem with 5m resolution
    src = rasterio.open(str(base_path / "input" / "dem_5m.tif"))
    dem_5m = src.read(1)
    dem_5m = np.where(dem_5m < 0, np.nan, dem_5m)

    # load locations of drinking water wells
    path = base_path / "input" / "groundwater_extraction_drinking_water_supply.csv"
    df_drinking_water_wells = pd.read_csv(path, sep=';')

    # load model parameters
    path = base_path / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    xcoords = ds_params.x.values
    ycoords = ds_params.y.values
    topography = ds_params['topography'].values
    mask = np.isfinite(topography)
    mask_porous = ds_params['mask_porous_aquifer'].values == 1
    mask_fissured = (ds_params['mask_black_forest'].values == 1) & np.isfinite(topography)

    _ids_to_remove = ['0190_069-0', '0118_070-0', '0122_069-1', '0831_018-1', '0131_069-2', '2311_120-2']  # wells with inconsistent data
    grid_extent = (xcoords[0], xcoords[-1], ycoords[0], ycoords[-1])

    # load catchment boundary
    path = base_path / "input" / "active_area.shp"
    catchment_boundary = gpd.read_file(path)

    # load groundwater head time series
    file = base_path / "observations" / "groundwater_head_time_series_filled.csv"
    df_gw_heads = pd.read_csv(file, index_col=0, sep=';')
    df_gw_heads.index = pd.to_datetime(df_gw_heads.index)
    # calculate data coverage for each well and year
    # data_coverage = (df_gw_heads.notna().groupby(df_gw_heads.index.year).sum() / 365.25) * 100  # in percent
    data_coverage = df_gw_heads.notna().groupby(df_gw_heads.index.year).mean() * 100
    data_coverage.loc['average', :] = data_coverage.mean(axis=0)
    data_coverage.loc[:, 'average'] = data_coverage.mean(axis=1)
    # save data coverage to csv
    file = base_path / "observations" / "data_coverage_wells.csv"
    data_coverage.to_csv(file, sep=';')

    # get ids of wells if average data coverage is less greater than 80%
    ids_to_remove = data_coverage.columns[data_coverage.loc['average', :] < 80]
    # add wells with inconsistent data to ids_to_remove
    ids_to_remove = ids_to_remove.tolist() + _ids_to_remove
    # remove duplicates
    ids_to_remove = list(set(ids_to_remove))

    # remove wells with inconsistent data
    df_gw_heads = df_gw_heads[
        [col for col in df_gw_heads.columns if col not in ids_to_remove]
    ]

    # Load observation wells
    file = base_path / "observations" / "groundwater_observation_wells.gpkg"
    groundwater_observation_wells = gpd.read_file(file)
    groundwater_observation_wells['station_id'] = groundwater_observation_wells['station_id'].str.replace('/', '_')
    # clip to catchment boundary
    groundwater_observation_wells = gpd.clip(groundwater_observation_wells, catchment_boundary)

    # remove wells with inconsistent data
    groundwater_observation_wells = groundwater_observation_wells[
        ~groundwater_observation_wells['station_id'].isin(ids_to_remove)
    ]
    df_gw_heads = df_gw_heads.loc[:, df_gw_heads.columns.isin(groundwater_observation_wells['station_id'])]

    # remove wells with little data coverage
    groundwater_observation_wells = groundwater_observation_wells[
        groundwater_observation_wells['station_id'].isin(df_gw_heads.columns)
    ]
    groundwater_observation_wells['avg_gw_depth'] = np.nan
    groundwater_observation_wells['avg_gw_head'] = np.nan

    # extract topography at observation well locations
    df_gw_heads_topo = pd.DataFrame(index=df_gw_heads.columns, columns=['x', 'y', 'topography'])
    for idx, row in groundwater_observation_wells.iterrows():
        x = row.geometry.x
        y = row.geometry.y
        row_idx, col_idx = src.index(x, y)
        topo = dem_5m[row_idx, col_idx]
        df_gw_heads_topo.loc[row['station_id'], 'x'] = x
        df_gw_heads_topo.loc[row['station_id'], 'y'] = y
        df_gw_heads_topo.loc[row['station_id'], 'topography'] = topo

    df_gw_depths = pd.DataFrame(index=df_gw_heads.index, columns=df_gw_heads.columns)
    # calculate groundwater depths
    for well_id in df_gw_heads.columns:
        topo = df_gw_heads_topo.loc[well_id, 'topography']
        gw_depth = topo - df_gw_heads[well_id]
        gw_depth = np.where(gw_depth < 0, np.nan, gw_depth) # set negative depths to 0
        df_gw_depths[well_id] = gw_depth
        groundwater_observation_wells.loc[groundwater_observation_wells['station_id'] == well_id, 'avg_gw_head'] = df_gw_heads[well_id].loc['1990-01-01':'2019-12-31'].mean()
        groundwater_observation_wells.loc[groundwater_observation_wells['station_id'] == well_id, 'avg_gw_depth'] = df_gw_depths[well_id].loc['1990-01-01':'2019-12-31'].mean()

    # drop columns with all NaN values
    df_gw_depths = df_gw_depths.dropna(axis=1, how='all')
    # drop year 2024
    df_gw_depths = df_gw_depths[df_gw_depths.index.year < 2024]

    years = df_gw_depths.index.year.unique().tolist()
    alphas = np.linspace(0.1, 1.0, len(years))
    # plot all time series in a single figure for visual inspection
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for well_id in df_gw_depths.columns:
        for i, year in enumerate(years):
            # increase alpha with number of years to improve visibility
            df_year = df_gw_depths[
                (df_gw_depths.index.year == year)
            ]
            # increase alpha with number of years to improve visibility
            ax.plot(df_year.index, df_year[well_id], color='black', alpha=alphas[i])
    ax.set_xlabel('Zeit', fontsize=12)
    ax.set_ylabel('GWFA [m]', fontsize=12)
    ax.set_ylim(0, )
    ax.set_xlim([df_gw_depths.index.min(), df_gw_depths.index.max()])
    ax.invert_yaxis()
    ax.tick_params("x", rotation=20)
    fig.tight_layout()
    file = base_path / "figures" / "groundwater_time_series" / "gw_depth_observations_all_years.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    # calculate average depth for each well
    avg_depths = df_gw_depths.loc['1990-01-01':'2019-12-31'].mean(axis=0)
    # calculate average head for each well
    avg_heads = df_gw_heads.loc['1990-01-01':'2019-12-31'].mean(axis=0)
    # calculate anomalies for each well
    anomalies_gw_depths = ((df_gw_depths - avg_depths) / avg_depths) * 100 * (-1) # in percent
    anomalies_gw_heads_m = df_gw_heads - avg_heads

    # plot average, median, min and max anomalies for all wells
    output_folder = base_path / "figures" / "gw_depth_anomalies_observations"
    output_folder.mkdir(parents=True, exist_ok=True) 
    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(anomalies_gw_depths.index, anomalies_gw_depths.median(axis=1), label='Median Anomaly', color='black')
    ax.hlines(0, anomalies_gw_depths.index[0], anomalies_gw_depths.index[-1], colors='#756bb1', linestyles='dashed')
    # fill area between percentiles 25 and 75
    ax.fill_between(anomalies_gw_depths.index,
                    anomalies_gw_depths.quantile(0.25, axis=1),
                    anomalies_gw_depths.quantile(0.75, axis=1),
                    color='#756bb1', alpha=0.75, label='25-75 Perzentil')
    # fill area between percentiles 5 and 95
    ax.fill_between(anomalies_gw_depths.index,
                    anomalies_gw_depths.quantile(0.05, axis=1),
                    anomalies_gw_depths.quantile(0.95, axis=1),
                    color='#9e9ac8', alpha=0.5, label='5-95 Perzentil')
    # fill area between percentiles 1 and 99
    ax.fill_between(anomalies_gw_depths.index,
                    anomalies_gw_depths.quantile(0.01, axis=1),
                    anomalies_gw_depths.quantile(0.99, axis=1),
                    color='#9e9ac8', alpha=0.25, label='1-99 Perzentil')

    # format x-axis and use %y-%m format
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.set_ylim([-100, 100])
    ax.set_xlim([anomalies_gw_depths.index[0], anomalies_gw_depths.index[-1]])
    ax.set_xlabel('Zeit', fontsize=12)
    ax.set_ylabel('GWFA Anomalie [%]', fontsize=12)
    ax.tick_params("x", rotation=20)
    fig.tight_layout()
    file = output_folder / "gw_depth_anomalies_wells.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.plot(anomalies_gw_heads_m.index, anomalies_gw_heads_m.median(axis=1), label='Median Anomaly', color='black')
    ax.hlines(0, anomalies_gw_heads_m.index[0], anomalies_gw_heads_m.index[-1], colors='#756bb1', linestyles='dashed')
    # fill area between percentiles 25 and 75
    ax.fill_between(anomalies_gw_heads_m.index,
                    anomalies_gw_heads_m.quantile(0.25, axis=1),
                    anomalies_gw_heads_m.quantile(0.75, axis=1),
                    color='#756bb1', alpha=0.75, label='25-75 Perzentil')
    # fill area between percentiles 5 and 95
    ax.fill_between(anomalies_gw_heads_m.index,
                    anomalies_gw_heads_m.quantile(0.05, axis=1),
                    anomalies_gw_heads_m.quantile(0.95, axis=1),
                    color='#9e9ac8', alpha=0.5, label='5-95 Perzentil')
    # fill area between percentiles 1 and 99
    ax.fill_between(anomalies_gw_heads_m.index,
                    anomalies_gw_heads_m.quantile(0.01, axis=1),
                    anomalies_gw_heads_m.quantile(0.99, axis=1),
                    color='#9e9ac8', alpha=0.25, label='1-99 Perzentil')

    # format x-axis and use %y-%m format
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.set_ylim([-5, 5])
    ax.set_xlim([anomalies_gw_heads_m.index[0], anomalies_gw_heads_m.index[-1]])
    ax.set_xlabel('Zeit', fontsize=12)
    ax.set_ylabel('GW-Hoehe Anomalie [m]', fontsize=12)
    ax.tick_params("x", rotation=20)
    fig.tight_layout()
    file = output_folder / "gw_head_anomalies_wells_m.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    anomalies_gw_depths_2013_2023 = anomalies_gw_depths.loc['2013-01-01':'2023-12-31']
    ax.plot(anomalies_gw_depths_2013_2023.index, anomalies_gw_depths_2013_2023.median(axis=1), label='Median Anomaly', color='black')
    ax.hlines(0, anomalies_gw_depths_2013_2023.index[0], anomalies_gw_depths_2013_2023.index[-1], colors='#756bb1', linestyles='dashed')
    # fill area between percentiles 25 and 75
    ax.fill_between(anomalies_gw_depths_2013_2023.index,
                    anomalies_gw_depths_2013_2023.quantile(0.25, axis=1),
                    anomalies_gw_depths_2013_2023.quantile(0.75, axis=1),
                    color='#756bb1', alpha=0.75, label='25-75 Perzentil')
    # fill area between percentiles 5 and 95
    ax.fill_between(anomalies_gw_depths_2013_2023.index,
                    anomalies_gw_depths_2013_2023.quantile(0.05, axis=1),
                    anomalies_gw_depths_2013_2023.quantile(0.95, axis=1),
                    color='#9e9ac8', alpha=0.5, label='5-95 Perzentil')
    # fill area between percentiles 1 and 99
    ax.fill_between(anomalies_gw_depths_2013_2023.index,
                    anomalies_gw_depths_2013_2023.quantile(0.01, axis=1),
                    anomalies_gw_depths_2013_2023.quantile(0.99, axis=1),
                    color='#9e9ac8', alpha=0.25, label='1-99 Perzentil')

    # format x-axis and use %y-%m format
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.set_ylim([-100, 100])
    ax.set_xlim([anomalies_gw_depths_2013_2023.index[0], anomalies_gw_depths_2013_2023.index[-1]])
    ax.set_xlabel('Zeit', fontsize=12)
    ax.set_ylabel('GWFA Anomalie [%]', fontsize=12)
    ax.tick_params("x", rotation=20)
    fig.tight_layout()
    file = output_folder / "gw_depth_anomalies_wells_2013_2023.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    anomalies_gw_heads_m_2013_2023 = anomalies_gw_heads_m.loc['2013-01-01':'2023-12-31']
    ax.plot(anomalies_gw_heads_m_2013_2023.index, anomalies_gw_heads_m_2013_2023.median(axis=1), label='Median Anomaly', color='black')
    ax.hlines(0, anomalies_gw_heads_m_2013_2023.index[0], anomalies_gw_heads_m_2013_2023.index[-1], colors='#756bb1', linestyles='dashed')
    # fill area between percentiles 25 and 75
    ax.fill_between(anomalies_gw_heads_m_2013_2023.index,
                    anomalies_gw_heads_m_2013_2023.quantile(0.25, axis=1),
                    anomalies_gw_heads_m_2013_2023.quantile(0.75, axis=1),
                    color='#756bb1', alpha=0.75, label='25-75 Perzentil')
    # fill area between percentiles 5 and 95
    ax.fill_between(anomalies_gw_heads_m_2013_2023.index,
                    anomalies_gw_heads_m_2013_2023.quantile(0.05, axis=1),
                    anomalies_gw_heads_m_2013_2023.quantile(0.95, axis=1),
                    color='#9e9ac8', alpha=0.5, label='5-95 Perzentil')
    # fill area between percentiles 1 and 99
    ax.fill_between(anomalies_gw_heads_m_2013_2023.index,
                    anomalies_gw_heads_m_2013_2023.quantile(0.01, axis=1),
                    anomalies_gw_heads_m_2013_2023.quantile(0.99, axis=1),
                    color='#9e9ac8', alpha=0.25, label='1-99 Perzentil')

    # format x-axis and use %y-%m format
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.set_ylim([-5, 5])
    ax.set_xlim([anomalies_gw_heads_m_2013_2023.index[0], anomalies_gw_heads_m_2013_2023.index[-1]])
    ax.set_xlabel('Zeit', fontsize=12)
    ax.set_ylabel('GW-Hoehe Anomalie [m]', fontsize=12)
    ax.tick_params("x", rotation=20)
    fig.tight_layout()
    file = output_folder / "gw_head_anomalies_wells_m_2013_2023.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    anomalies_gw_depths_2017_2021 = anomalies_gw_depths.loc['2017-01-01':'2021-12-31']
    ax.plot(anomalies_gw_depths_2017_2021.index, anomalies_gw_depths_2017_2021.median(axis=1), label='Median Anomaly', color='black')
    ax.hlines(0, anomalies_gw_depths_2013_2023.index[0], anomalies_gw_depths_2013_2023.index[-1], colors='#756bb1', linestyles='dashed')
    # fill area between percentiles 25 and 75
    ax.fill_between(anomalies_gw_depths_2017_2021.index,
                    anomalies_gw_depths_2017_2021.quantile(0.25, axis=1),
                    anomalies_gw_depths_2017_2021.quantile(0.75, axis=1),
                    color='#756bb1', alpha=0.75, label='25-75 Perzentil')
    # fill area between percentiles 5 and 95
    ax.fill_between(anomalies_gw_depths_2017_2021.index,
                    anomalies_gw_depths_2017_2021.quantile(0.05, axis=1),
                    anomalies_gw_depths_2017_2021.quantile(0.95, axis=1),
                    color='#9e9ac8', alpha=0.5, label='5-95 Perzentil')
    # fill area between percentiles 1 and 99
    ax.fill_between(anomalies_gw_depths_2017_2021.index,
                    anomalies_gw_depths_2017_2021.quantile(0.01, axis=1),
                    anomalies_gw_depths_2017_2021.quantile(0.99, axis=1),
                    color='#9e9ac8', alpha=0.25, label='1-99 Perzentil')
    # format x-axis and use %y-%m format
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.set_ylim([-100, 100])
    ax.set_xlim([anomalies_gw_depths_2017_2021.index[0], anomalies_gw_depths_2017_2021.index[-1]])
    ax.set_xlabel('Zeit', fontsize=12)
    ax.set_ylabel('GWFA Anomalie [%]', fontsize=12)
    ax.tick_params("x", rotation=20)
    fig.tight_layout()
    file = output_folder / "gw_depth_anomalies_wells_2017_2021.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    anomalies_gw_heads_m_2017_2021 = anomalies_gw_heads_m.loc['2017-01-01':'2021-12-31']
    ax.plot(anomalies_gw_heads_m_2017_2021.index, anomalies_gw_heads_m_2017_2021.median(axis=1), label='Median Anomaly', color='black')
    ax.hlines(0, anomalies_gw_heads_m_2017_2021.index[0], anomalies_gw_heads_m_2017_2021.index[-1], colors='#756bb1', linestyles='dashed')
    # fill area between percentiles 25 and 75
    ax.fill_between(anomalies_gw_heads_m_2017_2021.index,
                    anomalies_gw_heads_m_2017_2021.quantile(0.25, axis=1),
                    anomalies_gw_heads_m_2017_2021.quantile(0.75, axis=1),
                    color='#756bb1', alpha=0.75, label='25-75 Perzentil')
    # fill area between percentiles 5 and 95
    ax.fill_between(anomalies_gw_heads_m_2017_2021.index,
                    anomalies_gw_heads_m_2017_2021.quantile(0.05, axis=1),
                    anomalies_gw_heads_m_2017_2021.quantile(0.95, axis=1),
                    color='#9e9ac8', alpha=0.5, label='5-95 Perzentil')
    # fill area between percentiles 1 and 99
    ax.fill_between(anomalies_gw_heads_m_2017_2021.index,
                    anomalies_gw_heads_m_2017_2021.quantile(0.01, axis=1),
                    anomalies_gw_heads_m_2017_2021.quantile(0.99, axis=1),
                    color='#9e9ac8', alpha=0.25, label='1-99 Perzentil')
    # format x-axis and use %y-%m format
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    ax.set_ylim([-5, 5])
    ax.set_xlim([anomalies_gw_heads_m_2017_2021.index[0], anomalies_gw_heads_m_2017_2021.index[-1]])
    ax.set_xlabel('Zeit', fontsize=12)
    ax.set_ylabel('GW-Hoehe Anomalie [m]', fontsize=12)
    ax.tick_params("x", rotation=20)
    fig.tight_layout()
    file = output_folder / "gw_head_anomalies_wells_m_2017_2021.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    # loop over years and plot anomalies for each year
    for year in range(anomalies_gw_depths.index.year.min(), anomalies_gw_depths.index.year.max() + 1):
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        df_year = anomalies_gw_depths[
            (anomalies_gw_depths.index.year == year)
        ]
        ax.plot(df_year.index, df_year.median(axis=1), label='Median Anomaly', color='black')
        ax.hlines(0, df_year.index[0], df_year.index[-1], colors='#756bb1', linestyles='dashed')
        # fill area between percentiles 25 and 75
        ax.fill_between(df_year.index,
                        df_year.quantile(0.25, axis=1),
                        df_year.quantile(0.75, axis=1),
                        color='#756bb1', alpha=0.75, label='25-75 Perzentil')
        # fill area between percentiles 5 and 95
        ax.fill_between(df_year.index,
                        df_year.quantile(0.05, axis=1),
                        df_year.quantile(0.95, axis=1),
                        color='#9e9ac8', alpha=0.5, label='5-95 Perzentil')    
            # fill area between percentiles 1 and 99
        ax.fill_between(df_year.index,
                        df_year.quantile(0.01, axis=1),
                        df_year.quantile(0.99, axis=1),
                        color='#9e9ac8', alpha=0.25, label='1-99 Perzentil')    
        # format x-axis and use %y-%m format
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        ax.set_ylim([-100, 100])
        ax.set_xlim([datetime.datetime(year, 1, 1), datetime.datetime(year, 12, 31)])
        ax.set_xlabel('Zeit', fontsize=12)
        ax.set_ylabel('GWFA Anomalie [%]', fontsize=12)
        ax.tick_params("x", rotation=20)
        fig.tight_layout()
        file = output_folder / f"gw_depth_anomalies_wells_{year}.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

    for year in range(anomalies_gw_heads_m.index.year.min(), anomalies_gw_heads_m.index.year.max() + 1):
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        df_year = anomalies_gw_heads_m[
            (anomalies_gw_heads_m.index.year == year)
        ]
        ax.plot(df_year.index, df_year.median(axis=1), label='Median Anomaly', color='black')
        ax.hlines(0, df_year.index[0], df_year.index[-1], colors='#756bb1', linestyles='dashed')
        # fill area between percentiles 25 and 75
        ax.fill_between(df_year.index,
                        df_year.quantile(0.25, axis=1),
                        df_year.quantile(0.75, axis=1),
                        color='#756bb1', alpha=0.75, label='25-75 Perzentil')
        # fill area between percentiles 5 and 95
        ax.fill_between(df_year.index,
                        df_year.quantile(0.05, axis=1),
                        df_year.quantile(0.95, axis=1),
                        color='#9e9ac8', alpha=0.5, label='5-95 Perzentil')    
            # fill area between percentiles 1 and 99
        ax.fill_between(df_year.index,
                        df_year.quantile(0.01, axis=1),
                        df_year.quantile(0.99, axis=1),
                        color='#9e9ac8', alpha=0.25, label='1-99 Perzentil')    
        # format x-axis and use %y-%m format
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
        ax.set_ylim([-5, 5])
        ax.set_xlim([datetime.datetime(year, 1, 1), datetime.datetime(year, 12, 31)])
        ax.set_xlabel('Zeit', fontsize=12)
        ax.set_ylabel('GW-Hoehe Anomalie [m]', fontsize=12)
        ax.tick_params("x", rotation=20)
        fig.tight_layout()
        file = output_folder / f"gw_head_anomalies_wells_m_{year}.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

    # get year with lowest median anomaly
    median_anomalies = anomalies_gw_depths.median(axis=1)
    year_lowest_median = median_anomalies.idxmin().year

    # get year with lowest 95th percentile anomaly
    percentile_95_anomalies = anomalies_gw_depths.quantile(0.95, axis=1)
    year_lowest_95th = percentile_95_anomalies.idxmin().year

    # load monthly kriging results
    file = base_path / "observations" / "monthly_spatio_temporal_universal_kriging.nc"
    ds_monthly_interpolated_gw_heads = xr.open_dataset(file, decode_times=False)
    # convert time variable to datetime objects
    dates = pd.date_range(start="1990-01-01", periods=ds_monthly_interpolated_gw_heads.sizes["Time"], freq="ME")

    # calculate average head over time
    gw_heads_porous = ds_monthly_interpolated_gw_heads["interpolated_gw_heads_porous"].values
    gw_heads_porous_1990_2019 = ds_monthly_interpolated_gw_heads["interpolated_gw_heads_porous"].sel(Time=slice(0, 12*30)).values  # first 30 years (199
    variance_porous = ds_monthly_interpolated_gw_heads["uncertainty_porous"].values
    avg_variance_porous = np.nanmean(variance_porous, axis=0)
    mask_high_variance = avg_variance_porous > 1.5  # threshold for high variance
    gw_depths_porous = np.where(mask_high_variance[np.newaxis, :, :], np.nan, topography - gw_heads_porous)
    gw_depths_porous_1990_2019 = np.where(mask_high_variance[np.newaxis, :, :], np.nan, topography - gw_heads_porous_1990_2019)
    avg_gw_depths_porous_1990_2019 = np.nanmean(gw_depths_porous_1990_2019, axis=0)

    # calculate anomalies
    anomalies_gw_depths_porous = ((gw_depths_porous - avg_gw_depths_porous_1990_2019[np.newaxis, :, :]) / avg_gw_depths_porous_1990_2019[np.newaxis, :, :]) * 100 * (-1)  # in percent
    anomalies_gw_depths_porous_m = (gw_depths_porous - avg_gw_depths_porous_1990_2019[np.newaxis, :, :]) * (-1)  # in meters

    output_folder = base_path / "figures" / "gw_depth_anomalies_kriging"
    output_folder.mkdir(parents=True, exist_ok=True)
    xcoords = ds_monthly_interpolated_gw_heads.x.values
    ycoords = ds_monthly_interpolated_gw_heads.y.values

    rows, cols = np.where((np.isfinite(avg_gw_depths_porous_1990_2019)) & mask_porous)
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    _grid_extent = (xcoords[x1], xcoords[x2], ycoords[y2], ycoords[y1])

    # remove wells outside of the area with data
    mask_wells = np.isfinite(avg_gw_depths_porous_1990_2019) & mask_porous
    for idx, row in df_drinking_water_wells.iterrows():
        col_idx = row.cell_x
        row_idx = row.cell_y
        if not mask_wells[row_idx, col_idx]:
            df_drinking_water_wells = df_drinking_water_wells.drop(idx)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    avg_variance_porous = np.where(mask_porous, avg_variance_porous, np.nan)
    avg_variance_porous = avg_variance_porous[y1:y2+1, x1:x2+1]  # crop to area with data
    ax.imshow(avg_variance_porous, cmap="Oranges", extent=_grid_extent, vmin=0, vmax=1.5)
    fig.colorbar(ax.images[0], ax=ax, label="Varianz [m²]")
    ax.set_xlabel('X-Koordinate', fontsize=12)
    ax.set_ylabel('Y-Koordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    output_file = output_folder / "avg_variance_porous.png"
    fig.savefig(output_file)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    avg_gw_depths_porous_1990_2019 = np.where(mask_porous, avg_gw_depths_porous_1990_2019, np.nan)
    avg_gw_depths_porous_1990_2019 = avg_gw_depths_porous_1990_2019[y1:y2+1, x1:x2+1]  # crop to area with data
    ax.imshow(avg_gw_depths_porous_1990_2019, cmap="viridis", extent=_grid_extent, vmin=0, vmax=20)
    ax.scatter(df_drinking_water_wells['x-coordinate'], df_drinking_water_wells['y-coordinate'], c='magenta', s=20, marker='^', label='Trinkwasserbrunnen')
    ax.legend(loc='lower right', fontsize=11, facecolor='white', edgecolor='white')
    fig.colorbar(ax.images[0], ax=ax, label="Durchschnittlicher\n GWFA [m]")
    ax.set_xlabel('X-Koordinate', fontsize=12)
    ax.set_ylabel('Y-Koordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    output_file = output_folder / "avg_gw_depths_porous_1990_2019.png"
    fig.savefig(output_file)
    plt.close(fig)

    for i, time in enumerate(dates):
        if i > (12 * 23) - 1:
            anomaly = anomalies_gw_depths_porous[i, :, :]
            anomaly = np.where(mask_porous, anomaly, np.nan)  # apply land surface mask
            # interpolate NaN values for better visualization
            nan_mask = np.isnan(anomaly)
            anomaly_interp = anomaly.copy()
            anomaly_interp[nan_mask] = sp.ndimage.generic_filter(
                anomaly,
                function=lambda x: np.nanmean(x),
                size=3,
                mode='nearest'
            )[nan_mask]
            anomaly = anomaly_interp
            anomaly = np.where(mask_porous, anomaly, np.nan)
            anomaly = anomaly[y1:y2+1, x1:x2+1]  # crop to area with data

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.imshow(anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=_grid_extent)
            ax.scatter(df_drinking_water_wells['x-coordinate'], df_drinking_water_wells['y-coordinate'], c='magenta', s=20, marker='^', label='Trinkwasserbrunnen')
            ax.legend(loc='lower right', fontsize=11, facecolor='white', edgecolor='white')
            fig.colorbar(ax.images[0], ax=ax, label="GWFA Anomalie [%]")
            ax.set_xlabel('X-Koordinate', fontsize=12)
            ax.set_ylabel('Y-Koordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{time.strftime('%Y-%m')}", fontsize=14)
            output_file = output_folder / f"gw_depth_anomalies_{time.strftime('%Y-%m')}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            anomaly = anomalies_gw_depths_porous_m[i, :, :]
            anomaly = np.where(mask_porous, anomaly, np.nan)  # apply land surface mask
            # # interpolate NaN values for better visualization
            # nan_mask = np.isnan(anomaly)
            # anomaly_interp = anomaly.copy()
            # anomaly_interp[nan_mask] = sp.ndimage.generic_filter(
            #     anomaly,
            #     function=lambda x: np.nanmean(x),
            #     size=3,
            #     mode='nearest'
            # )[nan_mask]
            # anomaly = anomaly_interp
            anomaly = anomaly[y1:y2+1, x1:x2+1]  # crop to area with data
            
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.imshow(anomaly, cmap="RdBu", vmin=-5, vmax=5, extent=_grid_extent)
            ax.scatter(df_drinking_water_wells['x-coordinate'], df_drinking_water_wells['y-coordinate'], c='magenta', s=20, marker='^', label='Trinkwasserbrunnen')
            ax.legend(loc='lower right', fontsize=11, facecolor='white', edgecolor='white')
            fig.colorbar(ax.images[0], ax=ax, label="GW-Hoehe Anomalie [m]")
            ax.set_xlabel('X-Koordinate', fontsize=12)
            ax.set_ylabel('Y-Koordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{time.strftime('%Y-%m')}", fontsize=14)
            output_file = output_folder / f"gw_head_anomalies_m_{time.strftime('%Y-%m')}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            gw_depths = gw_depths_porous[i, :, :]
            gw_depths = np.where(mask_porous, gw_depths, np.nan)  # apply land surface mask
            # interpolate NaN values for better visualization
            nan_mask = np.isnan(gw_depths)
            gw_depths_interp = gw_depths.copy()
            gw_depths_interp[nan_mask] = sp.ndimage.generic_filter(
                gw_depths,
                function=lambda x: np.nanmean(x),
                size=3,
                mode='nearest'
            )[nan_mask]
            gw_depths = gw_depths_interp
            gw_depths = np.where(mask_porous, gw_depths, np.nan)
            gw_depths = gw_depths[y1:y2+1, x1:x2+1]  # crop to area with data
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            ax.imshow(gw_depths, cmap="viridis", vmin=0, vmax=25, extent=_grid_extent)
            ax.scatter(df_drinking_water_wells['x-coordinate'], df_drinking_water_wells['y-coordinate'], c='magenta', s=20, marker='^', label='Trinkwasserbrunnen')
            ax.legend(loc='lower right', fontsize=11, facecolor='white', edgecolor='white')
            fig.colorbar(ax.images[0], ax=ax, label="GWFA [m]")
            ax.set_xlabel('X-Koordinate', fontsize=12)
            ax.set_ylabel('Y-Koordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{time.strftime('%Y-%m')}", fontsize=14)
            output_file = output_folder / f"gw_depth_{time.strftime('%Y-%m')}.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

    return


if __name__ == "__main__":
    main()
