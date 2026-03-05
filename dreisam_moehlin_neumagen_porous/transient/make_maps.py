from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib_map_utils.core.north_arrow import north_arrow
from matplotlib_map_utils.core.scale_bar import scale_bar
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent

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

    _ids_to_remove = ['0190_069-0', '0118_070-0', '0122_069-1', '0831_018-1', '0131_069-2', '2311_120-2']  # wells with inconsistent data
    grid_extent = (xcoords[0], xcoords[-1], ycoords[0], ycoords[-1])

    # load catchment boundary
    path = base_path / "input" / "active_area.shp"
    catchment_boundary = gpd.read_file(path)

    path = base_path / "input" / "mask_catchment.gpkg"
    catchment_boundary_porous = gpd.read_file(path)

    # clip dringking water wells to catchment boundary porous
    gdf_drinking_water_wells = gpd.GeoDataFrame(
        df_drinking_water_wells,
        geometry=gpd.points_from_xy(df_drinking_water_wells['x-coordinate'], df_drinking_water_wells['y-coordinate']),
        crs="EPSG:25832"
    )
    gdf_drinking_water_wells = gpd.clip(gdf_drinking_water_wells, catchment_boundary_porous)
    

    # load groundwater head time series
    file = base_path / "observations" / "groundwater_head_time_series_filled.csv"
    df_gw_heads = pd.read_csv(file, index_col=0, sep=';')
    df_gw_heads.index = pd.to_datetime(df_gw_heads.index)
    # calculate data coverage for each well and year
    # data_coverage = (df_gw_heads.notna().groupby(df_gw_heads.index.year).sum() / 365.25) * 100  # in percent
    data_coverage = df_gw_heads.notna().groupby(df_gw_heads.index.year).mean() * 100
    data_coverage.loc['average', :] = data_coverage.mean(axis=0)
    data_coverage.loc[:, 'average'] = data_coverage.mean(axis=1)

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

    # plot location of observation wells and use average groundwater depth as color
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    groundwater_observation_wells.plot(column="avg_gw_depth", ax=ax, cmap='viridis', vmin=0, vmax=15)
    ax.scatter(gdf_drinking_water_wells['x-coordinate'], gdf_drinking_water_wells['y-coordinate'], c='magenta', s=20, marker='^', label='Trinkwasserbrunnen')
    # add colorbar
    cbar = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=15))
    fig.colorbar(cbar, ax=ax, label="Mittlerer GWFA [m]", shrink=0.9)
    # add catchment boundary
    catchment_boundary_porous.boundary.plot(ax=ax, color='black', linewidth=1)
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=groundwater_observation_wells.crs)
    north_arrow(
    ax, scale=0.25, location="upper right", rotation={"crs": groundwater_observation_wells.crs, "reference": "center"}
    )
    scale_bar(ax, location="lower right", style="boxes", bar={"projection": groundwater_observation_wells.crs, "height": 0.05}, text = {"fontfamily": "monospace", "fontsize": 10})
    ax.set_xlabel('X-Koordinate', fontsize=12)
    ax.set_ylabel('Y-Koordinate', fontsize=12)

    fig.tight_layout()
    file = base_path / "figures" / "gw_observation_wells_map.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


    # 1) Load a world dataset and filter to Germany
    file = base_path / "input" / "ne_110m_admin_0_countries.shp"
    world = gpd.read_file(file)
    germany = world[world["SOVEREIGNT"] == "Germany"].copy()

    if germany.empty:
        raise RuntimeError("Germany not found in the Natural Earth dataset.")

    # 2) Reproject to Web Mercator for Contextily tiles
    germany_25832 = germany.to_crs(epsg=25832)

    # 3) Plot boundary and add basemap
    fig, ax = plt.subplots(figsize=(8, 10))

    germany_25832.boundary.plot(ax=ax, color="black", linewidth=2)

    # Set plot extent to Germany bounds (with a little padding)
    minx, miny, maxx, maxy = germany_25832.total_bounds
    pad_x = (maxx - minx) * 0.10
    pad_y = (maxy - miny) * 0.10
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    # Add basemap tiles (pick a provider you like)
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.Positron,  # try also: cx.providers.OpenStreetMap.Mapnik
        crs=germany_25832.crs,
        attribution_size=6,
    )
    # add catchment boundary
    catchment_boundary_porous.boundary.plot(ax=ax, color='orange', linewidth=2)

    ax.set_axis_off()
    fig.tight_layout()
    file = base_path / "figures" / "location_overview_map.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
