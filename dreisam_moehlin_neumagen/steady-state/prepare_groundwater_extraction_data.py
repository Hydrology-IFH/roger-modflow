import pandas as pd
from pathlib import Path
import geopandas as gpd

base_path = Path(__file__).parent

path = base_path / "input" / "groundwater_extraction_.gpkg"
gdf = gpd.read_file(path)


gdf = gdf[['GW-Nummer', 'Ost', 'Nord', 'Gemeinde',
       'Nutzung11', '2013', '2014', '2015', '2016', '2017',
       '2018', '2019', '2020', '2021', '2022', '2023', 'Mittelwert', 'Zelle_x',
       'Zelle_y', 'geometry']]


gdf.columns = ['ID', 'x', 'y', 'municipality',
       'purpose', '2013', '2014', '2015', '2016', '2017',
       '2018', '2019', '2020', '2021', '2022', '2023', 'annual_average', 'cell_y',
       'cell_x', 'geometry']

# reorder columns
gdf = gdf[['ID', 'x', 'y', 'cell_x',
       'cell_y', 'municipality', 'purpose', '2013', '2014', '2015', '2016', '2017',
       '2018', '2019', '2020', '2021', '2022', '2023', 'annual_average', 'geometry']]

cell_x = gdf['cell_x'].values
cell_y = gdf['cell_y'].values
gdf['cell_x'] = cell_y
gdf['cell_y'] = cell_x

# replace german umlaut
gdf['municipality'] = gdf['municipality'].str.replace('ü', 'ue')
gdf['municipality'] = gdf['municipality'].str.replace('ö', 'oe')
gdf['municipality'] = gdf['municipality'].str.replace('ä', 'ae')
gdf['municipality'] = gdf['municipality'].str.replace('ß', 'ss')
gdf['purpose'] = gdf['purpose'].str.replace('ü', 'ue')
gdf['purpose'] = gdf['purpose'].str.replace('ö', 'oe')
gdf['purpose'] = gdf['purpose'].str.replace('Ö', 'oe')
gdf['purpose'] = gdf['purpose'].str.replace('ä', 'ae')
gdf['purpose'] = gdf['purpose'].str.replace('ß', 'ss')

# remove "stillgelegt"
cond = gdf['purpose'].str.contains('stillgelegt', case=False, na=False)
gdf = gdf[~cond]

# remove if more than 2 years of data is missing
gdf_ = gdf.loc[:, "2013":"2023"]
cond = gdf_.isnull().sum(axis=1) < 2
gdf = gdf.loc[cond, :]

# convert to m3/day
gdf['annual_average'] = gdf['annual_average'] / 365

# assign layer 
gdf['layer'] = 2
cond = gdf['municipality'].isin(['Kirchzarten', 'Oberried', 'Buchenbach', 'Stegen'])
gdf.loc[cond, 'layer'] = 3

# write to file
output_path = base_path / "input" / "groundwater_extraction.gpkg"
gdf.to_file(output_path, driver="GPKG")
# write to csv
output_path = base_path / "input" / "groundwater_extraction.csv"
gdf.drop(columns='geometry').to_csv(output_path, index=False, sep=';')