from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Union, Tuple, Optional


def remove_outliers(
    data: Union[pd.Series, pd.DataFrame],
    method: str = 'iqr',
    threshold: float = 1.5,
    window: Optional[int] = None,
    fill_method: str = 'interpolate'
) -> Tuple[Union[pd.Series, pd.DataFrame], pd.Series]:
    """
    Remove outliers from groundwater time series data.
    
    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Time series data with datetime index. If DataFrame, specify column name.
    method : str, default 'iqr'
        Method for outlier detection:
        - 'iqr': Interquartile range method
        - 'zscore': Z-score method (standard deviations from mean)
        - 'modified_zscore': Modified Z-score using median absolute deviation
        - 'rolling': Rolling window statistics
    threshold : float, default 1.5
        Threshold for outlier detection:
        - For 'iqr': multiplier for IQR (typically 1.5 or 3.0)
        - For 'zscore': number of standard deviations (typically 2-3)
        - For 'modified_zscore': MAD multiplier (typically 3.5)
        - For 'rolling': number of standard deviations from rolling mean
    window : int, optional
        Window size for rolling method (required if method='rolling')
    fill_method : str, default 'interpolate'
        How to handle removed outliers:
        - 'interpolate': Linear interpolation
        - 'nan': Leave as NaN
        - 'median': Replace with median
        - 'rolling_median': Replace with rolling median
    
    Returns
    -------
    cleaned_data : pd.Series or pd.DataFrame
        Data with outliers removed/replaced
    outlier_mask : pd.Series
        Boolean mask where True indicates outlier
    
    Examples
    --------
    >>> dates = pd.date_range('2020-01-01', periods=100, freq='D')
    >>> values = np.random.normal(10, 2, 100)
    >>> values[50] = 30  # Add outlier
    >>> ts = pd.Series(values, index=dates)
    >>> cleaned, mask = remove_outliers_groundwater(ts, method='iqr', threshold=1.5)
    >>> print(f"Found {mask.sum()} outliers")
    """
    
    # Convert to Series if needed
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("DataFrame must have exactly one column")
        series = data.iloc[:, 0].copy()
    else:
        series = data.copy()
    
    # Remove NaN values for calculation
    valid_mask = ~series.isna()
    valid_data = series[valid_mask]
    
    # Initialize outlier mask
    outlier_mask = pd.Series(False, index=series.index)
    
    # Detect outliers based on method
    if method == 'iqr':
        q1 = valid_data.quantile(0.25)
        q3 = valid_data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        outlier_mask[valid_mask] = (valid_data < lower_bound) | (valid_data > upper_bound)
    
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(valid_data, nan_policy='omit'))
        outlier_mask[valid_mask] = z_scores > threshold
    
    elif method == 'modified_zscore':
        median = valid_data.median()
        mad = np.median(np.abs(valid_data - median))
        modified_z_scores = 0.6745 * (valid_data - median) / mad
        outlier_mask[valid_mask] = (np.abs(modified_z_scores) > threshold)
    
    elif method == 'rolling':
        if window is None:
            raise ValueError("window parameter required for rolling method")
        rolling_mean = series.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = series.rolling(window=window, center=True, min_periods=1).std()
        lower_bound = rolling_mean - threshold * rolling_std
        upper_bound = rolling_mean + threshold * rolling_std
        outlier_mask = (series < lower_bound) | (series > upper_bound)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create cleaned data
    cleaned_series = series.copy()
    cleaned_series[outlier_mask] = np.nan
    
    # Fill outliers based on fill_method
    if fill_method == 'interpolate':
        cleaned_series = cleaned_series.interpolate(method='linear', limit=30)
    elif fill_method == 'nan':
        pass  # Already set to NaN
    elif fill_method == 'median':
        median_value = series[~outlier_mask].median()
        cleaned_series[outlier_mask] = median_value
    elif fill_method == 'rolling_median':
        rolling_med = series.rolling(window=window or 7, center=True, min_periods=1).median()
        cleaned_series[outlier_mask] = rolling_med[outlier_mask]
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")
    
    # Convert back to DataFrame if input was DataFrame
    if isinstance(data, pd.DataFrame):
        cleaned_data = pd.DataFrame(cleaned_series, columns=data.columns)
    else:
        cleaned_data = cleaned_series
    
    return cleaned_data, outlier_mask


# Additional helper function for visualizing outliers
def plot_outliers(
    original_data: pd.Series,
    cleaned_data: pd.Series,
    outlier_mask: pd.Series,
    title: str = "Groundwater Time Series - Outlier Removal"
):
    """
    Plot original data, cleaned data, and highlight outliers.
    
    Requires matplotlib to be installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot original data
    ax.plot(original_data.index, original_data, 'o-', 
            label='Original', alpha=0.5, markersize=4)
    
    # Plot cleaned data
    ax.plot(cleaned_data.index, cleaned_data, 'o-', 
            label='Cleaned', alpha=0.7, markersize=4)
    
    # Highlight outliers
    outliers = original_data[outlier_mask]
    ax.scatter(outliers.index, outliers, color='red', 
               s=100, marker='x', label='Outliers', zorder=5)
    
    ax.set_xlabel('Zeit')
    ax.set_ylabel('GW-Hoehe [m ü. M.]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return fig, ax

base_path = Path(__file__).parent

file = base_path / "observations" / "wasserstand_filter.gpkg"
gdf_dmn = gpd.read_file(file)
# select if Station ID contains Stadt Freiburg
gdf_dmn_fr = gdf_dmn[gdf_dmn["Station ID"].str.contains("Stadt Freiburg")]
gdf_dmn_lubw = gdf_dmn[gdf_dmn["Station ID"].str.contains("LUBW")]
gdf_dmn_bn = gdf_dmn[gdf_dmn["Station ID"].str.contains("Badenova")]

# split Station ID and take the first part as index
gdf_dmn_fr.index = gdf_dmn_fr["Station ID"].str.split(" - ").str[0]
gdf_dmn_fr.index.name = "station_id"
gdf_dmn_fr["provider"] = "Stadt Freiburg"
gdf_dmn_lubw.index = gdf_dmn_lubw["Station ID"].str.split(" - ").str[0]
gdf_dmn_lubw.index.name = "station_id"
gdf_dmn_lubw["provider"] = "LUBW"
gdf_dmn_bn.index = gdf_dmn_bn["Station ID"].str.split(" - ").str[0]
gdf_dmn_bn.index.name = "station_id"
gdf_dmn_bn["provider"] = "Badenova"

# rename column "Startdatum" to "von" in all gdfs
gdf_dmn_fr = gdf_dmn_fr.rename(columns={"Startdatum": "start_date"})
gdf_dmn_lubw = gdf_dmn_lubw.rename(columns={"Startdatum": "start_date"})
gdf_dmn_bn = gdf_dmn_bn.rename(columns={"Startdatum": "start_date"})

gdf_dmn_fr = gdf_dmn_fr[["start_date", "provider", "geometry"]]
gdf_dmn_lubw = gdf_dmn_lubw[["start_date", "provider", "geometry"]]
gdf_dmn_bn = gdf_dmn_bn[["start_date", "provider", "geometry"]]

# merge all gdfs
gdf_dmn = pd.concat([gdf_dmn_fr, gdf_dmn_lubw, gdf_dmn_bn])

file = base_path / "observations" / "lubw_gw.gpkg"
gdf_lubw = gpd.read_file(file)
gdf_lubw.index = gdf_lubw["GW-Nummer"]
# rename column "von" to "start_date" in all gdfs
gdf_lubw = gdf_lubw.rename(columns={"von": "start_date"})
gdf_lubw.index.name = "station_id"
gdf_lubw["provider"] = "LUBW"
gdf_lubw = gdf_lubw[["start_date", "provider", "geometry"]]

file = base_path / "observations" / "fr_gw.gpkg"
gdf_fr = gpd.read_file(file)
gdf_fr.index = gdf_fr["GWM"]
gdf_fr.index.name = "station_id"
gdf_fr = gdf_fr.rename(columns={"von": "start_date"})
gdf_fr["provider"] = "FR"
gdf_fr = gdf_fr[["start_date", "provider", "geometry"]]

# merge all gdfs
gdf_lubw_fr = pd.concat([gdf_lubw, gdf_fr])
# drop duplicates
gdf_lubw_fr = gdf_lubw_fr[~gdf_lubw_fr.index.duplicated(keep="first")]

# final merge of all groundwater observation points
gdf_gw = pd.concat([gdf_dmn, gdf_lubw_fr])
# drop duplicates
gdf_gw = gdf_gw[~gdf_gw.index.duplicated(keep="first")]

# write to file
file = base_path / "observations" / "groundwater_observation_wells.gpkg"
gdf_gw.to_file(file, driver="GPKG")
gdf_gw["end_date"] = ""

# iterate over all observation points and read the time series
date_time = pd.date_range(start="1990-01-01", end="2024-12-31", freq="D")
list_df = []
for station_id, row in gdf_gw.iterrows():
    provider = row["provider"]
    start_date = row["start_date"]
    print(f"Processing station {station_id} from provider {provider}")
    if provider == "Stadt Freiburg":
        _station_id = station_id.replace("/", "_")
        file_ts = base_path / "observations" / "time_series_raw" / "LUBW_Stadt_Freiburg" / f"{_station_id}.csv"
        df_ts = pd.read_csv(file_ts, sep=";", skiprows=3, encoding='latin1')
        try:
            df_ts = df_ts.rename(columns={"Datum": "date", "GW_Hoehe": "gw_head"})
            df_ts.loc[:, "date"] = pd.to_datetime(df_ts["date"], format="%Y-%m-%d")
        except KeyError:
            df_ts = df_ts.rename(columns={"Messzeitpunkt": "date", "Messwert": "gw_head"})
            df_ts.loc[:, "date"] = pd.to_datetime(df_ts["date"], format="%Y-%m-%d")
        except ValueError:
            df_ts = df_ts.rename(columns={"Messzeitpunkt": "date", "Messwert": "gw_head"})
            df_ts.loc[:, "date"] = pd.to_datetime(df_ts["date"], format="%d.%m.%Y")
        df_ts = df_ts.set_index("date")
        df = pd.DataFrame(index=date_time)
        df = df.join(df_ts["gw_head"].to_frame())
        df.columns = [f"{_station_id}"]
        list_df.append(df)
        gdf_gw.at[station_id, "start_date"] = df_ts.index.min().strftime("%d-%m-%Y")
        gdf_gw.at[station_id, "end_date"] = df_ts.index.max().strftime("%d-%m-%Y")

    elif provider == "LUBW":
        _station_id = station_id.replace("/", "_")
        file_ts = base_path / "observations" / "time_series_raw" / "LUBW_Stadt_Freiburg" / f"{_station_id}.csv"
        df_ts = pd.read_csv(file_ts, sep=";", skiprows=3, encoding='latin1')
        try:
            df_ts = df_ts.rename(columns={"Datum": "date", "GW_Hoehe": "gw_head"})
            df_ts.loc[:, "date"] = pd.to_datetime(df_ts["date"], format="%Y-%m-%d")
        except KeyError:
            try:
                df_ts = df_ts.rename(columns={"Messzeitpunkt": "date", "Messwert": "gw_head"})
                df_ts.loc[:, "date"] = pd.to_datetime(df_ts["date"], format="%Y-%m-%d")
            except ValueError:
                df_ts = df_ts.rename(columns={"Messzeitpunkt": "date", "Messwert": "gw_head"})
                df_ts.loc[:, "date"] = pd.to_datetime(df_ts["date"], format="%d.%m.%Y")
        df_ts = df_ts.set_index("date")
        df_ts["station_id"] = station_id
        df = pd.DataFrame(index=date_time)
        df = df.join(df_ts["gw_head"].to_frame())
        df.columns = [f"{_station_id}"]
        list_df.append(df)
        gdf_gw.at[station_id, "start_date"] = df_ts.index.min().strftime("%d-%m-%Y")
        gdf_gw.at[station_id, "end_date"] = df_ts.index.max().strftime("%d-%m-%Y")
    elif provider == "Badenova":
        file_ts = base_path / "observations" / "time_series_raw" / "Badenova" / f"{station_id}.csv"
        df_ts = pd.read_csv(file_ts, sep=",")
        df_ts.loc[:, "date"] = pd.to_datetime(df_ts["Startzeit"], format="%Y-%m-%d %H:%M:%S")
        df_ts = df_ts.rename(columns={"Interner Wert": "gw_head"})
        df_ts = df_ts.set_index("date")
        df_ts["station_id"] = station_id
        df = pd.DataFrame(index=date_time)
        df = df.join(df_ts["gw_head"].to_frame())
        df.columns = [f"{station_id}"]
        list_df.append(df)
        gdf_gw.at[station_id, "start_date"] = df_ts.index.min().strftime("%Y-%m-%d")
        gdf_gw.at[station_id, "end_date"] = df_ts.index.max().strftime("%Y-%m-%d")

# concatenate all dataframes
df_gw_heads = pd.concat(list_df, axis=1)
# write to file
file = base_path / "observations" / "groundwater_head_time_series.csv"
df_gw_heads.to_csv(file, sep=";")

file = base_path / "observations" / "groundwater_observation_wells.gpkg"
gdf_gw["xcoord"] = gdf_gw.geometry.x
gdf_gw["ycoord"] = gdf_gw.geometry.y
gdf_gw = gdf_gw[["xcoord", "ycoord","start_date", "end_date", "provider", "geometry"]]
gdf_gw.to_file(file, driver="GPKG")

df_gw_heads_filled = pd.DataFrame(index=df_gw_heads.index)
# plot time series of all observation points
for column in df_gw_heads.columns:
    fig, ax = plt.subplots(figsize=(6, 3))
    cleaned_data, _mask_outlier = remove_outliers(
        data=df_gw_heads[column],
        method="iqr",
        threshold=1.5,
        fill_method="interpolate")
    df_gw_heads_filled.loc[:, column] = cleaned_data
    df_gw_heads_filled = df_gw_heads_filled.copy()
    ax.scatter(df_gw_heads.index, df_gw_heads[column], color="blue", s=10, alpha=0.3, label="Original Data", zorder=2)
    ax.scatter(df_gw_heads_filled.index[_mask_outlier.values], df_gw_heads_filled.loc[_mask_outlier.values, column].values, color="red", s=20, label="Removed Outliers", zorder=3)
    ax.plot(df_gw_heads_filled.index, df_gw_heads_filled[column], color="black", linewidth=1, markersize=1, marker="o", label="Cleaned Data", zorder=1)
    ax.set_xlim(df_gw_heads_filled.index.min(), df_gw_heads_filled.index.max())
    ax.set_xlabel("Zeit")
    ax.set_ylabel("GW-Hoehe [m ü. M.]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    file = base_path / "figures" / "groundwater_time_series" / f"{column}_time_series.png"
    fig.savefig(file, dpi=300, bbox_inches="tight")
    plt.close(fig)

file = base_path / "observations" / "groundwater_head_time_series_filled.csv"
df_gw_heads_filled.to_csv(file, sep=";")