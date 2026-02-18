from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Union, Tuple, Optional


def remove_outliers(
    data: Union[pd.Series, pd.DataFrame],
    method: str = 'iqr',
    threshold: float = 1.5,
    window: Optional[int] = None,
    fill_method: str = 'interpolate',
    contamination: float = 0.05,
    use_temporal_features: bool = True
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
        - 'isolation_forest': Isolation Forest machine learning method
    threshold : float, default 1.5
        Threshold for outlier detection:
        - For 'iqr': multiplier for IQR (typically 1.5 or 3.0)
        - For 'zscore': number of standard deviations (typically 2-3)
        - For 'modified_zscore': MAD multiplier (typically 3.5)
        - For 'rolling': number of standard deviations from rolling mean
        - Not used for 'isolation_forest' (use contamination instead)
    window : int, optional
        Window size for rolling method (required if method='rolling')
        For 'isolation_forest': used for rolling statistics features (default: 7)
    fill_method : str, default 'interpolate'
        How to handle removed outliers:
        - 'interpolate': Linear interpolation
        - 'nan': Leave as NaN
        - 'median': Replace with median
        - 'rolling_median': Replace with rolling median
    contamination : float, default 0.05
        Only for 'isolation_forest': Expected proportion of outliers (0.01-0.5)
    use_temporal_features : bool, default True
        Only for 'isolation_forest': Whether to create temporal features
        (rate of change, rolling stats, etc.) for better detection
    
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
    >>> 
    >>> # Using IQR method
    >>> cleaned, mask = remove_outliers(ts, method='iqr', threshold=1.5)
    >>> 
    >>> # Using Isolation Forest
    >>> cleaned, mask = remove_outliers(ts, method='isolation_forest', contamination=0.05)
    >>> print(f"Found {mask.sum()} outliers")
    """
    
    # Convert to Series if needed
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            raise ValueError("DataFrame must have exactly one column")
        series = data.iloc[:, 0].copy()
    else:
        series = data.copy()

    # iterate over date and values
    _series = series.dropna()
    for i, item in enumerate(_series.items()):
        if i < len(_series) - 1:
            date1, value1 = item
            try:
                date2, value2 = _series.items()[i + 1]
                delta_days = (date2 - date1).days
                abs_diff = abs(value2 - value1)
                if delta_days <= 7 and abs_diff > 1:  # threshold for a week
                    series.loc[date2] = np.nan
                elif delta_days > 7 and delta_days <= 30 and abs_diff > 2:  # threshold for a month
                    series.loc[date2] = np.nan
            except TypeError:
                pass
    
    # Remove NaN values for calculation
    valid_mask = ~series.isna()
    valid_data = series[valid_mask]

    # if absolute difference between values within a week or a month is larger than a threshold, mark as outlier
    
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
        if mad == 0:
            # Handle case where MAD is zero (all values are identical)
            warnings.warn("MAD is zero, cannot compute modified z-scores. No outliers detected.")
            outlier_mask[:] = False
        else:
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
    
    elif method == 'isolation_forest':
        # Create feature dataframe
        df_features = pd.DataFrame(index=series.index)
        df_features['value'] = series
        
        if use_temporal_features:
            # Set default window if not provided
            if window is None:
                window = 7
            
            # Rate of change features
            df_features['rate_change'] = series.diff()
            df_features['rate_change_2'] = series.diff(periods=2)
            df_features['rate_change_abs'] = df_features['rate_change'].abs()
            
            # Rolling statistics (7-day and 30-day windows)
            for win in [window, min(30, len(series) // 3)]:
                df_features[f'rolling_mean_{win}'] = series.rolling(
                    window=win, center=True, min_periods=1
                ).mean()
                df_features[f'rolling_std_{win}'] = series.rolling(
                    window=win, center=True, min_periods=1
                ).std()
                df_features[f'deviation_from_mean_{win}'] = (
                    series - df_features[f'rolling_mean_{win}']
                )
            
            # Lag features
            for lag in [1, 7]:
                if lag < len(series):
                    df_features[f'lag_{lag}'] = series.shift(lag)
            
            # Seasonal features if datetime index
            if isinstance(series.index, pd.DatetimeIndex):
                df_features['day_of_year'] = series.index.dayofyear
                df_features['season_sin'] = np.sin(2 * np.pi * df_features['day_of_year'] / 365.25)
                df_features['season_cos'] = np.cos(2 * np.pi * df_features['day_of_year'] / 365.25)
                df_features = df_features.drop('day_of_year', axis=1)
        
        # Remove rows with NaN (from rolling/lag features)
        df_clean = df_features.dropna()
        
        if len(df_clean) < 10:
            warnings.warn(
                f"Not enough valid data points ({len(df_clean)}) for Isolation Forest. "
                "Returning original data without outlier detection."
            )
            outlier_mask[:] = False
        else:
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clean)
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                max_features=1.0,
                bootstrap=False,
                n_jobs=-1,
                verbose=0
            )
            
            # Predict outliers (-1 for outliers, 1 for inliers)
            predictions = iso_forest.fit_predict(X_scaled)
            
            # Update outlier mask
            outlier_mask.loc[df_clean.index] = (predictions == -1)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'iqr', 'zscore', 'modified_zscore', 'rolling', or 'isolation_forest'")
    
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
        win = window if window is not None else 7
        rolling_med = series.rolling(window=win, center=True, min_periods=1).median()
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
            df_ts.loc[:, "date"] = pd.to_datetime(df_ts["date"], format="%d.%m.%y")
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
                try:
                    df_ts.loc[:, "date"] = pd.to_datetime(df_ts["date"], format="%d.%m.%y")
                except ValueError:
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
# remove duplicate index
df_gw_heads = df_gw_heads[~df_gw_heads.index.duplicated(keep="first")]
# write to file
file = base_path / "observations" / "groundwater_head_time_series.csv"
df_gw_heads.to_csv(file, sep=";")

file = base_path / "observations" / "groundwater_observation_wells.gpkg"
gdf_gw["xcoord"] = gdf_gw.geometry.x
gdf_gw["ycoord"] = gdf_gw.geometry.y
gdf_gw = gdf_gw[["xcoord", "ycoord","start_date", "end_date", "provider", "geometry"]]
gdf_gw.to_file(file, driver="GPKG")

df_gw_heads_filled = pd.DataFrame(index=df_gw_heads.index)

# Create output directory if it doesn't exist
output_dir = base_path / "figures" / "groundwater_time_series"
output_dir.mkdir(parents=True, exist_ok=True)

# plot time series of all observation points
fill_method = "interpolate"
for column in df_gw_heads.columns:
    print(f"Processing outliers for station: {column}")
    
    fig, ax = plt.subplots(figsize=(6, 3))
    
    # Option 2: Use Isolation Forest (new robust method)
    try:
        cleaned_data, _mask_outlier = remove_outliers(
            data=df_gw_heads[column],
            method="isolation_forest",
            contamination=0.02,  # Expect 2% outliers
            window=7,  # Window for rolling features
            use_temporal_features=True,
            fill_method=fill_method
        )
        method_name = "Isolation Forest"
    except Exception as e:
        # Fallback to IQR if Isolation Forest fails
        print(f"Warning: Isolation Forest failed for {column}, using IQR. Error: {e}")
        cleaned_data, _mask_outlier = remove_outliers(
            data=df_gw_heads[column],
            method="iqr",
            threshold=2.0,
            fill_method=fill_method
        )
        method_name = "IQR"
    
    df_gw_heads_filled.loc[:, column] = cleaned_data
    df_gw_heads_filled = df_gw_heads_filled.copy()
    
    # Count outliers
    n_outliers = _mask_outlier.sum()
    outlier_pct = (n_outliers / len(df_gw_heads[column].dropna())) * 100 if len(df_gw_heads[column].dropna()) > 0 else 0
    
    # Plotting
    ax.scatter(df_gw_heads.index, df_gw_heads[column], color="blue", s=10, alpha=0.3, label="Original Data", zorder=1)
    ax.scatter(df_gw_heads.index[_mask_outlier.values], df_gw_heads.loc[_mask_outlier.values, column].values, color="red", s=20, label=f"Removed Outliers ({n_outliers})", zorder=3)
    if fill_method == "interpolate":
        ax.plot(df_gw_heads_filled.index, df_gw_heads_filled[column], color="black", linewidth=1, markersize=1, marker="o", label="Cleaned Data", zorder=2)
    elif fill_method in ["nan"]:
        ax.scatter(df_gw_heads_filled.index, df_gw_heads_filled[column], color="black", s=15, label="Cleaned Data", zorder=2)
    ax.set_xlim(df_gw_heads_filled.index.min(), df_gw_heads_filled.index.max())
    ax.set_xlabel("Zeit")
    ax.set_ylabel("GW-Hoehe [m ü. M.]")
    ax.set_title(f"{column} - {method_name} ({outlier_pct:.1f}% outliers)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    file = base_path / "figures" / "groundwater_time_series" / f"{column}_time_series.png"
    fig.savefig(file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  - Found {n_outliers} outliers ({outlier_pct:.1f}% of valid data)")

# remove duplicate index
df_gw_heads_filled = df_gw_heads_filled[~df_gw_heads_filled.index.duplicated(keep="first")]
file = base_path / "observations" / "groundwater_head_time_series_filled.csv"
df_gw_heads_filled.to_csv(file, sep=";")

print("\n" + "="*70)
print("Processing complete!")
print(f"Total stations processed: {len(df_gw_heads.columns)}")
print(f"Output saved to: {file}")
print("="*70)