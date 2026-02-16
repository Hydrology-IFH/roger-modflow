from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import gstools as gs
from scipy.optimize import OptimizeWarning
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import h5netcdf
import datetime
import warnings
import h5py

# Set random seed for reproducibility
np.random.seed(42)


def plot_measurement_points(x, y, heads, fig_name='measurement_points'):
    """Plot measurement point locations and values"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    scatter = ax.scatter(x, y, c=heads, s=100, cmap='viridis', 
                         edgecolors='black', linewidths=0.5)
    fig.colorbar(scatter, ax=ax, label='Groundwater Head (m)')
    ax.set_xlabel('X-coordinate', fontsize=12)
    ax.set_ylabel('Y-coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    fig.tight_layout()
    file = base_path / 'figures' / 'spatio_temporal_universal_kriging' / f'{fig_name}.png'
    fig.savefig(file, dpi=250, bbox_inches='tight')
    plt.close(fig)


def idw_interpolation(x, y, heads, grid_resolution=50, grid_extent=(0, 1000, 0, 1000), power=2, 
                     smoothing=0, max_neighbors=None):
    """
    Perform Inverse Distance Weighting (IDW) interpolation
    
    Parameters:
    -----------
    x, y : arrays
        Coordinates of measurement points
    heads : array
        Groundwater head values
    grid_resolution : int
        Grid spacing in meters
    power : float
        Power parameter for IDW (typically 1-3)
    smoothing : float
        Smoothing parameter to avoid division by zero
    max_neighbors : int, optional
        Maximum number of neighbors to use
    
    Returns:
    --------
    grid_x, grid_y : arrays
        Grid coordinates
    interpolated_heads : array
        Interpolated head values
    interpolation_variance : array
        Variance estimate
    """
    # Create interpolation grid
    grid_x = np.linspace(grid_extent[0], grid_extent[1], 
                        int((grid_extent[1] - grid_extent[0]) / grid_resolution) + 1)
    grid_y = np.linspace(grid_extent[2], grid_extent[3], 
                        int((grid_extent[3] - grid_extent[2]) / grid_resolution) + 1)
    X, Y = np.meshgrid(grid_x, grid_y)
    
    # Flatten grid for computation
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(np.column_stack([x, y]))
    
    # Initialize output arrays
    interpolated = np.zeros(len(grid_points))
    variance = np.zeros(len(grid_points))
    
    # Determine number of neighbors
    n_neighbors = len(x) if max_neighbors is None else min(max_neighbors, len(x))
    
    # Interpolate for each grid point
    for i, point in enumerate(grid_points):
        distances, indices = tree.query(point, k=n_neighbors)
        distances = distances + smoothing
        weights = 1.0 / (distances ** power)
        weights = weights / np.sum(weights)
        interpolated[i] = np.sum(weights * heads[indices])
        local_mean = np.sum(weights * heads[indices])
        variance[i] = np.sum(weights * (heads[indices] - local_mean) ** 2)
    
    # Reshape back to grid
    interpolated_heads = interpolated.reshape(X.shape)
    interpolation_variance = variance.reshape(X.shape)
    
    return grid_x, grid_y, interpolated_heads, interpolation_variance


def fit_variogram_model_from_timeseries(timeseries_dict, well_coords_dict_porous, 
                                       max_dist=None, n_sample_times=10, fig_name='variogram_timeseries_fit'):
    """
    Fit variogram model using data from multiple time steps
    This provides more robust parameter estimation
    
    Parameters:
    -----------
    timeseries_dict : dict
        Dictionary with timestamps as keys, head values as values
    well_coords_dict_porous : dict
        Dictionary with well IDs as keys, (x, y) coordinates as values
    max_dist : float, optional
        Maximum distance for variogram calculation
    n_sample_times : int
        Number of time steps to sample for fitting
    
    Returns:
    --------
    best_model : GSTools CovModel
        Fitted variogram model
    best_model_name : str
        Name of best model type
    """
    print("\nFitting variogram model from time series data...")
    
    # Sample time steps uniformly across the series
    timestamps = sorted(timeseries_dict.keys())
    n_times = len(timestamps)
    sample_indices = np.linspace(0, n_times-1, min(n_sample_times, n_times), dtype=int)
    sample_times = [timestamps[i] for i in sample_indices]
    
    # Collect all experimental variogram data
    all_bin_centers = []
    all_gammas = []
    all_heads = []
    
    for timestamp in sample_times:
        heads_dict = timeseries_dict[timestamp]
        well_ids = list(heads_dict.keys())
        
        # Get coordinates and heads
        x = np.array([well_coords_dict_porous[wid][0] for wid in well_ids])
        y = np.array([well_coords_dict_porous[wid][1] for wid in well_ids])
        heads = np.array([heads_dict[wid] for wid in well_ids])
        
        # Calculate max distance if not provided
        if max_dist is None:
            distances = np.sqrt((x[:, None] - x[None, :]) ** 2 + 
                              (y[:, None] - y[None, :]) ** 2)
            max_dist = np.percentile(distances[distances > 0], 50)

        # adaptive binning
        # distances = distances[distances > 0]
        # distances = np.sort(distances)
        # max_st_dist = np.percentile(distances, 75)  # Use 75th percentile

        # n_pairs = len(distances)
        # max_dist = np.percentile(distances, 75)  # Use 75th percentile

        # # Calculate how many bins we can support
        # n_bins_possible = n_pairs // 20
        # n_bins = min(n_bins_possible, 30)
        # n_bins = max(n_bins, 10)  # At least 10 bins

        # # Create bins with approximately equal number of pairs
        # percentiles = np.linspace(0, 100, n_bins + 1)
        # bin_edges = np.percentile(distances[distances <= max_dist], percentiles)
        # bin_edges_avg = np.unique(bin_edges) 
        
        # Calculate experimental variogram
        bin_edges = np.linspace(0, max_dist, 30)
        bin_center, gamma = gs.vario_estimate((x, y), heads, bin_edges=bin_edges)
        
        # Remove invalid values
        valid_mask = np.isfinite(gamma) & np.isfinite(bin_center) & (bin_center > 0)
        all_bin_centers.append(bin_center[valid_mask])
        all_gammas.append(gamma[valid_mask])
        all_heads.extend(heads)
    
    # Combine all experimental variogram data
    combined_bin_centers = np.concatenate(all_bin_centers)
    combined_gammas = np.concatenate(all_gammas)
    all_heads = np.array(all_heads)
    
    # Sort by distance
    sort_idx = np.argsort(combined_bin_centers)
    combined_bin_centers = combined_bin_centers[sort_idx]
    combined_gammas = combined_gammas[sort_idx]


    # Bin averaging to reduce noise
    n_bins = 30
    bin_edges_avg = np.linspace(combined_bin_centers.min(), 
                                combined_bin_centers.max(), n_bins + 1)
    binned_centers = []
    binned_gammas = []
    
    for i in range(n_bins):
        mask = (combined_bin_centers >= bin_edges_avg[i]) & \
               (combined_bin_centers < bin_edges_avg[i+1])
        if mask.sum() > 0:
            binned_centers.append(combined_bin_centers[mask].mean())
            binned_gammas.append(combined_gammas[mask].mean())
    
    bin_center = np.array(binned_centers)
    gamma = np.array(binned_gammas)
    
    # Try multiple model types
    models_to_try = [
        ('Gaussian', gs.Gaussian),
        ('Exponential', gs.Exponential),
        ('Spherical', gs.Spherical),
    ]
    
    best_model = None
    best_score = np.inf
    best_model_name = None
    
    for model_name, ModelClass in models_to_try:
        try:
            model = ModelClass(dim=2)
            
            # Initial parameter estimates
            var_init = np.var(all_heads) * 0.8
            len_init = max_dist / 3
            nugget_init = np.var(all_heads) * 0.1
            
            model.var = var_init
            model.len_scale = len_init
            model.nugget = nugget_init
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                
                try:
                    fit_result = model.fit_variogram(
                        bin_center, 
                        gamma, 
                        nugget=True,
                        max_eval=10000,
                        weights='inv',
                    )
                    
                    predicted = model.variogram(bin_center)
                    rmse = np.sqrt(np.mean((gamma - predicted) ** 2))
                    
                    if rmse < best_score and model.var > 0 and model.len_scale > 0:
                        best_score = rmse
                        best_model = model
                        best_model_name = model_name
                        
                except (RuntimeError, ValueError) as e:
                    print(f"   {model_name} fitting failed: {str(e)[:50]}...")
                    continue
                    
        except Exception as e:
            print(f"   Could not fit {model_name}: {str(e)[:50]}...")
            continue
    
    # Fallback if no model fitted successfully
    if best_model is None:
        print("   Warning: Automatic fitting failed. Using manual parameters.")
        best_model = gs.Gaussian(dim=2)
        best_model.var = np.var(all_heads) * 0.7
        best_model.len_scale = max_dist / 3
        best_model.nugget = np.var(all_heads) * 0.1
        best_model_name = "Gaussian (manual)"
        best_score = np.inf
    
    print(f"\nBest Variogram Model: {best_model_name}")
    print(f"  Variance (sill): {best_model.var:.2f}")
    print(f"  Correlation length: {best_model.len_scale:.2f}")
    print(f"  Nugget: {best_model.nugget:.2f}")
    if best_score < np.inf:
        print(f"  RMSE (variogram): {best_score:.4f}")
    
    # Plot combined variogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(bin_center, gamma, label='Experimental variogram', 
               color='red', s=50, alpha=0.7, zorder=3)
    
    x_plot = np.linspace(0, max(bin_center), 100)
    ax.plot(x_plot, best_model.variogram(x_plot), 
            label=f'Fitted model ({best_model_name})', linewidth=2.5, zorder=2)
    
    ax.axhline(y=best_model.var + best_model.nugget, 
               color='gray', linestyle='--', alpha=0.5, 
               label='Sill', zorder=1)
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Semivariance', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    
    file = base_path / 'figures' / 'spatio_temporal_universal_kriging' / f'{fig_name}.png'
    fig.savefig(file, dpi=250, bbox_inches='tight')
    plt.close(fig)
    
    return best_model, best_model_name, best_score


def create_idw_drift_function(x, y, heads, grid_x, grid_y, 
                              idw_power=2, smoothing=1.0, max_neighbors=10):
    """
    Create a drift function based on IDW interpolation
    
    Parameters:
    -----------
    x, y : arrays
        Well coordinates
    heads : array
        Groundwater head values
    grid_x, grid_y : arrays
        Grid for drift calculation
    idw_power : float
        IDW power parameter
    smoothing : float
        Smoothing parameter
    max_neighbors : int
        Maximum neighbors for IDW
    
    Returns:
    --------
    drift_function : callable
        Function that can be used as external drift in kriging
    """
    # Compute IDW interpolation as trend
    _, _, idw_trend, _ = idw_interpolation(
        x, y, heads, 
        grid_resolution=grid_x[1] - grid_x[0],
        grid_extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
        power=idw_power,
        smoothing=smoothing,
        max_neighbors=max_neighbors
    )
    
    # Create interpolator for the IDW trend
    idw_interpolator = RegularGridInterpolator(
        (grid_x, grid_y), 
        idw_trend.T,
        method='linear',
        bounds_error=False,
        fill_value=None
    )
    
    # Define drift function
    def drift_idw(x_pts, y_pts):
        """External drift based on IDW"""
        points = np.column_stack([x_pts.ravel(), y_pts.ravel()])
        return idw_interpolator(points).reshape(x_pts.shape)
    
    return drift_idw


def universal_kriging_timeseries(x, y, heads, model, drift_function,
                                 grid_x, grid_y, topography=None):
    """
    Perform Universal Kriging with pre-computed model and drift
    
    Parameters:
    -----------
    x, y : arrays
        Well coordinates
    heads : array
        Groundwater head values
    model : GSTools CovModel
        Pre-fitted variogram model (used as template)
    drift_function : callable
        Pre-computed drift function
    grid_x, grid_y : arrays
        Interpolation grid
    topography : array, optional
        Topography surface for constraining heads
    
    Returns:
    --------
    interpolated_heads : array
        Interpolated groundwater surface
    kriging_variance : array
        Kriging variance
    """
    # Perform Universal Kriging with IDW drift
    krige = gs.krige.Universal(
        model=model,
        cond_pos=[x, y],
        cond_val=heads,
        drift_functions=[drift_function]
    )
    
    # Interpolate on grid
    interpolated_heads, kriging_variance = krige.structured([grid_x, grid_y])
    interpolated_heads = interpolated_heads.T
    kriging_variance = kriging_variance.T
    
    # Constrain heads to topography if provided
    if topography is not None:
        interpolated_heads = np.where(interpolated_heads > topography, 
                                     topography, interpolated_heads)
    
    return interpolated_heads, kriging_variance


def plot_interpolation_results(grid_x, grid_y, interpolated_heads,
                               kriging_variance, topography, x, y, heads, 
                               depths, mask, method='Universal', fig_name='kriging_results'):
    """Plot interpolation results and kriging variance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Get valid extent
    rows, cols = np.where(mask)
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    X, Y = np.meshgrid(grid_x[x1:x2+1], grid_y[y1:y2+1])

    # Mask invalid regions
    interpolated_heads[~mask] = np.nan
    kriging_variance[~mask] = np.nan
    interpolated_heads = interpolated_heads[y1:y2+1, x1:x2+1]
    kriging_variance = kriging_variance[y1:y2+1, x1:x2+1]
    topography = topography[y1:y2+1, x1:x2+1]
    
    # Plot interpolated heads
    contour = ax1.contourf(X, Y, interpolated_heads, levels=20, 
                          cmap='viridis', alpha=0.8)
    ax1.scatter(x, y, c=heads, s=80, cmap='viridis', 
               edgecolors='white', linewidths=1.5, label='Measurements')
    contour_lines = ax1.contour(X, Y, interpolated_heads, levels=10, 
                                colors='black', alpha=0.4, linewidths=0.5)
    ax1.clabel(contour_lines, inline=True, fontsize=8)
    
    cbar1 = plt.colorbar(contour, ax=ax1, label='Groundwater Head (m)')
    ax1.set_xlabel('X-coordinate', fontsize=12)
    ax1.set_ylabel('Y-coordinate', fontsize=12)
    ax1.set_title(f'{method} Kriging - Interpolated Heads', fontsize=14)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot kriging variance
    variance_plot = ax2.contourf(X, Y, kriging_variance, levels=20, 
                                cmap='Reds', alpha=0.8)
    ax2.scatter(x, y, c='blue', s=80, marker='x', 
               linewidths=2, label='Measurements')
    
    cbar2 = plt.colorbar(variance_plot, ax=ax2, label='Variance (m²)')
    ax2.set_xlabel('X-coordinate', fontsize=12)
    ax2.set_ylabel('Y-coordinate', fontsize=12)
    ax2.set_title('Kriging Uncertainty', fontsize=14)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    file = base_path / 'figures' / 'spatio_temporal_universal_kriging' / f'{method.lower()}_{fig_name}.png'
    fig.savefig(file, dpi=250, bbox_inches='tight')
    plt.close(fig)
    
    # Plot depths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    depths_array = np.where(topography - interpolated_heads > 25, 25, 
                           topography - interpolated_heads)
    
    contour = ax1.contourf(X, Y, depths_array, levels=10, 
                          cmap='viridis', alpha=0.8, vmin=0, vmax=25)
    ax1.scatter(x, y, c=depths, s=80, cmap='viridis', vmin=0, vmax=25,  
               edgecolors='white', linewidths=1.5)
    contour_lines = ax1.contour(X, Y, depths_array, levels=10, 
                                colors='black', alpha=0.4, linewidths=0.5)
    ax1.clabel(contour_lines, inline=True, fontsize=8)
    
    cbar1 = plt.colorbar(contour, ax=ax1, label='Groundwater depth (m)')
    ax1.set_xlabel('X-coordinate', fontsize=12)
    ax1.set_ylabel('Y-coordinate', fontsize=12)
    ax1.set_title('Interpolated Depths', fontsize=14)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    variance_plot = ax2.contourf(X, Y, kriging_variance, levels=20, 
                                cmap='Reds', alpha=0.8)
    ax2.scatter(x, y, c='blue', s=80, marker='x', linewidths=2)
    
    cbar2 = plt.colorbar(variance_plot, ax=ax2, label='Variance (m²)')
    ax2.set_xlabel('X-coordinate', fontsize=12)
    ax2.set_ylabel('Y-coordinate', fontsize=12)
    ax2.set_title('Uncertainty', fontsize=14)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    file = base_path / 'figures' / 'spatio_temporal_universal_kriging' / f'{method.lower()}_{fig_name}_depth.png'
    fig.savefig(file, dpi=250, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

base_path = Path(__file__).parent

# observations wells in the prous aquifer close to the fissured aquifer
observation_well_ids_porous = ['0160_070-1', '0113_071-9', '2317_071-5', '2064_120-9', '0101_120-2', '2027_120-2', '0107_119-3', '0104_071-8', '2047_120-2', '0109_119-2', 'PE 01', 'PE 39', 'PE 41']
ids_to_remove = ['0190_069-0', '0118_070-0', '0122_069-1', '0831_018-1', '0131_069-2', '2311_120-2']  # wells with inconsistent data

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

# Load observation wells
file = base_path / "observations" / "groundwater_observation_wells.gpkg"
groundwater_observation_wells = gpd.read_file(file)
groundwater_observation_wells['station_id'] = groundwater_observation_wells['station_id'].str.replace('/', '_')
# remove wells outside grid_extent
groundwater_observation_wells = groundwater_observation_wells.cx[
    grid_extent[0]:grid_extent[1], grid_extent[2]:grid_extent[3]
]
# remove wells with inconsistent data
groundwater_observation_wells = groundwater_observation_wells[
    ~groundwater_observation_wells['station_id'].isin(ids_to_remove)
]

# Load groundwater head time series
file = base_path / "observations" / "groundwater_head_time_series_filled.csv"
_df_gw_heads = pd.read_csv(file, sep=";", index_col=0)
_df_gw_heads.index = pd.to_datetime(_df_gw_heads.index, format="%Y-%m-%d")
date_time = pd.date_range(start="1990-01-01", end="2023-12-31", freq="D")
df_gw_heads = pd.DataFrame(index=date_time)
df_gw_heads = df_gw_heads.join(_df_gw_heads, how="left")
# resample to monthly means
df_gw_heads = df_gw_heads.resample('ME').mean()

file = base_path / "observations" / "groundwater_head_time_series_monthly_filled.csv"
df_gw_heads.to_csv(file, sep=";")

# load interpolated groundwater heads
src = rasterio.open(str(base_path / "input" / "groundwater_heads_interpolated_50m.tif"))
gw_heads_interpolated = src.read(1)
gw_heads_interpolated_fissured = np.where(mask_fissured, gw_heads_interpolated, np.nan)
# get extent of finite values
rows, cols = np.where(np.isfinite(gw_heads_interpolated_fissured))
min_row, max_row = rows.min(), rows.max()
min_col, max_col = cols.min(), cols.max()

x_ = np.random.uniform(xcoords[min_col], xcoords[max_col], 500)
y_ = np.random.uniform(ycoords[min_row], ycoords[max_row], 500)
# remove points outside mask
x = []
y = []
for i in range(len(x_)):
    row, col = src.index(x_[i], y_[i])
    if mask_fissured[row, col]:
        x.append(x_[i])
        y.append(y_[i])

x_fissured = np.array(x)
y_fissured = np.array(y)
print(len(x_fissured), "measurement points in fissured area")

_columns = [f"f_{i+1}" for i in range(len(x_fissured))]
df_gw_heads_pseudowells = pd.DataFrame(index=df_gw_heads.index, columns=_columns)

# sample pseudowells of the fissured aquifer
for month_i, timestamp in enumerate(df_gw_heads.index):
    year_month = timestamp.strftime("%Y-%m")
    df_gw_heads_initial = df_gw_heads.loc[timestamp, :].to_frame()
    df_gw_heads_initial.columns = ['groundwater_head']

    path = base_path / "observations" / "groundwater_observation_wells.gpkg"
    _gdf_gw_heads_initial = gpd.read_file(path)
    # remove wells outside grid_extent
    gdf_gw_heads_initial = _gdf_gw_heads_initial.cx[grid_extent[0]:grid_extent[1], grid_extent[2]:grid_extent[3]]
    gdf_gw_heads_initial["station_id"] = gdf_gw_heads_initial["station_id"].str.replace('/', '_')
    # remove wells with inconsistent data
    gdf_gw_heads_initial = gdf_gw_heads_initial[
        ~gdf_gw_heads_initial['station_id'].isin(ids_to_remove)
    ]
    gdf_gw_heads_initial.index = gdf_gw_heads_initial["station_id"]
    # merge with groundwater heads
    gdf_gw_heads_initial = gdf_gw_heads_initial.join(df_gw_heads_initial, how='left')
    _gdf_gw_heads_initial = gdf_gw_heads_initial[gdf_gw_heads_initial['station_id'].isin(observation_well_ids_porous)]

    # load interpolated groundwater heads
    src = rasterio.open(str(base_path / "input" / "groundwater_heads_interpolated_50m.tif"))
    gw_heads_interpolated = src.read(1)

    df_diff = pd.DataFrame(index=_gdf_gw_heads_initial.index)
    df_diff['difference'] = np.nan
    df_diff['observed_head'] = _gdf_gw_heads_initial['groundwater_head']
    df_diff['interpolated_head'] = np.nan
    for idx, row in _gdf_gw_heads_initial.iterrows():
        x = row.geometry.x
        y = row.geometry.y
        row, col = src.index(x, y)
        head_interpolated = gw_heads_interpolated[row, col]
        head_observed = _gdf_gw_heads_initial.loc[idx, 'groundwater_head']
        diff = head_observed - head_interpolated
        df_diff.loc[idx, 'difference'] = diff
        df_diff.loc[idx, 'interpolated_head'] = head_interpolated
    df_diff.loc['avg', 'difference'] = df_diff['difference'].mean()
    df_diff.loc['std', 'difference'] = df_diff['difference'].std()
    # save to csv
    file = base_path / 'figures' / 'spatio_temporal_universal_kriging' / f'interpolation_difference_observed_interpolated_heads_{year_month}.csv'
    df_diff.to_csv(file, sep=";", index=True)

    heads_fissured = np.zeros_like(x_fissured)
    for i in range(len(x_fissured)):
        row, col = src.index(x_fissured[i], y_fissured[i])
        head = gw_heads_interpolated_fissured[row, col] + (df_diff['difference'].mean() * 50)
        if head > topography[row, col]:
            head = topography[row, col] - 1
        heads_fissured[i] = head
    depths_fissured = np.zeros_like(x_fissured)
    for i in range(len(x_fissured)):
        row, col = src.index(x_fissured[i], y_fissured[i])
        depths_fissured[i] = topography[row, col] - heads_fissured[i]

    # add to dataframe
    for i in range(len(x_fissured)):
        df_gw_heads_pseudowells.loc[timestamp, :f"f_{len(heads_fissured)}"] = heads_fissured

file = base_path / 'figures' / 'spatio_temporal_universal_kriging' / 'monthly_pseudowells.csv'
df_gw_heads_pseudowells.to_csv(file, sep=";", index=True)

well_coords_dict_fissured = {}
for i, station_id in enumerate(df_gw_heads_pseudowells.columns):
    well_coords_dict_fissured[station_id] = (x_fissured[i], y_fissured[i])

# Create time series dictionary (only for porous aquifer wells initially)
timeseries_dict_fissured = {}
for timestamp in df_gw_heads_pseudowells.index:
    heads_dict = {}
    for col in df_gw_heads_pseudowells.columns:
        value = df_gw_heads_pseudowells.loc[timestamp, col]
        if pd.notna(value):
            heads_dict[col] = value
    if len(heads_dict) > 3:  # Need minimum number of points
        timeseries_dict_fissured[timestamp] = heads_dict

# ============================================================================
# PREPARE TIME SERIES DATA FOR VARIOGRAM FITTING
# ============================================================================

print("=" * 70)
print("Preparing Time Series Data for Variogram Fitting")
print("=" * 70)

# Create well coordinates dictionary
well_coords_dict_porous = {}
for idx, row in groundwater_observation_wells.iterrows():
    station_id = row['station_id'].replace('/', '_')
    if station_id in df_gw_heads.columns:
        well_coords_dict_porous[station_id] = (row.geometry.x, row.geometry.y)

# Create time series dictionary (only for porous aquifer wells initially)
timeseries_dict_porous = {}
for timestamp in df_gw_heads.index:
    heads_dict = {}
    for col in well_coords_dict_porous.keys():
        value = df_gw_heads.loc[timestamp, col]
        if pd.notna(value):
            heads_dict[col] = value
    if len(heads_dict) > 3:  # Need minimum number of points
        timeseries_dict_porous[timestamp] = heads_dict

well_coords_dict_combined = {}
for i, station_id in enumerate(df_gw_heads_pseudowells.columns):
    well_coords_dict_combined[station_id] = (x_fissured[i], y_fissured[i])
for idx, row in groundwater_observation_wells.iterrows():
    station_id = row['station_id'].replace('/', '_')
    well_coords_dict_combined[station_id] = (row.geometry.x, row.geometry.y)

# merge dicionaries coords and timeseries dictionaries
timeseries_dict_combined = {}
for timestamp in df_gw_heads.index:
    heads_dict_combined = {}
    for col in well_coords_dict_porous.keys():
        value = df_gw_heads.loc[timestamp, col]
        if pd.notna(value):
            heads_dict_combined[col] = value
    for col in df_gw_heads_pseudowells.columns:
        value = df_gw_heads_pseudowells.loc[timestamp, col]
        if pd.notna(value):
            heads_dict_combined[col] = value
    timeseries_dict_combined[timestamp] = heads_dict_combined

# ============================================================================
# FIT VARIOGRAM MODEL ONCE FROM TIME SERIES
# ============================================================================

print("\n" + "=" * 70)
print("Fitting Variogram Model from Time Series (POROUS AQUIFER)")
print("=" * 70)

model_porous, model_name_porous, rmse_model_porous = fit_variogram_model_from_timeseries(
    timeseries_dict_porous,
    well_coords_dict_porous,
    n_sample_times=len(df_gw_heads.index),
    fig_name='variogram_timeseries_fit_porous'
)

# model_fissured, model_name_fissured, rmse_model_fissured = fit_variogram_model_from_timeseries(
#     timeseries_dict_fissured,
#     well_coords_dict_fissured,
#     n_sample_times=len(df_gw_heads_pseudowells.index),
#     fig_name='variogram_timeseries_fit_fissured'
# )

# model_combined, model_name_combined, rmse_model_combined = fit_variogram_model_from_timeseries(
#     timeseries_dict_combined,
#     well_coords_dict_combined,
#     n_sample_times=len(df_gw_heads.index),
#     fig_name='variogram_timeseries_fit_combined'
# )

# ============================================================================
# CREATE NETCDF OUTPUT FILE
# ============================================================================

ds = xr.Dataset()
ds["spatial_ref"] = spatial_ref
file = base_path / "observations" / "monthly_spatio_temporal_universal_kriging.nc"
ds.to_netcdf(file, engine="h5netcdf")
ds.close()

file = base_path / "observations" / "monthly_spatio_temporal_universal_kriging.nc"
with h5netcdf.File(file, "a", decode_vlen_strings=False) as f:
    f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="Interpolated groundwater heads using space-time universal kriging with IDW drift in the Dreisam-Möhlin-Neumagen catchment",
        institution="University of Freiburg, Chair of Hydrology",
        spatial_ref="EPSG:25832",
        x_origin=396331.5,
        y_origin=5325918.5,
    )
    dict_dim = {
        "y": len(ds_params['y'].values), 
        "x": len(ds_params['x'].values), 
        "Time": len(df_gw_heads.index),
        'scalar': 1
    }
    f.dimensions = dict_dim
    
    v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
    v.attrs["long_name"] = "X-coordinate"
    v.attrs["units"] = "m"
    v[:] = xcoords
    
    v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
    v.attrs["long_name"] = "Y-coordinate"
    v.attrs["units"] = "m"
    v[:] = ycoords
    
    v = f.create_variable('cell_width', ('scalar',), float)
    v.attrs['long_name'] = 'Cell width'
    v.attrs['units'] = 'm'
    v[:] = 50.0
    
    v = f.create_variable("Time", ("Time",), float, compression="gzip", compression_opts=1)
    v.attrs["long_name"] = "Time"
    v.attrs["units"] = "months since 2013-01-01"
    v[:] = np.arange(len(df_gw_heads.index))

    # Create variables
    v = f.create_variable(
        "interpolated_gw_heads_porous", ("Time", "y", "x"), np.float64, 
        compression="gzip", compression_opts=1
    )
    v.attrs.update(long_name="Interpolated groundwater heads (porous)", 
                   units="m a.s.l.", grid_mapping="spatial_ref")

    # v = f.create_variable(
    #     "interpolated_gw_heads_fissured", ("Time", "y", "x"), np.float64, 
    #     compression="gzip", compression_opts=1
    # )
    # v.attrs.update(long_name="Interpolated groundwater heads (fissured)", 
    #                units="m a.s.l.", grid_mapping="spatial_ref")

    # v = f.create_variable(
    #     "interpolated_gw_heads", ("Time", "y", "x"), np.float64, 
    #     compression="gzip", compression_opts=1
    # )
    # v.attrs.update(long_name="Interpolated groundwater heads (combined)", 
    #                units="m a.s.l.", grid_mapping="spatial_ref")

    v = f.create_variable(
        "uncertainty_porous", ("Time", "y", "x"), np.float64, 
        compression="gzip", compression_opts=1
    )
    v.attrs.update(long_name="Kriging variance (porous)", 
                   units="m2", grid_mapping="spatial_ref")

    # v = f.create_variable(
    #     "uncertainty_fissured", ("Time", "y", "x"), np.float64, 
    #     compression="gzip", compression_opts=1
    # )
    # v.attrs.update(long_name="Kriging variance (fissured)", 
    #                units="m2", grid_mapping="spatial_ref")

    # v = f.create_variable(
    #     "uncertainty_combined", ("Time", "y", "x"), np.float64, 
    #     compression="gzip", compression_opts=1
    # )
    # v.attrs.update(long_name="Kriging variance (combined)", 
    #                units="m2", grid_mapping="spatial_ref")

    vlen_str_dtype = h5py.special_dtype(vlen=str)
    v = f.create_variable(
        "model_type_porous", ("scalar",), dtype=vlen_str_dtype
    )
    v.attrs.update(long_name="Variogram model type of porous aquifer")
    v[0] = model_name_porous

    # v = f.create_variable(
    #     "model_type_fissured", ("scalar",), dtype=vlen_str_dtype
    # )
    # v.attrs.update(long_name="Variogram model type of fissured aquifer")
    # v[0] = model_name_fissured

    # v = f.create_variable(
    #     "model_type_combined", ("scalar",), dtype=vlen_str_dtype
    # )
    # v.attrs.update(long_name="Variogram model type of combined aquifer")
    # v[0] = model_name_combined

    v = f.create_variable(
        "rmse_porous", ("Time",), float
    )
    v.attrs.update(long_name="Root mean square error of the universal kriging model of porous aquifer")

    # v = f.create_variable(
    #     "rmse_fissured", ("scalar",), float
    # )
    # v.attrs.update(long_name="Root mean square error of the universal kriging model of fissured aquifer")
    # v[0] = rmse_model_fissured
    # v = f.create_variable(
    #     "rmse_combined", ("scalar",), float
    # )
    # v.attrs.update(long_name="Root mean square error of the universal kriging model of combined aquifer")
    # v[0] = rmse_model_combined

# ============================================================================
# LOOP OVER TIME STEPS AND PERFORM KRIGING
# ============================================================================

print("\n" + "=" * 70)
print("Processing Time Series with Pre-fitted Model")
print("=" * 70)

# Load reference interpolation
src = rasterio.open(str(base_path / "input" / "groundwater_heads_interpolated_50m.tif"))
gw_heads_interpolated = src.read(1)

# Define grid
grid_x = xcoords
grid_y = ycoords

for month_i, timestamp in enumerate(df_gw_heads.index):
    year_month = timestamp.strftime("%Y-%m")
    print(f"\nProcessing {year_month} ({month_i+1}/{len(df_gw_heads)})")
    
    # Get current time step data
    df_gw_heads_current = df_gw_heads.loc[timestamp, :].to_frame()
    df_gw_heads_current.columns = ['groundwater_head']
    
    # Merge with well locations
    gdf_gw_heads = groundwater_observation_wells.copy()
    gdf_gw_heads["station_id"] = gdf_gw_heads["station_id"].str.replace('/', '_')
    gdf_gw_heads.index = gdf_gw_heads["station_id"]
    gdf_gw_heads = gdf_gw_heads.join(df_gw_heads_current, how='left')
    gdf_gw_heads = gdf_gw_heads.dropna(subset=['groundwater_head'])
    
    # Extract coordinates and heads for porous aquifer
    x_porous = gdf_gw_heads.geometry.x.values
    y_porous = gdf_gw_heads.geometry.y.values
    heads_porous = gdf_gw_heads['groundwater_head'].values
    depths_porous = np.zeros_like(x_porous)
    
    for i in range(len(x_porous)):
        row, col = src.index(x_porous[i], y_porous[i])
        depths_porous[i] = topography[row, col] - heads_porous[i]
    
    # Create IDW drift function for this time step
    drift_func_porous = create_idw_drift_function(
        x_porous, y_porous, heads_porous,
        grid_x, grid_y,
        idw_power=2,
        smoothing=1.0,
        max_neighbors=10
    )
    
    # Perform universal kriging with pre-fitted model and current drift
    interpolated_heads_porous, variance_porous = universal_kriging_timeseries(
        x_porous, y_porous, heads_porous,
        model_porous,
        drift_func_porous,
        grid_x, grid_y,
        topography=topography
    )
    # get kriged heads at observation points
    heads_porous_kriged = np.zeros_like(x_porous)
    for i in range(len(x_porous)):
        row, col = src.index(x_porous[i], y_porous[i])
        head = interpolated_heads_porous[row, col]
        heads_porous_kriged[i] = head

    interpolated_heads_porous = np.where(interpolated_heads_porous > topography, topography, interpolated_heads_porous)
    interpolated_heads_porous = np.where(mask_porous, interpolated_heads_porous, np.nan)
    variance_porous = np.where(mask_porous, variance_porous, np.nan)
    
    # Plot results (only for first, end of year, and last time steps)
    if month_i in [0, len(df_gw_heads)//12, len(df_gw_heads)-1]:
        plot_interpolation_results(
            grid_x, grid_y, interpolated_heads_porous, variance_porous,
            topography, x_porous, y_porous, heads_porous, depths_porous,
            mask_porous, method='Universal', 
            fig_name=f'porous_{year_month}'
        )

    # calculate RMSE
    rmse_porous = np.sqrt(np.mean((heads_porous - heads_porous_kriged) ** 2))
    print(f'RMSE (empirical): {rmse_porous:.2f}')

    # # Get current time step data
    # df_gw_heads_current = df_gw_heads_pseudowells.loc[timestamp, :].to_frame()
    # df_gw_heads_current.columns = ['groundwater_head']
    
    # heads_fissured = df_gw_heads_current['groundwater_head'].values
    # depths_fissured = np.zeros_like(x_fissured)
    
    # for i in range(len(x_fissured)):
    #     row, col = src.index(x_fissured[i], y_fissured[i])
    #     depths_fissured[i] = topography[row, col] - heads_fissured[i]
    
    # # Create IDW drift function for this time step
    # drift_func_fissured = create_idw_drift_function(
    #     x_fissured, y_fissured, heads_fissured,
    #     grid_x, grid_y,
    #     idw_power=2,
    #     smoothing=1.0,
    #     max_neighbors=10
    # )
    
    # # Perform universal kriging with pre-fitted model and current drift
    # interpolated_heads_fissured, variance_fissured = universal_kriging_timeseries(
    #     x_fissured, y_fissured, heads_fissured,
    #     model_fissured,
    #     drift_func_fissured,
    #     grid_x, grid_y,
    #     topography=topography
    # )
    
    # # Plot results (only for first, end of year, and last time steps)
    # if month_i in [0, len(df_gw_heads_pseudowells)//12, len(df_gw_heads_pseudowells)-1]:
    #     plot_interpolation_results(
    #         grid_x, grid_y, interpolated_heads_fissured, variance_fissured,
    #         topography, x_fissured, y_fissured, heads_fissured, depths_fissured,
    #         mask_fissured, method='Universal', 
    #         fig_name=f'fissured_{year_month}'
    #     )

    # # Get current time step data
    # df_gw_heads_current = pd.concat([df_gw_heads.loc[timestamp, :].to_frame(), df_gw_heads_pseudowells.loc[timestamp, :].to_frame()])
    # df_gw_heads_current.columns = ['groundwater_head']
    
    # # Merge with well locations
    # gdf_gw_heads_combined = groundwater_observation_wells.copy()
    # gdf_gw_heads_combined["station_id"] = gdf_gw_heads_combined["station_id"].str.replace('/', '_')
    # gdf_gw_heads_combined["xcoords"] = gdf_gw_heads_combined.geometry.x
    # gdf_gw_heads_combined["ycoords"] = gdf_gw_heads_combined.geometry.y
    # for i, station_id in enumerate(df_gw_heads_pseudowells.columns):
    #     x_coord, y_coord = well_coords_dict_fissured[station_id]
    #     gdf_gw_heads_combined = gdf_gw_heads_combined._append({
    #         "station_id": station_id,
    #         "geometry": Point(x_coord, y_coord),
    #         "xcoords": x_coord,
    #         "ycoords": y_coord
    #     }, ignore_index=True)

    # gdf_gw_heads_combined.index = gdf_gw_heads_combined["station_id"]
    # gdf_gw_heads_combined = gdf_gw_heads_combined[["geometry", "xcoords", "ycoords"]]
    # gdf_gw_heads_combined = gdf_gw_heads_combined.join(df_gw_heads_current, how='left')
    # gdf_gw_heads_combined = gdf_gw_heads_combined.dropna(subset=['groundwater_head'])
    
    # # Extract coordinates and heads for porous aquifer
    # x_combined = gdf_gw_heads_combined.geometry.x.values
    # y_combined = gdf_gw_heads_combined.geometry.y.values
    # heads_combined = gdf_gw_heads_combined['groundwater_head'].values
    # depths_combined = np.zeros_like(x_combined)
    
    # for i in range(len(x_combined)):
    #     row, col = src.index(x_combined[i], y_combined[i])
    #     depths_combined[i] = topography[row, col] - heads_combined[i]
    
    # # Create IDW drift function for this time step
    # drift_func_combined = create_idw_drift_function(
    #     x_combined, y_combined, heads_combined,
    #     grid_x, grid_y,
    #     idw_power=2,
    #     smoothing=1.0,
    #     max_neighbors=10
    # )
    
    # # Perform universal kriging with pre-fitted model and current drift
    # interpolated_heads_combined, variance_combined = universal_kriging_timeseries(
    #     x_combined, y_combined, heads_combined,
    #     model_combined,
    #     drift_func_combined,
    #     grid_x, grid_y,
    #     topography=topography
    # )
    
    # # Plot results (only for first, end of year, and last time steps)
    # if month_i in [0, len(df_gw_heads)//12, len(df_gw_heads)-1]:
    #     plot_interpolation_results(
    #         grid_x, grid_y, interpolated_heads_combined, variance_combined,
    #         topography, x_combined, y_combined, heads_combined, depths_combined,
    #         mask_combined, method='Universal', 
    #         fig_name=f'combined_{year_month}'
    #     )

    
    # Save to NetCDF
    file = base_path / "observations" / "monthly_spatio_temporal_universal_kriging.nc"
    with h5netcdf.File(file, "a", decode_vlen_strings=False) as f:
        var_obj = f.variables.get("interpolated_gw_heads_porous")
        var_obj[month_i, :, :] = interpolated_heads_porous
        var_obj = f.variables.get("uncertainty_porous")
        var_obj[month_i, :, :] = variance_porous
        # var_obj = f.variables.get("interpolated_gw_heads_fissured")
        # var_obj[month_i, :, :] = interpolated_heads_fissured
        # var_obj = f.variables.get("uncertainty_fissured")
        # var_obj[month_i, :, :] = variance_fissured
        # var_obj = f.variables.get("interpolated_gw_heads")
        # var_obj[month_i, :, :] = interpolated_heads_combined
        # var_obj = f.variables.get("uncertainty_combined")
        # var_obj[month_i, :, :] = variance_combined
        var_obj = f.variables.get("rmse_porous")
        var_obj[month_i] = rmse_porous

print("\n" + "=" * 70)
print("Time Series Kriging Complete!")
print("=" * 70)