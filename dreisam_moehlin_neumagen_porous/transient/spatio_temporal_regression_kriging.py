from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from scipy.optimize import OptimizeWarning, minimize
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve
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
    file = base_path / 'figures' / 'spatio_temporal_regression_kriging' / f'{fig_name}.png'
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


def compute_spacetime_distance(x1, y1, t1, x2, y2, t2, k_temporal=1.0):
    """
    Compute space-time distance matrix
    
    Parameters:
    -----------
    x1, y1 : arrays
        Spatial coordinates of first set of points
    t1 : array
        Temporal coordinates of first set of points (in days since reference)
    x2, y2 : arrays
        Spatial coordinates of second set of points
    t2 : array
        Temporal coordinates of second set of points (in days since reference)
    k_temporal : float
        Temporal scaling parameter (controls relative importance of time vs space)
    
    Returns:
    --------
    distance : array
        Space-time distance matrix
    """
    # Ensure inputs are arrays
    x1, y1, t1 = np.atleast_1d(x1), np.atleast_1d(y1), np.atleast_1d(t1)
    x2, y2, t2 = np.atleast_1d(x2), np.atleast_1d(y2), np.atleast_1d(t2)
    
    # Reshape for broadcasting
    x1 = x1.reshape(-1, 1)
    y1 = y1.reshape(-1, 1)
    t1 = t1.reshape(-1, 1)
    x2 = x2.reshape(1, -1)
    y2 = y2.reshape(1, -1)
    t2 = t2.reshape(1, -1)
    
    # Compute spatial distance
    spatial_dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # Compute temporal distance
    temporal_dist = np.abs(t1 - t2)
    
    # Combine into space-time distance
    # d = sqrt(d_spatial^2 + k_temporal * d_temporal^2)
    spacetime_dist = np.sqrt(spatial_dist**2 + k_temporal * temporal_dist**2)
    
    return spacetime_dist


def spacetime_variogram(h, params, model_type='exponential'):
    """
    Calculate space-time variogram value
    
    Parameters:
    -----------
    h : array
        Space-time distance
    params : dict
        Variogram parameters (sill, range, nugget)
    model_type : str
        Type of variogram model
    
    Returns:
    --------
    gamma : array
        Variogram values
    """
    sill = params['sill']
    range_param = params['range']
    nugget = params['nugget']
    
    if model_type == 'exponential':
        return nugget + sill * (1 - np.exp(-h / range_param))
    elif model_type == 'spherical':
        result = np.where(h <= range_param,
                        nugget + sill * (1.5 * h / range_param - 0.5 * (h / range_param)**3),
                        nugget + sill)
        return result
    elif model_type == 'gaussian':
        return nugget + sill * (1 - np.exp(-(h / range_param)**2))
    else:
        raise ValueError(f"Unknown variogram model: {model_type}")


def spacetime_covariance(h, params, model_type='exponential'):
    """
    Calculate space-time covariance from variogram
    C(h) = sill - gamma(h) + nugget
    
    Parameters:
    -----------
    h : array
        Space-time distance
    params : dict
        Variogram parameters
    model_type : str
        Type of variogram model
    
    Returns:
    --------
    covariance : array
        Covariance values
    """
    gamma = spacetime_variogram(h, params, model_type)
    # Covariance = sill - (variogram - nugget)
    covariance = params['sill'] + params['nugget'] - gamma
    return covariance


def fit_spacetime_variogram_model(timeseries_dict, well_coords_dict, 
                                  reference_date=None, max_dist=None, 
                                  n_sample_times=20, fig_name='variogram_spacetime_fit'):
    """
    Fit space-time variogram model using data from multiple time steps
    
    Parameters:
    -----------
    timeseries_dict : dict
        Dictionary with timestamps as keys, head values as values
    well_coords_dict : dict
        Dictionary with well IDs as keys, (x, y) coordinates as values
    reference_date : datetime, optional
        Reference date for temporal scaling (first date if None)
    max_dist : float, optional
        Maximum distance for variogram calculation
    n_sample_times : int
        Number of time steps to sample for fitting
    
    Returns:
    --------
    best_params : dict
        Fitted variogram parameters including k_temporal
    best_model_name : str
        Name of best model type
    best_score : float
        RMSE of best fit
    """
    print("\nFitting space-time variogram model...")
    
    # Sample time steps uniformly across the series
    timestamps = sorted(timeseries_dict.keys())
    n_times = len(timestamps)
    sample_indices = np.linspace(0, n_times-1, min(n_sample_times, n_times), dtype=int)
    sample_times = [timestamps[i] for i in sample_indices]
    
    # Set reference date
    if reference_date is None:
        reference_date = timestamps[0]
    
    # Collect all space-time data
    all_x = []
    all_y = []
    all_t = []  # in days since reference
    all_heads = []
    n_samples = []
    
    for timestamp in sample_times:
        heads_dict = timeseries_dict[timestamp]
        well_ids = list(heads_dict.keys())
        n_samples.append(len(well_ids))
        
        # Get coordinates and heads
        x = np.array([well_coords_dict[wid][0] for wid in well_ids])
        y = np.array([well_coords_dict[wid][1] for wid in well_ids])
        heads = np.array([heads_dict[wid] for wid in well_ids])
        
        # Convert timestamp to days since reference
        if isinstance(timestamp, pd.Timestamp):
            t_days = (timestamp - reference_date).days
        else:
            t_days = (pd.Timestamp(timestamp) - pd.Timestamp(reference_date)).days
        
        t = np.full(len(well_ids), t_days)
        
        all_x.extend(x)
        all_y.extend(y)
        all_t.extend(t)
        all_heads.extend(heads)
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_t = np.array(all_t)
    all_heads = np.array(all_heads)
    n_samples = np.array(n_samples)
    
    print(f"  Total data points for fitting: {len(all_heads)}")
    print(f"  Average number of data points per time step: {np.mean(n_samples):.1f}")
    print(f"  Spatial extent: X=[{all_x.min():.0f}, {all_x.max():.0f}], Y=[{all_y.min():.0f}, {all_y.max():.0f}]")
    print(f"  Temporal extent: {all_t.min():.0f} to {all_t.max():.0f} days")
    
    # Calculate max distance if not provided (use spatial distance only for max_dist)
    if max_dist is None:
        spatial_distances = np.sqrt((all_x[:, None] - all_x[None, :])**2 + 
                                    (all_y[:, None] - all_y[None, :])**2)
        max_dist = np.percentile(spatial_distances[spatial_distances > 0], 50)
    
    print(f"  Maximum spatial distance for variogram: {max_dist:.0f} m")
    
    # Try multiple model types and temporal scaling parameters
    models_to_try = [
        ('Gaussian', 'gaussian'),
        ('Exponential', 'exponential'),
        ('Spherical', 'spherical'),
    ]
    
    best_params = None
    best_score = np.inf
    best_model_name = None
    best_bin_centers = None
    best_bin_gammas = None
    
    # Try different temporal scaling values
    k_temporal_values = [0.1, 0.5, 1.0, 5.0, 10.0]
    
    print("\n  Testing different temporal scaling parameters...")
    
    for k_temporal in k_temporal_values:
        print(f"\n    k_temporal = {k_temporal}")
        
        # Compute space-time distances for all pairs
        st_distances = compute_spacetime_distance(
            all_x, all_y, all_t,
            all_x, all_y, all_t,
            k_temporal=k_temporal
        )
        
        # adapative binning
        max_st_dist = np.percentile(st_distances[st_distances > 0], 75)  # Use 75th percentile
        # Get upper triangle distances (unique pairs)
        triu_indices = np.triu_indices_from(st_distances, k=1)
        distances = st_distances[triu_indices]
        distances = distances[distances > 0]
        distances = np.sort(distances)
        
        n_pairs = len(distances)
        max_dist = np.percentile(distances, 75)  # Use 75th percentile
        
        # Calculate how many bins we can support
        n_bins_possible = n_pairs // 20
        n_bins = min(n_bins_possible, 30)
        n_bins = max(n_bins, 10)  # At least 10 bins
        
        # Create bins with approximately equal number of pairs
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(distances[distances <= max_dist], percentiles)
        bin_edges = np.unique(bin_edges) 

        # # Calculate experimental variogram
        # # gamma(h) = 0.5 * mean((z_i - z_j)^2) for all pairs at distance h
        # n_bins = 30
        # max_st_dist = np.percentile(st_distances[st_distances > 0], 75)  # Use 75th percentile
        # bin_edges = np.linspace(0, max_st_dist, n_bins + 1)
        
        bin_centers = []
        bin_gammas = []
        
        for i in range(n_bins):
            mask = (st_distances >= bin_edges[i]) & (st_distances < bin_edges[i+1])
            # Only use upper triangle to avoid duplicate pairs
            mask = np.triu(mask, k=1)
            
            if mask.sum() > 5:  # Need minimum number of pairs
                indices = np.where(mask)
                gamma = 0.5 * np.mean((all_heads[indices[0]] - all_heads[indices[1]])**2)
                
                if np.isfinite(gamma):
                    bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
                    bin_gammas.append(gamma)
        
        if len(bin_centers) < 5:
            print(f"      Insufficient bins ({len(bin_centers)}), skipping...")
            continue
        
        bin_centers = np.array(bin_centers)
        bin_gammas = np.array(bin_gammas)
        
        # Try fitting each model type
        for model_name, model_type in models_to_try:
            try:
                # Initial parameter estimates
                var_init = np.var(all_heads) * 0.7
                len_init = max_st_dist / 3
                nugget_init = np.var(all_heads) * 0.1
                
                # Optimization function
                def objective(params_array):
                    if np.any(params_array <= 0):
                        return 1e10
                    
                    params = {
                        'sill': params_array[0],
                        'range': params_array[1],
                        'nugget': params_array[2]
                    }
                    
                    predicted_gamma = spacetime_variogram(bin_centers, params, model_type)
                    residuals = bin_gammas - predicted_gamma
                    # Weighted by number of pairs in each bin (inverse distance weighting)
                    weights = 1.0 / (bin_centers + 1.0)
                    return np.sum(weights * residuals**2)
                
                # Initial parameters
                initial_params = [var_init, len_init, nugget_init]
                
                # Bounds (all positive)
                bounds = [
                    (0, np.var(all_heads) * 2),  # sill
                    (0, max_st_dist * 2),         # range
                    (0, np.var(all_heads) * 0.5)  # nugget
                ]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    warnings.simplefilter("ignore", RuntimeWarning)
                    
                    result = minimize(objective, initial_params, bounds=bounds, 
                                     method='L-BFGS-B', options={'maxiter': 1000})
                    
                    if result.success:
                        params = {
                            'sill': result.x[0],
                            'range': result.x[1],
                            'nugget': result.x[2],
                            'k_temporal': k_temporal
                        }
                        
                        predicted = spacetime_variogram(bin_centers, params, model_type)
                        rmse = np.sqrt(np.mean((bin_gammas - predicted)**2))
                        
                        print(f"      {model_name}: RMSE = {rmse:.4f}")
                        
                        if rmse < best_score:
                            best_score = rmse
                            best_params = params
                            best_model_name = model_name
                            best_params['model_type'] = model_type
                            best_bin_centers = bin_centers
                            best_bin_gammas = bin_gammas
                            
            except Exception as e:
                print(f"      {model_name} fitting failed: {str(e)[:50]}...")
                continue
    
    # Fallback if no model fitted successfully
    if best_params is None:
        print("\n  Warning: Automatic fitting failed. Using manual parameters.")
        spatial_var = np.var(all_heads)
        best_params = {
            'sill': spatial_var * 0.7,
            'range': max_dist / 3,
            'nugget': spatial_var * 0.1,
            'k_temporal': 1.0,
            'model_type': 'gaussian'
        }
        best_model_name = "Gaussian (manual)"
        best_score = np.inf
    
    print(f"\n  Best Space-Time Variogram Model: {best_model_name}")
    print(f"    Variance (sill): {best_params['sill']:.2f}")
    print(f"    Range: {best_params['range']:.2f}")
    print(f"    Nugget: {best_params['nugget']:.2f}")
    print(f"    Temporal scaling (k): {best_params['k_temporal']:.4f}")
    if best_score < np.inf:
        print(f"    RMSE: {best_score:.4f}")
    
    # Plot variogram
    if best_bin_centers is not None and best_bin_gammas is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        ax.scatter(best_bin_centers, best_bin_gammas, 
                  label='Experimental variogram', 
                  color='red', s=50, alpha=0.7, zorder=3)
        
        x_plot = np.linspace(0, max(best_bin_centers), 100)
        y_plot = spacetime_variogram(x_plot, best_params, best_params['model_type'])
        ax.plot(x_plot, y_plot, 
                label=f'Fitted model ({best_model_name})', 
                linewidth=2.5, zorder=2)
        
        ax.axhline(y=best_params['sill'] + best_params['nugget'], 
                  color='gray', linestyle='--', alpha=0.5, 
                  label='Sill', zorder=1)
        
        ax.set_xlabel('Space-Time Distance', fontsize=12)
        ax.set_ylabel('Semivariance', fontsize=12)
        ax.set_title(f'Space-Time Variogram (k_temporal={best_params["k_temporal"]:.2f})', 
                    fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        file = base_path / 'figures' / 'spatio_temporal_regression_kriging' / f'{fig_name}.png'
        fig.savefig(file, dpi=250, bbox_inches='tight')
        plt.close(fig)
    
    return best_params, best_model_name, best_score


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
    drift_values : array
        Precomputed drift values at observation points
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
    
    # Compute drift at observation points
    obs_points = np.column_stack([x, y])
    drift_at_obs = idw_interpolator(obs_points)
    
    # Define drift function
    def drift_idw(x_pts, y_pts):
        """External drift based on IDW"""
        points = np.column_stack([x_pts.ravel(), y_pts.ravel()])
        return idw_interpolator(points).reshape(x_pts.shape)
    
    return drift_idw, drift_at_obs


def universal_kriging_numpy(x, y, heads, params, drift_values, 
                            grid_x, grid_y, topography=None,
                            nugget_boost=1e-6):  # NEW PARAMETER
    """
    Perform Universal Kriging using pure numpy/scipy with improved stability
    
    Parameters:
    -----------
    nugget_boost : float
        Additional regularization to add to diagonal (default: 1e-6)
    """
    n_obs = len(x)
    
    # Compute covariance matrix for observations
    distances_obs = compute_spacetime_distance(
        x, y, np.zeros(n_obs),
        x, y, np.zeros(n_obs),
        k_temporal=params.get('k_temporal', 1.0)
    )
    
    C_obs = spacetime_covariance(distances_obs, params, params['model_type'])
    
    # IMPROVED REGULARIZATION
    # 1. Add nugget_boost (stronger than before)
    # 2. Scale based on variance to ensure appropriate magnitude
    variance_scale = params['sill'] + params['nugget']
    regularization = nugget_boost * variance_scale
    
    C_obs += np.eye(n_obs) * regularization
    
    # Check condition number
    cond_number = np.linalg.cond(C_obs)
    if cond_number > 1e10:
        print(f"    WARNING: High condition number ({cond_number:.2e}), increasing regularization")
        C_obs += np.eye(n_obs) * regularization * 10
    
    # Build kriging system with drift
    F = np.column_stack([np.ones(n_obs), drift_values])
    n_drift = F.shape[1]
    
    # Build full kriging matrix
    K = np.zeros((n_obs + n_drift, n_obs + n_drift))
    K[:n_obs, :n_obs] = C_obs
    K[:n_obs, n_obs:] = F
    K[n_obs:, :n_obs] = F.T
    
    # Create meshgrid for prediction
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    n_pred = X_grid.size
    
    # Flatten grid
    x_pred = X_grid.ravel()
    y_pred = Y_grid.ravel()
    
    # Initialize output arrays
    interpolated = np.zeros(n_pred)
    variance = np.zeros(n_pred)
    
    # Precompute drift at all prediction points
    from scipy.interpolate import griddata
    drift_at_pred = griddata(
        np.column_stack([x, y]),
        drift_values,
        np.column_stack([x_pred, y_pred]),
        method='linear',
        fill_value=np.mean(drift_values)
    )
    
    # Perform kriging in batches for efficiency
    batch_size = 1000
    n_batches = int(np.ceil(n_pred / batch_size))
    
    print(f"    Kriging {n_pred} points in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, n_pred)
        
        x_batch = x_pred[start_idx:end_idx]
        y_batch = y_pred[start_idx:end_idx]
        drift_batch = drift_at_pred[start_idx:end_idx]
        
        n_batch = len(x_batch)
        
        # Compute covariance between prediction points and observations
        distances_pred = compute_spacetime_distance(
            x_batch, y_batch, np.zeros(n_batch),
            x, y, np.zeros(n_obs),
            k_temporal=params.get('k_temporal', 1.0)
        )
        
        c_pred = spacetime_covariance(distances_pred, params, params['model_type'])
        
        # Build right-hand side for each prediction point
        for i in range(n_batch):
            # RHS vector
            rhs = np.zeros(n_obs + n_drift)
            rhs[:n_obs] = c_pred[i, :]
            rhs[n_obs] = 1.0  # constant drift term
            rhs[n_obs + 1] = drift_batch[i]  # IDW drift term
            
            # Solve kriging system
            try:
                weights = solve(K, rhs, assume_a='sym')
            except np.linalg.LinAlgError:
                # Fallback to least squares if system is singular
                weights, _, _, _ = np.linalg.lstsq(K, rhs, rcond=None)
            
            # Prediction
            interpolated[start_idx + i] = np.dot(weights[:n_obs], heads)
            
            # Variance
            c0 = params['sill'] + params['nugget']
            variance[start_idx + i] = c0 - np.dot(weights, rhs)
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"      Completed batch {batch_idx + 1}/{n_batches}")
    
    # Reshape to grid
    interpolated_heads = interpolated.reshape(X_grid.shape)
    kriging_variance = np.maximum(0, variance.reshape(X_grid.shape))  # Ensure non-negative
    
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
    file = base_path / 'figures' / 'spatio_temporal_regression_kriging' / f'{method.lower()}_{fig_name}.png'
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
    file = base_path / 'figures' / 'spatio_temporal_regression_kriging' / f'{method.lower()}_{fig_name}_depth.png'
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
    file = base_path / 'figures' / 'spatio_temporal_regression_kriging' / f'interpolation_difference_observed_interpolated_heads_{year_month}.csv'
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

file = base_path / 'figures' / 'spatio_temporal_regression_kriging' / 'monthly_pseudowells.csv'
df_gw_heads_pseudowells.to_csv(file, sep=";", index=True)

well_coords_dict_fissured = {}
for i, station_id in enumerate(df_gw_heads_pseudowells.columns):
    well_coords_dict_fissured[station_id] = (x_fissured[i], y_fissured[i])

# Create time series dictionary (only for porous aquifer wells initially)
timeseries_dict_fissured = {}
for timestamp in df_gw_heads_pseudowells.index:  # Use subset for fitting
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
print("Preparing Time Series Data for Space-Time Variogram Fitting")
print("=" * 70)

# Create well coordinates dictionary
well_coords_dict_porous = {}
for idx, row in groundwater_observation_wells.iterrows():
    station_id = row['station_id'].replace('/', '_')
    if station_id in df_gw_heads.columns:
        well_coords_dict_porous[station_id] = (row.geometry.x, row.geometry.y)

# Create time series dictionary (only for porous aquifer wells initially)
timeseries_dict_porous = {}
for timestamp in df_gw_heads.index:  # Use subset for fitting
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
# FIT SPACE-TIME VARIOGRAM MODELS
# ============================================================================

print("\n" + "=" * 70)
print("Fitting Space-Time Variogram Models")
print("=" * 70)

# Fit for porous aquifer
print("\n" + "-" * 70)
print("POROUS AQUIFER")
print("-" * 70)
params_porous, model_name_porous, rmse_model_porous = fit_spacetime_variogram_model(
    timeseries_dict_porous,
    well_coords_dict_porous,
    reference_date=df_gw_heads.index[0],
    n_sample_times=len(df_gw_heads.index),
    fig_name='variogram_spacetime_fit_porous'
)

# # Fit for fissured aquifer
# print("\n" + "-" * 70)
# print("FISSURED AQUIFER")
# print("-" * 70)
# params_fissured, model_name_fissured, rmse_model_fissured = fit_spacetime_variogram_model(
#     timeseries_dict_fissured,
#     well_coords_dict_fissured,
#     reference_date=df_gw_heads_pseudowells.index[0],
#     n_sample_times=len(df_gw_heads_pseudowells.index),
#     fig_name='variogram_spacetime_fit_fissured'
# )

# # Fit for combined aquifer
# print("\n" + "-" * 70)
# print("COMBINED AQUIFER")
# print("-" * 70)
# params_combined, model_name_combined, rmse_model_combined = fit_spacetime_variogram_model(
#     timeseries_dict_combined,
#     well_coords_dict_combined,
#     reference_date=df_gw_heads.index[0],
#     n_sample_times=len(df_gw_heads.index),
#     fig_name='variogram_spacetime_fit_combined'
# )

# ============================================================================
# CREATE NETCDF OUTPUT FILE
# ============================================================================

ds = xr.Dataset()
ds["spatial_ref"] = spatial_ref
file = base_path / "observations" / "monthly_spatio_temporal_regression_kriging.nc"
ds.to_netcdf(file, engine="h5netcdf")
ds.close()

file = base_path / "observations" / "monthly_spatio_temporal_regression_kriging.nc"
with h5netcdf.File(file, "a", decode_vlen_strings=False) as f:
    f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title="Interpolated groundwater heads using space-time regression kriging with IDW drift in the Dreisam-Möhlin-Neumagen catchment",
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

    v = f.create_variable(
        "interpolated_gw_heads_fissured", ("Time", "y", "x"), np.float64, 
        compression="gzip", compression_opts=1
    )
    v.attrs.update(long_name="Interpolated groundwater heads (fissured)", 
                   units="m a.s.l.", grid_mapping="spatial_ref")

    v = f.create_variable(
        "interpolated_gw_heads", ("Time", "y", "x"), np.float64, 
        compression="gzip", compression_opts=1
    )
    v.attrs.update(long_name="Interpolated groundwater heads (combined)", 
                   units="m a.s.l.", grid_mapping="spatial_ref")

    v = f.create_variable(
        "uncertainty_porous", ("Time", "y", "x"), np.float64, 
        compression="gzip", compression_opts=1
    )
    v.attrs.update(long_name="Kriging variance (porous)", 
                   units="m2", grid_mapping="spatial_ref")

    v = f.create_variable(
        "uncertainty_fissured", ("Time", "y", "x"), np.float64, 
        compression="gzip", compression_opts=1
    )
    v.attrs.update(long_name="Kriging variance (fissured)", 
                   units="m2", grid_mapping="spatial_ref")

    v = f.create_variable(
        "uncertainty_combined", ("Time", "y", "x"), np.float64, 
        compression="gzip", compression_opts=1
    )
    v.attrs.update(long_name="Kriging variance (combined)", 
                   units="m2", grid_mapping="spatial_ref")

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
        "rmse_porous", ("scalar",), float
    )
    v.attrs.update(long_name="Root mean square error of the space-time kriging model of porous aquifer")
    v[0] = rmse_model_porous

    # v = f.create_variable(
    #     "rmse_fissured", ("scalar",), float
    # )
    # v.attrs.update(long_name="Root mean square error of the space-time kriging model of fissured aquifer")
    # v[0] = rmse_model_fissured
    
    # v = f.create_variable(
    #     "rmse_combined", ("scalar",), float
    # )
    # v.attrs.update(long_name="Root mean square error of the space-time kriging model of combined aquifer")
    # v[0] = rmse_model_combined

    v = f.create_variable(
        "k_temporal_porous", ("scalar",), float
    )
    v.attrs.update(long_name="Temporal scaling parameter for porous aquifer space-time kriging")
    v[0] = params_porous['k_temporal']

    # v = f.create_variable(
    #     "k_temporal_fissured", ("scalar",), float
    # )
    # v.attrs.update(long_name="Temporal scaling parameter for fissured aquifer space-time kriging")
    # v[0] = params_fissured['k_temporal']

    # v = f.create_variable(
    #     "k_temporal_combined", ("scalar",), float
    # )
    # v.attrs.update(long_name="Temporal scaling parameter for combined aquifer space-time kriging")
    # v[0] = params_combined['k_temporal']

# ============================================================================
# LOOP OVER TIME STEPS AND PERFORM KRIGING
# ============================================================================

print("\n" + "=" * 70)
print("Processing Time Series with Space-Time Kriging")
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
    print("  Creating IDW drift for porous aquifer...")
    drift_func_porous, drift_vals_porous = create_idw_drift_function(
        x_porous, y_porous, heads_porous,
        grid_x, grid_y,
        idw_power=2,
        smoothing=1.0,
        max_neighbors=10
    )
    
    # Perform universal kriging with numpy implementation
    print("  Performing universal kriging for porous aquifer...")
    interpolated_heads_porous, variance_porous = universal_kriging_numpy(
        x_porous, y_porous, heads_porous,
        params_porous,
        drift_vals_porous,
        grid_x, grid_y,
        topography=topography
    )
    
    # Plot results (only for first, middle, and last time steps)
    if month_i in [0, len(df_gw_heads)//12, len(df_gw_heads)-1]:
        plot_interpolation_results(
            grid_x, grid_y, interpolated_heads_porous, variance_porous,
            topography, x_porous, y_porous, heads_porous, depths_porous,
            mask_porous, method='SpaceTime', 
            fig_name=f'porous_{year_month}'
        )

    # # Get current time step data for fissured aquifer
    # df_gw_heads_current = df_gw_heads_pseudowells.loc[timestamp, :].to_frame()
    # df_gw_heads_current.columns = ['groundwater_head']
    
    # heads_fissured = df_gw_heads_current['groundwater_head'].values
    # depths_fissured = np.zeros_like(x_fissured)
    
    # for i in range(len(x_fissured)):
    #     row, col = src.index(x_fissured[i], y_fissured[i])
    #     depths_fissured[i] = topography[row, col] - heads_fissured[i]
    
    # # Create IDW drift function for this time step
    # print("  Creating IDW drift for fissured aquifer...")
    # drift_func_fissured, drift_vals_fissured = create_idw_drift_function(
    #     x_fissured, y_fissured, heads_fissured,
    #     grid_x, grid_y,
    #     idw_power=2,
    #     smoothing=1.0,
    #     max_neighbors=10
    # )
    
    # # Perform universal kriging with numpy implementation
    # print("  Performing universal kriging for fissured aquifer...")
    # interpolated_heads_fissured, variance_fissured = universal_kriging_numpy(
    #     x_fissured, y_fissured, heads_fissured,
    #     params_fissured,
    #     drift_vals_fissured,
    #     grid_x, grid_y,
    #     topography=topography
    # )
    
    # # Plot results (only for first, middle, and last time steps)
    # if month_i in [0, len(df_gw_heads_pseudowells)//2, len(df_gw_heads_pseudowells)-1]:
    #     plot_interpolation_results(
    #         grid_x, grid_y, interpolated_heads_fissured, variance_fissured,
    #         topography, x_fissured, y_fissured, heads_fissured, depths_fissured,
    #         mask_fissured, method='SpaceTime', 
    #         fig_name=f'fissured_{year_month}'
    #     )

    # # Get current time step data for combined aquifer
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
    
    # # Extract coordinates and heads for combined aquifer
    # x_combined = gdf_gw_heads_combined.geometry.x.values
    # y_combined = gdf_gw_heads_combined.geometry.y.values
    # heads_combined = gdf_gw_heads_combined['groundwater_head'].values
    # depths_combined = np.zeros_like(x_combined)
    
    # for i in range(len(x_combined)):
    #     row, col = src.index(x_combined[i], y_combined[i])
    #     depths_combined[i] = topography[row, col] - heads_combined[i]
    
    # # Create IDW drift function for this time step
    # print("  Creating IDW drift for combined aquifer...")
    # drift_func_combined, drift_vals_combined = create_idw_drift_function(
    #     x_combined, y_combined, heads_combined,
    #     grid_x, grid_y,
    #     idw_power=2,
    #     smoothing=1.0,
    #     max_neighbors=10
    # )
    
    # # Perform universal kriging with numpy implementation
    # print("  Performing universal kriging for combined aquifer...")
    # interpolated_heads_combined, variance_combined = universal_kriging_numpy(
    #     x_combined, y_combined, heads_combined,
    #     params_combined,
    #     drift_vals_combined,
    #     grid_x, grid_y,
    #     topography=topography
    # )
    
    # # Plot results (only for first, middle, and last time steps)
    # if month_i in [0, len(df_gw_heads)//2, len(df_gw_heads)-1]:
    #     plot_interpolation_results(
    #         grid_x, grid_y, interpolated_heads_combined, variance_combined,
    #         topography, x_combined, y_combined, heads_combined, depths_combined,
    #         mask_combined, method='SpaceTime', 
    #         fig_name=f'combined_{year_month}'
    #     )

    
    # Save to NetCDF
    file = base_path / "observations" / "monthly_spatio_temporal_regression_kriging.nc"
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

print("\n" + "=" * 70)
print("Space-Time Kriging Complete!")
print("=" * 70)
print("\nKey improvements:")
print("- Pure numpy/scipy implementation (no GSTools dependency)")
print("- Space-time variogram with temporal correlation")
print("- Optimal temporal scaling parameter (k_temporal) automatically determined")
print("- Universal Kriging with IDW drift using matrix inversion")
print("- Separate models for porous, fissured, and combined aquifers")
print("=" * 70)