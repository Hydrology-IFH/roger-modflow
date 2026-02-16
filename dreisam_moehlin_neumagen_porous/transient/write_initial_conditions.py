from pathlib import Path
import numpy as np
import scipy as sp
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
import gstools as gs
from scipy.optimize import OptimizeWarning
from scipy.spatial import cKDTree
import h5netcdf
import datetime
import warnings

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
    file = base_path / 'figures' / 'initial_conditions' / f'{fig_name}.png'
    fig.savefig(file, dpi=300, bbox_inches='tight')
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
        Number of grid points in each direction
    power : float
        Power parameter for IDW (typically 1-3)
        Higher values = more weight to nearby points
    smoothing : float
        Smoothing parameter added to distances to avoid division by zero
        and reduce influence of very close points
    max_neighbors : int, optional
        Maximum number of neighbors to use for interpolation
        If None, uses all points
    
    Returns:
    --------
    grid_x, grid_y : arrays
        Grid coordinates
    interpolated_heads : array
        Interpolated head values
    interpolation_variance : array
        Variance estimate (based on distance-weighted variance)
    """
    # Create interpolation grid
    grid_x = np.linspace(grid_extent[0], grid_extent[1], int((grid_extent[1] - grid_extent[0]) / grid_resolution) + 1)
    grid_y = np.linspace(grid_extent[2], grid_extent[3], int((grid_extent[2] - grid_extent[3]) / grid_resolution) + 1)
    X, Y = np.meshgrid(grid_x, grid_y)
    
    # Flatten grid for easier computation
    grid_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(np.column_stack([x, y]))
    
    # Initialize output arrays
    interpolated = np.zeros(len(grid_points))
    variance = np.zeros(len(grid_points))
    
    # Determine number of neighbors to use
    n_neighbors = len(x) if max_neighbors is None else min(max_neighbors, len(x))
    
    # Interpolate for each grid point
    for i, point in enumerate(grid_points):
        # Find nearest neighbors
        distances, indices = tree.query(point, k=n_neighbors)
        
        # Add smoothing parameter to avoid division by zero
        distances = distances + smoothing
        
        # Calculate weights using inverse distance
        weights = 1.0 / (distances ** power)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate interpolated value
        interpolated[i] = np.sum(weights * heads[indices])
        
        # Calculate variance estimate (weighted variance of nearby points)
        local_mean = np.sum(weights * heads[indices])
        variance[i] = np.sum(weights * (heads[indices] - local_mean) ** 2)
    
    # Reshape back to grid
    interpolated_heads = interpolated.reshape(X.shape)
    interpolation_variance = variance.reshape(X.shape)
    
    return grid_x, grid_y, interpolated_heads, interpolation_variance


def analyze_variogram(x, y, heads, max_dist=None, fig_name='variogram_analysis'):
    """
    Perform variogram analysis to determine spatial correlation structure
    
    Parameters:
    -----------
    x, y : arrays
        Coordinates of measurement points
    heads : array
        Groundwater head values
    max_dist : float, optional
        Maximum distance for variogram calculation
    
    Returns:
    --------
    model : GSTools CovModel
        Fitted variogram model
    """
    # Calculate maximum distance if not provided
    if max_dist is None:
        # Use approximately 1/3 of the maximum distance
        distances = np.sqrt((x[:, None] - x[None, :]) ** 2 + 
                          (y[:, None] - y[None, :]) ** 2)
        max_dist = np.percentile(distances[distances > 0], 50)
    
    # Calculate experimental variogram with more bins
    bin_edges = np.linspace(0, max_dist, 30)
    bin_center, gamma = gs.vario_estimate((x, y), heads, bin_edges=bin_edges)
    
    # Remove any NaN or infinite values
    valid_mask = np.isfinite(gamma) & np.isfinite(bin_center) & (bin_center > 0)
    bin_center = bin_center[valid_mask]
    gamma = gamma[valid_mask]
    
    # Try multiple model types and fitting strategies
    models_to_try = [
        ('Gaussian', gs.Gaussian),
        ('Exponential', gs.Exponential),
        ('Spherical', gs.Spherical),
    ]
    
    best_model = None
    best_score = np.inf
    
    for model_name, ModelClass in models_to_try:
        try:
            # Initialize model with reasonable default parameters
            model = ModelClass(dim=2)
            
            # Estimate initial parameters
            var_init = np.var(heads) * 0.8  # Initial variance
            len_init = max_dist / 3  # Initial correlation length
            nugget_init = np.var(heads) * 0.1  # Initial nugget
            
            # Set initial parameters
            model.var = var_init
            model.len_scale = len_init
            model.nugget = nugget_init
            
            # Try to fit with increased maxfev
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                
                try:
                    # Fit with more iterations and bounds
                    fit_result = model.fit_variogram(
                        bin_center, 
                        gamma, 
                        nugget=True,
                        max_eval=10000,  # Increased iterations
                        weights='inv',   # Use inverse distance weighting
                    )
                    
                    # Calculate fit quality (RMSE)
                    predicted = model.variogram(bin_center)
                    rmse = np.sqrt(np.mean((gamma - predicted) ** 2))
                    
                    if rmse < best_score and model.var > 0 and model.len_scale > 0:
                        best_score = rmse
                        best_model = model
                        best_model_name = model_name
                        
                except (RuntimeError, ValueError) as e:
                    print(f"   Warning: {model_name} fitting failed: {str(e)[:50]}...")
                    continue
                    
        except Exception as e:
            print(f"   Warning: Could not fit {model_name} model: {str(e)[:50]}...")
            continue
    
    # If no model fitted successfully, use manual parameters
    if best_model is None:
        print("   Warning: Automatic fitting failed. Using manual parameters.")
        best_model = gs.Gaussian(dim=2)
        best_model.var = np.var(heads) * 0.7
        best_model.len_scale = max_dist / 3
        best_model.nugget = np.var(heads) * 0.1
        best_model_name = "Gaussian (manual)"
    
    print(f"\nBest Variogram Model: {best_model_name}")
    print(f"  Variance (sill): {best_model.var:.2f}")
    print(f"  Correlation length: {best_model.len_scale:.2f}")
    print(f"  Nugget: {best_model.nugget:.2f}")
    if best_score < np.inf:
        print(f"  RMSE: {best_score:.4f}")
    
    # Plot experimental and fitted variogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.scatter(bin_center, gamma, label='Experimental variogram', 
                color='red', s=50, alpha=0.7, zorder=3)
    
    # Plot fitted model
    x_plot = np.linspace(0, max(bin_center), 100)
    ax.plot(x_plot, best_model.variogram(x_plot), 
             label=f'Fitted model ({best_model_name})', linewidth=2.5, zorder=2)
    
    # Add sill line
    ax.axhline(y=best_model.var + best_model.nugget, 
                color='gray', linestyle='--', alpha=0.5, 
                label='Sill', zorder=1)
    
    ax.set_xlabel('Distance (m)', fontsize=12)
    ax.set_ylabel('Semivariance', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(bin_center) * 1.05)
    ax.set_ylim(0, max(gamma) * 1.1)
    fig.tight_layout()
    file = base_path / 'figures' / 'initial_conditions' / f'{fig_name}.png'
    fig.savefig(file, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return best_model

def ordinary_kriging_interpolation(x, y, heads, model, grid_resolution=50, grid_extent=(0, 1000, 0, 1000)):
    """
    Perform Ordinary Kriging interpolation
    
    Parameters:
    -----------
    x, y : arrays
        Coordinates of measurement points
    heads : array
        Groundwater head values
    model : GSTools CovModel
        Variogram model
    grid_resolution : int
        Number of grid points in each direction
    
    Returns:
    --------
    grid_x, grid_y : arrays
        Grid coordinates
    interpolated_heads : array
        Interpolated head values
    kriging_variance : array
        Kriging variance (uncertainty)
    krige : Krige instance
        The kriging object (needed for CondSRF)
    """
    # Create interpolation grid
    grid_x = np.linspace(grid_extent[0], grid_extent[1], int((grid_extent[1] - grid_extent[0]) / grid_resolution) + 1)
    grid_y = np.linspace(grid_extent[2], grid_extent[3], int((grid_extent[2] - grid_extent[3]) / grid_resolution) + 1)
    
    # Perform Ordinary Kriging
    krige = gs.krige.Ordinary(
        model=model,
        cond_pos=[x, y],
        cond_val=heads
    )
    
    # Interpolate on the grid
    interpolated_heads, kriging_variance = krige.structured([grid_x, grid_y])
    
    return grid_x, grid_y, interpolated_heads.T, kriging_variance.T, krige


def universal_kriging_interpolation(x, y, heads, model, grid_resolution=50, grid_extent=(0, 1000, 0, 1000),
                                   use_idw_trend=False, idw_power=2):
    """
    Perform Universal Kriging with linear trend or IDW-based trend
    
    Parameters:
    -----------
    x, y : arrays
        Coordinates of measurement points
    heads : array
        Groundwater head values
    model : GSTools CovModel
        Variogram model
    grid_resolution : int
        Number of grid points in each direction
    use_idw_trend : bool
        If True, use IDW interpolation as the trend component
        If False, use linear drift functions
    idw_power : float
        Power parameter for IDW trend (if use_idw_trend=True)
    
    Returns:
    --------
    grid_x, grid_y : arrays
        Grid coordinates
    interpolated_heads : array
        Interpolated head values
    kriging_variance : array
        Kriging variance
    krige : Krige instance
        The kriging object
    """
    # Create interpolation grid
    grid_x = np.linspace(grid_extent[0], grid_extent[1], int((grid_extent[1] - grid_extent[0]) / grid_resolution) + 1)
    grid_y = np.linspace(grid_extent[2], grid_extent[3], int((grid_extent[2] - grid_extent[3]) / grid_resolution) + 1)
    
    if use_idw_trend:
        # Use IDW as external drift
        print("   Using IDW interpolation as external drift...")
        
        # First, compute IDW interpolation as trend
        _, _, idw_trend, _ = idw_interpolation(
            x, y, heads, 
            grid_resolution=grid_resolution,
            grid_extent=grid_extent, 
            power=idw_power,
            smoothing=1.0,
            max_neighbors=10
        )
        
        # Define drift function based on IDW
        # We need to create a callable that GSTools can use
        from scipy.interpolate import RegularGridInterpolator
        
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
            points = np.column_stack([y_pts.ravel(), x_pts.ravel()])
            return idw_interpolator(points).reshape(x_pts.shape)
        
        # Perform Universal Kriging with IDW trend
        krige = gs.krige.Universal(
            model=model,
            cond_pos=[x, y],
            cond_val=heads,
            drift_functions=[drift_idw]
        )
        
    else:
        # Use traditional linear drift functions
        print("   Using linear drift functions...")
        
        # Define drift functions (linear trend in x and y)
        def drift_x(x, y):
            return x
        
        def drift_y(x, y):
            return y
        
        # Perform Universal Kriging
        krige = gs.krige.Universal(
            model=model,
            cond_pos=[x, y],
            cond_val=heads,
            drift_functions=[drift_x, drift_y]
        )
    
    # Interpolate on the grid
    interpolated_heads, kriging_variance = krige.structured([grid_x, grid_y])
    
    return grid_x, grid_y, interpolated_heads.T, kriging_variance.T, krige


def generate_conditional_random_field(krige, grid_resolution=50, n_realizations=3, grid_extent=(0, 1000, 0, 1000)):
    """
    Generate conditional random field (multiple realizations)
    
    Parameters:
    -----------
    krige : gs.krige instance
        Kriging instance with model and conditioning data
    grid_resolution : int
        Number of grid points in each direction
    n_realizations : int
        Number of realizations to generate
    
    Returns:
    --------
    grid_x, grid_y : arrays
        Grid coordinates
    realizations : list
        List of conditional random field realizations
    """
    # Create interpolation grid
    grid_x = np.linspace(grid_extent[0], grid_extent[1], int((grid_extent[1] - grid_extent[0]) / grid_resolution) + 1)
    grid_y = np.linspace(grid_extent[2], grid_extent[3], int((grid_extent[2] - grid_extent[3]) / grid_resolution) + 1)
    
    # Create conditional spatial random field using the kriging instance
    cond_srf = gs.CondSRF(krige)
    
    # Generate multiple realizations
    realizations = []
    seed = gs.random.MasterRNG(42)
    
    for i in range(n_realizations):
        field = cond_srf.structured([grid_x, grid_y], seed=seed())
        realizations.append(field.T)
    
    return grid_x, grid_y, realizations


def plot_interpolation_results(grid_x, grid_y, interpolated_heads,
                               kriging_variance, topography, x, y, heads, depths, mask, method='Ordinary', fig_name='kriging_results'):
    """Plot interpolation results and kriging variance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot interpolated surface
    rows, cols = np.where(mask)
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    X, Y = np.meshgrid(grid_x[x1:x2+1], grid_y[y1:y2+1])

    interpolated_heads[~mask] = np.nan
    kriging_variance[~mask] = np.nan
    interpolated_heads = interpolated_heads[y1:y2+1, x1:x2+1]
    kriging_variance = kriging_variance[y1:y2+1, x1:x2+1]
    topography = topography[y1:y2+1, x1:x2+1]
    
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
    ax1.set_title(f'Interpolated Heads', fontsize=14)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot kriging variance (uncertainty)
    variance_plot = ax2.contourf(X, Y, kriging_variance, levels=20, 
                                cmap='Reds', alpha=0.8)
    ax2.scatter(x, y, c='blue', s=80, marker='x', 
               linewidths=2, label='Measurements')
    
    cbar2 = plt.colorbar(variance_plot, ax=ax2, 
                        label='Variance (m²)')
    ax2.set_xlabel('X-coordinate', fontsize=12)
    ax2.set_ylabel('Y-coordinate', fontsize=12)
    ax2.set_title(f'Uncertainty', fontsize=14)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    file = base_path / 'figures' / 'initial_conditions' / f'{method.lower()}_{fig_name}.png'
    fig.savefig(file, dpi=300, bbox_inches='tight')
    plt.close(fig)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    depths_array = np.where(topography - interpolated_heads > 25, 25, topography - interpolated_heads)
    contour = ax1.contourf(X, Y, depths_array, levels=10, 
                          cmap='viridis', alpha=0.8, vmin=0, vmax=25)
    ax1.scatter(x, y, c=depths, s=80, cmap='viridis', vmin=0, vmax=25,  
               edgecolors='white', linewidths=1.5, label='Measurements')
    contour_lines = ax1.contour(X, Y, depths_array, levels=10, 
                                colors='black', alpha=0.4, linewidths=0.5)
    ax1.clabel(contour_lines, inline=True, fontsize=8)
    
    cbar1 = plt.colorbar(contour, ax=ax1, label='Groundwater depth (m)')
    ax1.set_xlabel('X-coordinate', fontsize=12)
    ax1.set_ylabel('Y-coordinate', fontsize=12)
    ax1.set_title(f'Interpolated Depths', fontsize=14)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot kriging variance (uncertainty)
    variance_plot = ax2.contourf(X, Y, kriging_variance, levels=20, 
                                cmap='Reds', alpha=0.8)
    ax2.scatter(x, y, c='blue', s=80, marker='x', 
               linewidths=2, label='Measurements')
    
    cbar2 = plt.colorbar(variance_plot, ax=ax2, 
                        label='Variance (m²)')
    ax2.set_xlabel('X-coordinate', fontsize=12)
    ax2.set_ylabel('Y-coordinate', fontsize=12)
    ax2.set_title(f'Uncertainty', fontsize=14)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    file = base_path / 'figures' / 'initial_conditions' / f'{method.lower()}_{fig_name}_depth.png'
    fig.savefig(file, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_conditional_realizations(grid_x, grid_y, realizations, topography, x, y, heads, depths, mask, fig_name='conditional_realizations'):
    """Plot multiple conditional random field realizations"""
    
    rows, cols = np.where(mask)
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    X, Y = np.meshgrid(grid_x[x1:x2+1], grid_y[y1:y2+1])
    topography = topography[y1:y2+1, x1:x2+1]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ax, realization) in enumerate(zip(axes, realizations)):
        realization[~mask] = np.nan
        realization = realization[y1:y2+1, x1:x2+1]
        contour = ax.contourf(X, Y, realization, levels=20, 
                            cmap='viridis', alpha=0.8)
        ax.scatter(x, y, c=heads, s=60, cmap='viridis', 
                  edgecolors='white', linewidths=1)
        plt.colorbar(contour, ax=ax, label='Groundwater Head (m)')
        ax.set_xlabel('X-coordinate', fontsize=11)
        ax.set_ylabel('Y-coordinate', fontsize=11)
        ax.set_title(f'Realization {i+1}', fontsize=13)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    file = base_path / 'figures' / 'initial_conditions' / f'{fig_name}.png'
    fig.savefig(file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (ax, realization) in enumerate(zip(axes, realizations)):
        realization[~mask] = np.nan
        realization = realization[y1:y2+1, x1:x2+1]
        depths_array = np.where(topography - realization > 25, 25, topography - realization)
        contour = ax.contourf(X, Y, depths_array, levels=10, 
                            cmap='viridis', alpha=0.8, vmin=0, vmax=25)
        ax.scatter(x, y, c=depths, s=60, cmap='viridis', vmin=0, vmax=25,
                  edgecolors='white', linewidths=1)
        plt.colorbar(contour, ax=ax, label='Groundwater depth (m)')
        ax.set_xlabel('X-coordinate', fontsize=11)
        ax.set_ylabel('Y-coordinate', fontsize=11)
        ax.set_title(f'Realization {i+1}', fontsize=13)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    file = base_path / 'figures' / 'initial_conditions' / f'{fig_name}_depth.png'
    fig.savefig(file, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_comparison(grid_x, grid_y, interpolated_heads1, interpolated_heads2, topography, mask, fig_name='comparison_results'):
    """Plot interpolation results and kriging variance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot interpolated surface
    rows, cols = np.where(mask)
    y1, y2 = rows.min(), rows.max()
    x1, x2 = cols.min(), cols.max()
    X, Y = np.meshgrid(grid_x[x1:x2+1], grid_y[y1:y2+1])

    interpolated_heads1[~mask] = np.nan
    interpolated_heads1 = interpolated_heads1[y1:y2+1, x1:x2+1]
    interpolated_heads2[~mask] = np.nan
    interpolated_heads2 = interpolated_heads2[y1:y2+1, x1:x2+1]
    topography = topography[y1:y2+1, x1:x2+1]
    
    contour = ax1.contourf(X, Y, interpolated_heads1, levels=20, 
                          cmap='viridis', alpha=0.8)
    contour_lines = ax1.contour(X, Y, interpolated_heads1, levels=10, 
                                colors='black', alpha=0.4, linewidths=0.5)
    ax1.clabel(contour_lines, inline=True, fontsize=8)
    
    cbar1 = plt.colorbar(contour, ax=ax1, label='Groundwater Head (m)')
    ax1.set_xlabel('X-coordinate', fontsize=12)
    ax1.set_ylabel('Y-coordinate', fontsize=12)
    ax1.set_title(f'Interpolated Heads', fontsize=14)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    contour = ax2.contourf(X, Y, interpolated_heads2, levels=20, 
                          cmap='viridis', alpha=0.8)
    contour_lines = ax2.contour(X, Y, interpolated_heads2, levels=10, 
                                colors='black', alpha=0.4, linewidths=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8)
    
    cbar2 = plt.colorbar(contour, ax=ax2, label='Groundwater Head (m)')
    ax2.set_xlabel('X-coordinate', fontsize=12)
    ax2.set_ylabel('Y-coordinate', fontsize=12)
    ax2.set_title(f'Interpolated Heads', fontsize=14)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    file = base_path / 'figures' / 'initial_conditions' / f'{fig_name}.png'
    fig.savefig(file, dpi=300, bbox_inches='tight')
    plt.close(fig)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    depths_array1 = np.where(topography - interpolated_heads1 > 25, 25, topography - interpolated_heads1)
    contour = ax1.contourf(X, Y, depths_array1, levels=10, 
                          cmap='viridis', alpha=0.8, vmin=0, vmax=25)
    contour_lines = ax1.contour(X, Y, depths_array1, levels=10, 
                                colors='black', alpha=0.4, linewidths=0.5)
    ax1.clabel(contour_lines, inline=True, fontsize=8)
    
    cbar1 = plt.colorbar(contour, ax=ax1, label='Groundwater depth (m)')
    ax1.set_xlabel('X-coordinate', fontsize=12)
    ax1.set_ylabel('Y-coordinate', fontsize=12)
    ax1.set_title(f'Interpolated Depths', fontsize=14)
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    depths_array2 = np.where(topography - interpolated_heads2 > 25, 25, topography - interpolated_heads2)
    contour = ax2.contourf(X, Y, depths_array2, levels=10, 
                          cmap='viridis', alpha=0.8, vmin=0, vmax=25)
    contour_lines = ax2.contour(X, Y, depths_array2, levels=10, 
                                colors='black', alpha=0.4, linewidths=0.5)
    ax2.clabel(contour_lines, inline=True, fontsize=8)
    
    cbar2 = plt.colorbar(contour, ax=ax2, label='Groundwater depth (m)')
    ax2.set_xlabel('X-coordinate', fontsize=12)
    ax2.set_ylabel('Y-coordinate', fontsize=12)
    ax2.set_title(f'Interpolated Depths', fontsize=14)
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    file = base_path / 'figures' / 'initial_conditions' / f'{fig_name}_depth.png'
    fig.savefig(file, dpi=300, bbox_inches='tight')
    plt.close(fig)


base_path = Path(__file__).parent

# load MODFLOW parameters
path = base_path / "input" / "parameters_modflow.nc"
ds_params = xr.open_dataset(path, engine="h5netcdf")
spatial_ref = ds_params.spatial_ref
xcoords = ds_params.x.values
ycoords = ds_params.y.values
topography = ds_params['topography'].values
mask = np.isfinite(topography)
mask_porous = ds_params['mask_porous_aquifer'].values == 1
mask_fissured = (ds_params['mask_black_forest'].values == 1) & np.isfinite(topography)
grid_extent = (xcoords[0], xcoords[-1], ycoords[0], ycoords[-1])

# load locations of observation wells
file = base_path / "observations" / "groundwater_observation_wells.gpkg"
groundwater_observation_wells = gpd.read_file(file)
# remove wells outside grid_extent
groundwater_observation_wells = groundwater_observation_wells.cx[grid_extent[0]:grid_extent[1], grid_extent[2]:grid_extent[3]]

# load groundwater heads time series
file = base_path / "observations" / "groundwater_head_time_series_filled.csv"
_df_gw_heads = pd.read_csv(file, sep=";", index_col=0)
_df_gw_heads.index = pd.to_datetime(_df_gw_heads.index, format="%Y-%m-%d")
date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
df_gw_heads = pd.DataFrame(index=date_time)
df_gw_heads = df_gw_heads.join(_df_gw_heads, how="left")

# get number of missing values per row
missing_values_per_row = df_gw_heads.isnull().sum(axis=1)
# get row with lowest missing values
min_missing_values_row = missing_values_per_row.idxmin()
# print date with lowest missing values and number of missing values
print(f"Date with lowest missing values: {min_missing_values_row}, Number of missing values: {missing_values_per_row[min_missing_values_row]}")

df_gw_heads_initial = df_gw_heads.loc["2013-01-01"].to_frame()
df_gw_heads_initial.columns = ['groundwater_head']

path = base_path / "observations" / "groundwater_observation_wells.gpkg"
_gdf_gw_heads_initial = gpd.read_file(path)
# remove wells outside grid_extent
gdf_gw_heads_initial = _gdf_gw_heads_initial.cx[grid_extent[0]:grid_extent[1], grid_extent[2]:grid_extent[3]]
gdf_gw_heads_initial["station_id"] = gdf_gw_heads_initial["station_id"].str.replace('/', '_')
gdf_gw_heads_initial.index = gdf_gw_heads_initial["station_id"]
# merge with groundwater heads
gdf_gw_heads_initial = gdf_gw_heads_initial.join(df_gw_heads_initial, how='left')

# load interpolated groundwater heads
src = rasterio.open(str(base_path / "input" / "groundwater_heads_interpolated_50m.tif"))
gw_heads_interpolated = src.read(1)

df_diff = pd.DataFrame(index=gdf_gw_heads_initial.index)
df_diff['difference'] = np.nan
df_diff['observed_head'] = gdf_gw_heads_initial['groundwater_head']
df_diff['interpolated_head'] = np.nan
for idx, row in gdf_gw_heads_initial.iterrows():
    x = row.geometry.x
    y = row.geometry.y
    row, col = src.index(x, y)
    head_interpolated = gw_heads_interpolated[row, col]
    head_observed = gdf_gw_heads_initial.loc[idx, 'groundwater_head']
    diff = head_observed - head_interpolated
    df_diff.loc[idx, 'difference'] = diff
    df_diff.loc[idx, 'interpolated_head'] = head_interpolated
# linear regression to predict difference using interpolated heads
cond = np.isfinite(df_diff['observed_head'].values)
slope, intercept, r_value, p_value, std_err = sp.stats.linregress(
    df_diff.loc[cond, 'interpolated_head'].values,
    df_diff.loc[cond, 'observed_head'].values
)
# predict difference
df_diff['predicted_observed_head'] = intercept + slope * df_diff['interpolated_head'].values
# fill data gaps in difference using predicted difference
cond = np.isnan(df_diff['observed_head'].values)
df_diff.loc[cond, 'difference'] = df_diff.loc[cond, 'predicted_observed_head'] - df_diff.loc[cond, 'interpolated_head']
df_diff.loc[cond, 'observed_head'] = df_diff.loc[cond, 'predicted_observed_head']
df_diff.loc['avg', 'difference'] = df_diff['difference'].mean()
df_diff.loc['std', 'difference'] = df_diff['difference'].std()
# save to csv
file = base_path / 'figures' / 'initial_conditions' / 'interpolation_difference_observed_interpolated_heads.csv'
df_diff.to_csv(file, sep=";", index=True)


"""Main execution function"""
print("=" * 70)
print("Spatial Interpolation of Groundwater Heads using GSTools")
print("=" * 70)

# remove rows with NaN groundwater heads
gdf_gw_heads_initial = gdf_gw_heads_initial.dropna(subset=['groundwater_head'])
x_porous = gdf_gw_heads_initial.geometry.x.values
y_porous = gdf_gw_heads_initial.geometry.y.values
heads_porous = gdf_gw_heads_initial['groundwater_head'].values
depths_porous = np.zeros_like(x_porous)
for i in range(len(x_porous)):
    row, col = src.index(x_porous[i], y_porous[i])
    depths_porous[i] = topography[row, col] - heads_porous[i]

# Plot measurement points
plot_measurement_points(x_porous, y_porous, heads_porous, fig_name='measurement_points_porous')

# Step 2: Variogram analysis
print("\n2. Performing variogram analysis...")
model = analyze_variogram(x_porous, y_porous, heads_porous, fig_name='variogram_analysis_porous')

# Step 3: Ordinary Kriging
print("\n3. Performing Ordinary Kriging interpolation...")
grid_x, grid_y, ok_heads, ok_variance, ok_krige = ordinary_kriging_interpolation(
    x_porous, y_porous, heads_porous, model, grid_extent=grid_extent
)
ok_heads = np.where(ok_heads > topography, topography, ok_heads)
plot_interpolation_results(grid_x, grid_y, ok_heads, ok_variance, topography,
                            x_porous, y_porous, heads_porous, depths_porous, mask_porous, method='Ordinary', fig_name='kriging_results_porous')

# Step 4: Universal Kriging
print("\n4. Performing Universal Kriging interpolation...")
grid_x, grid_y, idw_heads, idw_variance = idw_interpolation(
    x_porous, y_porous, heads_porous, grid_extent=grid_extent
)
idw_heads = np.where(idw_heads > topography, topography, idw_heads)
plot_interpolation_results(grid_x, grid_y, idw_heads, idw_variance, topography,
                            x_porous, y_porous, heads_porous, depths_porous, mask_porous, method='IDW', fig_name='results_porous')

grid_x, grid_y, uk_heads, uk_variance, uk_krige = universal_kriging_interpolation(
    x_porous, y_porous, heads_porous, model, grid_extent=grid_extent, use_idw_trend=True, idw_power=2
)
uk_heads = np.where(uk_heads > topography, topography, uk_heads)
# get kriged heads at observation points
heads_porous_kriged = np.zeros_like(x_porous)
for i in range(len(x_porous)):
    row, col = src.index(x_porous[i], y_porous[i])
    head = uk_heads[row, col]
    heads_porous_kriged[i] = head

interpolated_heads_porous = uk_heads.copy()
plot_interpolation_results(grid_x, grid_y, uk_heads, uk_variance, topography,
                            x_porous, y_porous, heads_porous, depths_porous, mask_porous, method='Universal', fig_name='kriging_results_porous')

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(heads_porous, heads_porous_kriged, c='black', s=20)
ax.set_xlim(np.floor(np.min([heads_porous.min() - 5, heads_porous_kriged.min() - 5])), np.ceil(np.max([heads_porous.max() + 5, heads_porous_kriged.max() + 5])))
ax.set_ylim(np.floor(np.min([heads_porous.min() - 5, heads_porous_kriged.min() - 5])), np.ceil(np.max([heads_porous.max() + 5, heads_porous_kriged.max() + 5])))
# make 1:1 line
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_xlim()[0], ax.get_xlim()[1]], 'k--', lw=2)
ax.set_xlabel('Observed Groundwater Head (m)')
ax.set_ylabel('Kriged Groundwater Head (m)')
plt.grid(True)
fig.tight_layout()
file = base_path / 'figures' / 'initial_conditions' / f'comparison_observed_kriged_heads_porous_initial.png'
fig.savefig(file, dpi=250, bbox_inches='tight')

# calculate RMSE
rmse_porous = np.sqrt(np.mean((heads_porous - heads_porous_kriged) ** 2))
print(f'RMSE (empirical): {rmse_porous:.2f}')

# Step 5: Conditional Random Fields (using ordinary kriging instance)
print("\n5. Generating conditional random field realizations...")
grid_x, grid_y, realizations = generate_conditional_random_field(
    ok_krige, grid_resolution=50, n_realizations=3, grid_extent=grid_extent
)
realizations[0] = np.where(realizations[0] > topography, topography, realizations[0])
realizations[1] = np.where(realizations[1] > topography, topography, realizations[1])
realizations[2] = np.where(realizations[2] > topography, topography, realizations[2])
plot_conditional_realizations(grid_x, grid_y, realizations, topography, x_porous, y_porous, heads_porous, depths_porous, mask_porous, fig_name='conditional_realizations_porous')

gw_heads_interpolated_fissured = np.where(mask_fissured, gw_heads_interpolated, np.nan)
# get extent of finite values
rows, cols = np.where(np.isfinite(gw_heads_interpolated_fissured))
min_row, max_row = rows.min(), rows.max()
min_col, max_col = cols.min(), cols.max()

x_ = np.random.uniform(xcoords[min_col], xcoords[max_col], 400)
y_ = np.random.uniform(ycoords[min_row], ycoords[max_row], 400)
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
heads_fissured = np.zeros_like(x)
for i in range(len(x_fissured)):
    row, col = src.index(x_fissured[i], y_fissured[i])
    heads_fissured[i] = gw_heads_interpolated_fissured[row, col] + (df_diff['difference'].mean() * 50)
depths_fissured = np.zeros_like(x_fissured)
for i in range(len(x_fissured)):
    row, col = src.index(x_fissured[i], y_fissured[i])
    depths_fissured[i] = topography[row, col] - heads_fissured[i]

print("=" * 70)
print("Spatial Interpolation of Groundwater Heads using GSTools")
print("=" * 70)

plot_measurement_points(x_fissured, y_fissured, heads_fissured, fig_name='measurement_points_fissured')

# Step 2: Variogram analysis
print("\n2. Performing variogram analysis...")
model = analyze_variogram(x_fissured, y_fissured, heads_fissured, fig_name='variogram_analysis_fissured')
# Step 3: Ordinary Kriging
print("\n3. Performing Ordinary Kriging interpolation...")
grid_x, grid_y, ok_heads, ok_variance, ok_krige = ordinary_kriging_interpolation(
    x_fissured, y_fissured, heads_fissured, model, grid_extent=grid_extent
)
ok_heads = np.where(ok_heads > topography, topography, ok_heads)
plot_interpolation_results(grid_x, grid_y, ok_heads, ok_variance, topography,
                            x_fissured, y_fissured, heads_fissured, depths_fissured, mask_fissured, method='Ordinary', fig_name='kriging_results_fissured')

# Step 4: Universal Kriging
print("\n4. Performing Universal Kriging interpolation...")
grid_x, grid_y, idw_heads, idw_variance = idw_interpolation(
    x_fissured, y_fissured, heads_fissured, grid_extent=grid_extent
)
idw_heads = np.where(idw_heads > topography, topography, idw_heads)
plot_interpolation_results(grid_x, grid_y, idw_heads, idw_variance, topography,
                            x_fissured, y_fissured, heads_fissured, depths_fissured, mask_fissured, method='IDW', fig_name='results_fissured')
grid_x, grid_y, uk_heads, uk_variance, uk_krige = universal_kriging_interpolation(
    x_fissured, y_fissured, heads_fissured, model, grid_extent=grid_extent, use_idw_trend=True, idw_power=2
)
uk_heads = np.where(uk_heads > topography, topography, uk_heads)
# get kriged heads at observation points
heads_fissured_kriged = np.zeros_like(x_fissured)
for i in range(len(x_fissured)):
    row, col = src.index(x_fissured[i], y_fissured[i])
    head = uk_heads[row, col]
    heads_fissured_kriged[i] = head
plot_interpolation_results(grid_x, grid_y, uk_heads, uk_variance, topography,
                            x_fissured, y_fissured, heads_fissured, depths_fissured, mask_fissured, method='Universal', fig_name='kriging_results_fissured')

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(heads_fissured, heads_fissured_kriged, c='black', s=20)
ax.set_xlim(np.floor(np.min([heads_fissured.min() - 5, heads_fissured_kriged.min() - 5])), np.ceil(np.max([heads_fissured.max() + 5, heads_fissured_kriged.max() + 5])))
ax.set_ylim(np.floor(np.min([heads_fissured.min() - 5, heads_fissured_kriged.min() - 5])), np.ceil(np.max([heads_fissured.max() + 5, heads_fissured_kriged.max() + 5])))
# make 1:1 line
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_xlim()[0], ax.get_xlim()[1]], 'k--', lw=2)
ax.set_xlabel('Observed Groundwater Head (m)')
ax.set_ylabel('Kriged Groundwater Head (m)')
plt.grid(True)
fig.tight_layout()
file = base_path / 'figures' / 'initial_conditions' / f'comparison_observed_kriged_heads_fissured_initial.png'
fig.savefig(file, dpi=250, bbox_inches='tight')

# calculate RMSE
rmse_fissured = np.sqrt(np.mean((heads_fissured - heads_fissured_kriged) ** 2))
print(f'RMSE (empirical): {rmse_fissured:.2f}')

# Step 5: Conditional Random Fields (using ordinary kriging instance)
print("\n5. Generating conditional random field realizations...")
grid_x, grid_y, realizations = generate_conditional_random_field(
    ok_krige, grid_resolution=50, n_realizations=3, grid_extent=grid_extent
)
realizations[0] = np.where(realizations[0] > topography, topography, realizations[0])
realizations[1] = np.where(realizations[1] > topography, topography, realizations[1])
realizations[2] = np.where(realizations[2] > topography, topography, realizations[2])
plot_conditional_realizations(grid_x, grid_y, realizations, topography, x_fissured, y_fissured, heads_fissured, depths_fissured, mask_fissured, fig_name='conditional_realizations_fissured')

x_combined = np.concatenate([x_porous, x_fissured])
y_combined = np.concatenate([y_porous, y_fissured])
heads_combined = np.concatenate([heads_porous, heads_fissured])
depths_combined = np.concatenate([depths_porous, depths_fissured])

print("=" * 70)
print("Spatial Interpolation of Groundwater Heads using GSTools")
print("=" * 70)

plot_measurement_points(x_combined, y_combined, heads_combined, fig_name='measurement_points_combined')

# Step 2: Variogram analysis
print("\n2. Performing variogram analysis...")
model = analyze_variogram(x_combined, y_combined, heads_combined, fig_name='variogram_analysis_combined')
# Step 3: Ordinary Kriging
print("\n3. Performing Ordinary Kriging interpolation...")
grid_x, grid_y, ok_heads, ok_variance, ok_krige = ordinary_kriging_interpolation(
    x_combined, y_combined, heads_combined, model, grid_extent=grid_extent
)
ok_heads = np.where(ok_heads > topography, topography, ok_heads)
plot_interpolation_results(grid_x, grid_y, ok_heads, ok_variance, topography,
                           x_combined, y_combined, heads_combined, depths_combined, mask, method='Ordinary', fig_name='kriging_results_combined')

# Step 4: Universal Kriging
print("\n4. Performing Universal Kriging interpolation...")
grid_x, grid_y, idw_heads, idw_variance = idw_interpolation(
    x_combined, y_combined, heads_combined, grid_extent=grid_extent
)
idw_heads = np.where(idw_heads > topography, topography, idw_heads) 
plot_interpolation_results(grid_x, grid_y, idw_heads, idw_variance, topography,
                            x_combined, y_combined, heads_combined, depths_combined, mask, method='IDW', fig_name='results_combined')
grid_x, grid_y, uk_heads, uk_variance, uk_krige = universal_kriging_interpolation(
    x_combined, y_combined, heads_combined, model, grid_extent=grid_extent, use_idw_trend=True, idw_power=2
)
uk_heads = np.where(uk_heads > topography, topography, uk_heads)
# get kriged heads at observation points
heads_combined_kriged = np.zeros_like(x_combined)
for i in range(len(x_combined)):
    row, col = src.index(x_combined[i], y_combined[i])
    head = uk_heads[row, col]
    heads_combined_kriged[i] = head
plot_interpolation_results(grid_x, grid_y, uk_heads, uk_variance, topography,
                           x_combined, y_combined, heads_combined, depths_combined, mask, method='Universal', fig_name='kriging_results_combined')

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(heads_combined, heads_combined_kriged, c='black', s=20)
ax.set_xlim(np.floor(np.min([heads_combined.min() - 5, heads_combined_kriged.min() - 5])), np.ceil(np.max([heads_combined.max() + 5, heads_combined_kriged.max() + 5])))
ax.set_ylim(np.floor(np.min([heads_combined.min() - 5, heads_combined_kriged.min() - 5])), np.ceil(np.max([heads_combined.max() + 5, heads_combined_kriged.max() + 5])))
# make 1:1 line
ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_xlim()[0], ax.get_xlim()[1]], 'k--', lw=2)
ax.set_xlabel('Observed Groundwater Head (m)')
ax.set_ylabel('Kriged Groundwater Head (m)')
plt.grid(True)
fig.tight_layout()
file = base_path / 'figures' / 'initial_conditions' / f'comparison_observed_kriged_heads_combined_initial.png'
fig.savefig(file, dpi=250, bbox_inches='tight')

# calculate RMSE
rmse_combined = np.sqrt(np.mean((heads_combined - heads_combined_kriged) ** 2))
print(f'RMSE (empirical): {rmse_combined:.2f}')

# Step 5: Conditional Random Fields (using ordinary kriging instance)
print("\n5. Generating conditional random field realizations...")
grid_x, grid_y, realizations = generate_conditional_random_field(
    ok_krige, grid_resolution=50, n_realizations=3, grid_extent=grid_extent
)
realizations[0] = np.where(realizations[0] > topography, topography, realizations[0])
realizations[1] = np.where(realizations[1] > topography, topography, realizations[1])
realizations[2] = np.where(realizations[2] > topography, topography, realizations[2])
plot_conditional_realizations(grid_x, grid_y, realizations, topography, x_combined, y_combined, heads_combined, depths_combined, mask, fig_name='conditional_realizations_combined')

plot_comparison(grid_x, grid_y, uk_heads, gw_heads_interpolated + df_diff['difference'].mean(), topography, mask)

ds = xr.Dataset()
ds["spatial_ref"] = spatial_ref
file = base_path / "input" / "initial_conditions.nc"
ds.to_netcdf(file, engine="h5netcdf")
ds.close()

file = base_path / "input" / "initial_conditions.nc"
with h5netcdf.File(file, "a", decode_vlen_strings=False) as f:
    f.attrs.update(
    date_created=datetime.datetime.today().isoformat(),
    title="Initial conditions of the Dreisam-Möhlin-Neumagen catchment",
    institution="University of Freiburg, Chair of Hydrology",
    references="",
    comment="",
    spatial_ref="EPSG:25832",
    x_origin=396331.5,
    y_origin=5325918.5,
    )
    dict_dim = {"y": len(ds_params['y'].values), "x": len(ds_params['x'].values), 'scalar': 1}
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

    v = f.create_variable(
        "initial_head", ("y", "x"), np.float64, compression="gzip", compression_opts=1
    )
    v[:, :] = uk_heads[:, :]
    v.attrs.update(long_name="Interpolated heads", units=" ", grid_mapping="spatial_ref", coordinates="spatial_ref")

    v = f.create_variable(
        "initial_head_porous", ("y", "x"), np.float64, compression="gzip", compression_opts=1
    )
    v[:, :] = interpolated_heads_porous[:, :]
    v.attrs.update(long_name="Interpolated heads of the porous aquifer", units=" ", grid_mapping="spatial_ref", coordinates="spatial_ref")