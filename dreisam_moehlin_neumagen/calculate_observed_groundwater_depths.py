from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import click

@click.command("main", short_help="Evaluate the steady-state simulation")
def main():
    base_path = Path(__file__).parent

    # load MODFLOW parameters
    path = Path(__file__).parent / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")

    # load the topography and elevation of the aquifer layers
    topography = ds_params['elevations'].isel(z=0).values
    # derive the model domain from the topography
    mask = np.isfinite(topography)

    # load observed groundwater heads (average values of the observation wells)
    path = base_path / "observations" / "observed_groundwater_heads_avg_std.csv"
    observed_groundwater_heads = pd.read_csv(path, sep=";", skiprows=0)

    # load observed groundwater heads
    obs = observed_groundwater_heads.iloc[:, -2].values  # observed groundwater heads
    rows = observed_groundwater_heads.iloc[:, -3].values  # row IDs of the observation wells
    cols = observed_groundwater_heads.iloc[:, -4].values  # column IDs of the observation wells

    gw_depth = []
    for i in range(len(obs)):
        gw_depth.append(topography[int(rows[i]), int(cols[i])] - obs[i])

    df = pd.DataFrame()
    df['station_id'] = observed_groundwater_heads['Station ID']
    df['lat'] = observed_groundwater_heads['y-Koordinate']
    df['lon'] = observed_groundwater_heads['x-Koordinate']
    df['avg'] = gw_depth
    df['std'] = observed_groundwater_heads['Standard Deviation [m]']

    df.columns = [
        ["", "", "", "[m]", "[m]"],
        ["station_id", "lat", "lon", "avg", "std"],
    ]

    path = base_path / "observations" / "observed_groundwater_depths.csv"
    df.to_csv(path, sep=";", index=False)

    grid_extent = (ds_params.x.values[0], ds_params.x.values[-1], ds_params.y.values[-1], ds_params.y.values[0])
    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    # wells_rows = [266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]
    # wells_cols = [66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]
    # wells_lat = [ds_params.y.values[i] for i in wells_rows]
    # wells_lon = [ds_params.x.values[i] for i in wells_cols]
    # plt.scatter(wells_lon, wells_lat, marker='x', s=5, c='black')
    wells_obs_rows = observed_groundwater_heads.iloc[:, -3].values.tolist()  # row IDs of the observation wells
    wells_obs_cols = observed_groundwater_heads.iloc[:, -4].values.tolist()  # column IDs of the observation wells
    wells_obs_lat = [ds_params.y.values[i] for i in wells_obs_rows]
    wells_obs_lon = [ds_params.x.values[i] for i in wells_obs_cols]
    plt.scatter(wells_obs_lon, wells_obs_lat, c=gw_depth, s=5, cmap="viridis", vmin=0, vmax=20)
    plt.colorbar(label='groundwater depth [m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    axes.ticklabel_format(style='plain')
    axes.tick_params(axis='y', labelrotation=75)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate')
    plt.ylabel('y-coordinate')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / "observed_groundwater_depths.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    return

if __name__ == "__main__":
    main()
