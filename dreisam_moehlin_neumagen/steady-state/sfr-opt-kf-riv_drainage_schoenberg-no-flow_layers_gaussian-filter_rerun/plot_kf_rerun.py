from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import scipy as sp
import rasterio
import matplotlib.pyplot as plt
import click
import yaml

def recalc_specific_yield(hydraulic_conductivity, specific_yield_min=0.05, specific_yield_max=0.35):
    """Recalculate specific yield based on hydraulic conductivity using the formula of Marotz (1968)

    Args:
        hydraulic_conductivity (numpy.ndarray): hydraulic conductivity in m/day
        specific_yield_min (float, optional): Constraint of specific yield. Default is 0.05.

    Returns:
        numpy.ndarray: specific yield
    """
    specific_yield = 0.462 + 0.045 * np.log(hydraulic_conductivity/86400)
    specific_yield[specific_yield < specific_yield_min] = specific_yield_min
    specific_yield[specific_yield > specific_yield_max] = specific_yield_max
    return specific_yield

@click.option("-mr", "--model-run", type=int, default=5)
@click.command("main", short_help="Evaluate the steady-state simulation")
def main(model_run):
    base_path = Path(__file__).parent

    # load MODFLOW parameters
    path = Path(__file__).parent.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    grid_extent = (ds_params.x.values[0] / 1000, ds_params.x.values[-1] / 1000, ds_params.y.values[-1] / 1000, ds_params.y.values[0] / 1000)

    path = base_path / "fudge_parameters_modflow.csv"
    fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)
    
    # load the simulated groundwater heads
    output_file = base_path / "output" / f"modflow_output_run_{model_run}.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
    groundwater_heads = ds_mf["head"].values[0, 1, ...]

    # load the config file
    file_config = base_path.parent / "config.yml"
    with open(file_config, "r") as file:
        modflow_config = yaml.safe_load(file)

    # load the topography and elevation of the aquifer layers
    topography = ds_params['elevations'].isel(z=0).values
    # derive the model domain from the topography
    mask = np.isfinite(topography)
    # set Schoenberg to inactive
    mask_schoenberg = (ds_params['mask_schoenberg'].values == 1)
    mask = np.where(mask_schoenberg, False, mask)

    elevation_bottom_layer1 = ds_params['elevations'].isel(z=1).values
    elevation_bottom_layer2 = ds_params['elevations'].isel(z=2).values
    elevation_bottom_layer3 = ds_params['elevations'].isel(z=3).values
    elevation_bottom_layer4 = ds_params['elevations'].isel(z=4).values
    elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

    topography = ds_params['elevations'].isel(z=0).values
    elevation_bottom_layer1 = ds_params['elevations'].isel(z=1).values
    elevation_bottom_layer2 = ds_params['elevations'].isel(z=2).values
    elevation_bottom_layer3 = ds_params['elevations'].isel(z=3).values
    elevation_bottom_layer4 = ds_params['elevations'].isel(z=4).values
    elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]
    elevation_layers = [topography, elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

    mask_ = np.isfinite(topography)
    mask_schoenberg = ds_params['mask_schoenberg'].values
    mask = mask_ & (mask_schoenberg == False)
    domain = np.empty_like(topography)
    domain[mask] = 1
    domain[~mask] = -1

    hydraulic_conductivities_layer1 = ds_params['kf'].isel(layer=0).values
    hydraulic_conductivities_layer2 = ds_params['kf'].isel(layer=1).values
    hydraulic_conductivities_layer3 = ds_params['kf'].isel(layer=2).values
    hydraulic_conductivities_layer4 = ds_params['kf'].isel(layer=3).values


    hydraulic_conductivities_layer1_ = ds_params['kf'].isel(layer=0).values / 86400
    hydraulic_conductivities_layer2_ = ds_params['kf'].isel(layer=1).values / 86400
    hydraulic_conductivities_layer3_ = ds_params['kf'].isel(layer=2).values / 86400
    hydraulic_conductivities_layer4_ = ds_params['kf'].isel(layer=3).values / 86400
    
    # fudge parameters
    mask1 = (hydraulic_conductivities_layer1_ <= 10e-10)
    mask2 = (hydraulic_conductivities_layer2_ <= 10e-10)
    mask3 = (hydraulic_conductivities_layer3_ <= 10e-10)
    mask4 = (hydraulic_conductivities_layer4_ <= 10e-10)  
    hydraulic_conductivities_layer1[mask1] = hydraulic_conductivities_layer1[mask1] * 10000
    hydraulic_conductivities_layer2[mask2] = hydraulic_conductivities_layer2[mask2] * 10000
    hydraulic_conductivities_layer3[mask3] = hydraulic_conductivities_layer3[mask3] * 10000
    hydraulic_conductivities_layer4[mask4] = hydraulic_conductivities_layer4[mask4] * 10000

    mask_ = (hydraulic_conductivities_layer2_ == 1.9999999e-07)
    hydraulic_conductivities_layer2_ = np.where(mask_, 1.9722222e-07, hydraulic_conductivities_layer2_)
    mask_ = (hydraulic_conductivities_layer3_ == 1.9999999e-07)
    hydraulic_conductivities_layer3_ = np.where(mask_, 1.9722222e-07, hydraulic_conductivities_layer3_)
    mask_ = (hydraulic_conductivities_layer4_ == 1.9999999e-07)
    hydraulic_conductivities_layer4_ = np.where(mask_, 1.9722222e-07, hydraulic_conductivities_layer4_)

    mask81 = (hydraulic_conductivities_layer1_ == 1.1574075e-08) | (hydraulic_conductivities_layer1_ == 2.7777778e-08)

    mask71 = (hydraulic_conductivities_layer1_ == 1.9444444e-07) | (hydraulic_conductivities_layer1_ == 1.9722222e-07) | (hydraulic_conductivities_layer1_ == 2.3055554e-07) | (hydraulic_conductivities_layer1_ == 5.7777777e-07)
    mask72 = (hydraulic_conductivities_layer2_ == 1.9722222e-07)
    mask73 = (hydraulic_conductivities_layer3_ == 1.9722222e-07)
    mask74 = (hydraulic_conductivities_layer4_ == 1.9722222e-07)

    mask61 = (hydraulic_conductivities_layer1_ >= 1.1583334e-06) & (hydraulic_conductivities_layer1_ <= 8.1027783e-06)

    mask51 = (hydraulic_conductivities_layer1_ == 1.1575000e-05) | (hydraulic_conductivities_layer1_ == 1.8181944e-04)
    mask52 = (hydraulic_conductivities_layer2_ == 1.8180555e-05)
    mask53 = (hydraulic_conductivities_layer3_ == 1.8180555e-05)
    mask54 = (hydraulic_conductivities_layer4_ == 1.8180555e-05)

    mask42 = (hydraulic_conductivities_layer2_ == 1.8181944e-04)
    mask43 = (hydraulic_conductivities_layer3_ == 1.8181944e-04)
    mask44 = (hydraulic_conductivities_layer4_ == 1.8181944e-04)

    mask132 = (hydraulic_conductivities_layer2_ == 1.0000000e-03)
    mask133 = (hydraulic_conductivities_layer3_ == 1.0000000e-03)

    mask232 = (hydraulic_conductivities_layer2_ == 1.8181807e-03)
    mask233 = (hydraulic_conductivities_layer3_ == 1.8181807e-03)
    mask234 = (hydraulic_conductivities_layer4_ == 1.8181807e-03)

    mask332 = (hydraulic_conductivities_layer2_ == 3.0000000e-03)
    mask333 = (hydraulic_conductivities_layer3_ == 3.0000000e-03)

    mask432 = (hydraulic_conductivities_layer2_ == 4.0000002e-03)
    mask433 = (hydraulic_conductivities_layer3_ == 4.0000002e-03)

    # fudge parameters
    hydraulic_conductivities_layer1[mask81] = hydraulic_conductivities_layer1[mask81] * fudge_parameters['-8_1'].values[model_run]

    hydraulic_conductivities_layer1[mask71] = hydraulic_conductivities_layer1[mask71] * fudge_parameters['-7_1'].values[model_run]
    hydraulic_conductivities_layer2[mask72] = hydraulic_conductivities_layer2[mask72] * fudge_parameters['-7_2'].values[model_run]
    hydraulic_conductivities_layer3[mask73] = hydraulic_conductivities_layer3[mask73] * fudge_parameters['-7_3'].values[model_run]
    hydraulic_conductivities_layer4[mask74] = hydraulic_conductivities_layer4[mask74] * fudge_parameters['-7_4'].values[model_run]

    hydraulic_conductivities_layer1[mask61] = hydraulic_conductivities_layer1[mask61] * fudge_parameters['-6_1'].values[model_run]

    hydraulic_conductivities_layer1[mask51] = hydraulic_conductivities_layer1[mask51] * fudge_parameters['-5_1'].values[model_run]
    hydraulic_conductivities_layer2[mask52] = hydraulic_conductivities_layer2[mask52] * fudge_parameters['-5_2'].values[model_run]
    hydraulic_conductivities_layer3[mask53] = hydraulic_conductivities_layer3[mask53] * fudge_parameters['-5_3'].values[model_run]
    hydraulic_conductivities_layer4[mask54] = hydraulic_conductivities_layer4[mask54] * fudge_parameters['-5_4'].values[model_run]

    hydraulic_conductivities_layer2[mask42] = hydraulic_conductivities_layer2[mask42] * fudge_parameters['-4_2'].values[model_run]
    hydraulic_conductivities_layer3[mask43] = hydraulic_conductivities_layer3[mask43] * fudge_parameters['-4_3'].values[model_run]
    hydraulic_conductivities_layer4[mask44] = hydraulic_conductivities_layer4[mask44] * fudge_parameters['-4_4'].values[model_run]

    hydraulic_conductivities_layer2[mask132] = hydraulic_conductivities_layer2[mask132] * fudge_parameters['1-3_2'].values[model_run]
    hydraulic_conductivities_layer3[mask133] = hydraulic_conductivities_layer3[mask133] * fudge_parameters['1-3_3'].values[model_run]

    hydraulic_conductivities_layer2[mask232] = hydraulic_conductivities_layer2[mask232] * fudge_parameters['1.8-3_2'].values[model_run]
    hydraulic_conductivities_layer3[mask233] = hydraulic_conductivities_layer3[mask233] * fudge_parameters['1.8-3_3'].values[model_run]
    hydraulic_conductivities_layer4[mask234] = hydraulic_conductivities_layer4[mask234] * fudge_parameters['1.8-3_4'].values[model_run]

    hydraulic_conductivities_layer2[mask332] = hydraulic_conductivities_layer2[mask332] * fudge_parameters['3-3_2'].values[model_run]
    hydraulic_conductivities_layer3[mask333] = hydraulic_conductivities_layer3[mask333] * fudge_parameters['3-3_3'].values[model_run]

    hydraulic_conductivities_layer2[mask432] = hydraulic_conductivities_layer2[mask432] * fudge_parameters['4-3_2'].values[model_run]
    hydraulic_conductivities_layer3[mask433] = hydraulic_conductivities_layer3[mask433] * fudge_parameters['4-3_3'].values[model_run]

    # adjust hydraulic conductivities in the fissured aquifer if the groundwater table is above the surface 
    gw_depth = topography - ds_mf['head'].isel(Time=0, layer=3).values
    cond2 = (gw_depth < 0) & (hydraulic_conductivities_layer2_ <= 10.0e-07)
    cond3 = (gw_depth < 0) & (hydraulic_conductivities_layer3_ <= 10.0e-07)
    cond4 = (gw_depth < 0) & (hydraulic_conductivities_layer4_ <= 10.0e-07)
    hydraulic_conductivities_layer2[cond2] = hydraulic_conductivities_layer2[cond2] * 50
    hydraulic_conductivities_layer3[cond3] = hydraulic_conductivities_layer3[cond3] * 50
    hydraulic_conductivities_layer4[cond4] = hydraulic_conductivities_layer4[cond4] * 50

    # smooth transition between fissured and porous aquifers
    hydraulic_conductivities_layer1[np.isnan(hydraulic_conductivities_layer1)] = 0
    hydraulic_conductivities_layer2[np.isnan(hydraulic_conductivities_layer2)] = 0
    hydraulic_conductivities_layer3[np.isnan(hydraulic_conductivities_layer3)] = 0
    hydraulic_conductivities_layer4[np.isnan(hydraulic_conductivities_layer4)] = 0
    _hydraulic_conductivities_layer1 = sp.ndimage.gaussian_filter(hydraulic_conductivities_layer1, [1.5, 1.5], mode="constant")
    _hydraulic_conductivities_layer2 = sp.ndimage.gaussian_filter(hydraulic_conductivities_layer2, [1.5, 1.5], mode="constant")
    _hydraulic_conductivities_layer3 = sp.ndimage.gaussian_filter(hydraulic_conductivities_layer3, [1.5, 1.5], mode="constant")
    _hydraulic_conductivities_layer4 = sp.ndimage.gaussian_filter(hydraulic_conductivities_layer4, [1.5, 1.5], mode="constant")
    cond1 = (hydraulic_conductivities_layer1_ < 10.0e-07)
    cond2 = (hydraulic_conductivities_layer2_ < 10.0e-07)
    cond3 = (hydraulic_conductivities_layer3_ < 10.0e-07)
    cond4 = (hydraulic_conductivities_layer4_ < 10.0e-07)
    hydraulic_conductivities_layer1[cond1] = _hydraulic_conductivities_layer1[cond1]
    hydraulic_conductivities_layer2[cond2] = _hydraulic_conductivities_layer2[cond2]
    hydraulic_conductivities_layer3[cond3] = _hydraulic_conductivities_layer3[cond3]
    hydraulic_conductivities_layer4[cond4] = _hydraulic_conductivities_layer4[cond4]
    hydraulic_conductivities_layer1[~mask] = np.nan
    hydraulic_conductivities_layer2[~mask] = np.nan
    hydraulic_conductivities_layer3[~mask] = np.nan
    hydraulic_conductivities_layer4[~mask] = np.nan


    hydraulic_conductivities = np.array([hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4])

    hydraulic_conductivities_layers = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]
    for i, hydraulic_conductivities_layer in enumerate(hydraulic_conductivities_layers):
        fig, axes = plt.subplots(figsize=(4, 4))
        hydraulic_conductivities_layer[~mask] = np.nan
        plt.imshow(hydraulic_conductivities_layer/(24*60*60), extent=grid_extent, cmap='Oranges', aspect='equal', vmin=10e-7, vmax=10e-2, norm='log')
        cbar = plt.colorbar(label='$k_f$ [m/s]', shrink=0.48)
        plt.grid(zorder=0)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.tight_layout()
        file = Path(__file__).parent / "figures" / f"hydraulic_conductivity_layer_{i}_{model_run}_rerun.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)
    return

if __name__ == "__main__":
    main()
