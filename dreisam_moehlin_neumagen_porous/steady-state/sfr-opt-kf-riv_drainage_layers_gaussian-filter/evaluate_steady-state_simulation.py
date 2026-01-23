from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import scipy as sp
import rasterio
import matplotlib.pyplot as plt
import scipy
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

    path = base_path / "fudge_parameters_modflow.csv"
    fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)

    # load the SFR reaches
    reaches = pd.read_csv(base_path.parent / 'input' / 'sfr_packagedata_modified.csv', sep=';')
    reaches["rno"] = reaches["rno"] - 1
    reaches["k"] = reaches["k"] - 1
    reaches["i"] = reaches["i"] - 1
    reaches["j"] = reaches["j"] - 1
    
    # load the simulated groundwater heads
    output_file = base_path / "output" / f"modflow_output_run_{model_run}.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
    groundwater_heads = ds_mf["head"].values[0, 1, ...]
    gw_sw = np.nanmean(ds_mf['gw_sw'].isel(Time=0).values, axis=0) / 86400

    # load the SFR output file
    output_file = base_path / "output" / f"dmn_run_{model_run}_sfr.obs.csv"
    df_sfr_ = pd.read_csv(output_file, sep=",")

    # load the config file
    file_config = base_path.parent / "config.yml"
    with open(file_config, "r") as file:
        modflow_config = yaml.safe_load(file)

    # load the topography and elevation of the aquifer layers
    topography = ds_params['elevations'].isel(z=0).values
    mask = (ds_params["mask_porous_aquifer"].values == 1)
    # set Schoenberg to inactive
    mask_schoenberg = (ds_params['mask_schoenberg'].values == 1)
    mask = np.where(mask_schoenberg, False, mask)

    topography = ds_params['elevations'].isel(z=0).values
    elevation_bottom_layer1 = ds_params['elevations'].isel(z=1).values
    elevation_bottom_layer2 = ds_params['elevations'].isel(z=2).values
    elevation_bottom_layer3 = ds_params['elevations'].isel(z=3).values
    elevation_bottom_layer4 = ds_params['elevations'].isel(z=4).values

    topography[~mask] = np.nan
    elevation_bottom_layer1[~mask] = np.nan
    elevation_bottom_layer2[~mask] = np.nan
    elevation_bottom_layer3[~mask] = np.nan
    elevation_bottom_layer4[~mask] = np.nan

    elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

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

    # smooth transition between fissured and porous aquifers
    hydraulic_conductivities_layer1[np.isnan(hydraulic_conductivities_layer1)] = 0
    hydraulic_conductivities_layer2[np.isnan(hydraulic_conductivities_layer2)] = 0
    hydraulic_conductivities_layer3[np.isnan(hydraulic_conductivities_layer3)] = 0
    hydraulic_conductivities_layer4[np.isnan(hydraulic_conductivities_layer4)] = 0
    _hydraulic_conductivities_layer1 = sp.ndimage.gaussian_filter(hydraulic_conductivities_layer1, [1.5, 1.5],  mode="constant")
    _hydraulic_conductivities_layer2 = sp.ndimage.gaussian_filter(hydraulic_conductivities_layer2, [1.5, 1.5],  mode="constant")
    _hydraulic_conductivities_layer3 = sp.ndimage.gaussian_filter(hydraulic_conductivities_layer3, [1.5, 1.5],  mode="constant")
    _hydraulic_conductivities_layer4 = sp.ndimage.gaussian_filter(hydraulic_conductivities_layer4, [1.5, 1.5],  mode="constant")
    cond1 = (hydraulic_conductivities_layer1_ < 10.0e-07)
    cond2 = (hydraulic_conductivities_layer2_ < 10.0e-07)
    cond3 = (hydraulic_conductivities_layer3_ < 10.0e-07)
    cond4 = (hydraulic_conductivities_layer4_ < 10.0e-07)
    hydraulic_conductivities_layer1[cond1] = _hydraulic_conductivities_layer1[cond1]
    hydraulic_conductivities_layer2[cond2] = _hydraulic_conductivities_layer2[cond2]
    hydraulic_conductivities_layer3[cond3] = _hydraulic_conductivities_layer3[cond3]
    hydraulic_conductivities_layer4[cond4] = _hydraulic_conductivities_layer4[cond4]

    # increase the hydraulic conductivities of the reach cell by a factor of xx
    reaches["kf"] = np.nan
    c_fissured = 1  # factor to increase the hydraulic conductivity in fissured layers
    for rno, z, y, x in zip(reaches.loc[:, "rno"], reaches.loc[:, "k"], reaches.loc[:, "i"], reaches.loc[:, "j"]):
        if z == 0:
            kf_riv = hydraulic_conductivities_layer1[y, x] / 86400
            if kf_riv < 10e-6:
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured) / 86400
            else:
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run]
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run]) / 86400
        elif z == 1:
            kf_riv = hydraulic_conductivities_layer2[y, x] / 86400
            if kf_riv < 10e-6:
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer2[y, x] = hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured) / 86400
            else:  
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer2[y, x] = hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run]) / 86400
        elif z == 2:
            kf_riv = hydraulic_conductivities_layer3[y, x] / 86400
            if kf_riv < 10e-6:
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer2[y, x] = hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer3[y, x] = hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured) / 86400
            else:
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer2[y, x] = hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer3[y, x] = hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured) / 86400
        elif z == 3:
            kf_riv = hydraulic_conductivities_layer4[y, x] / 86400
            if kf_riv < 10e-6:
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run]
                hydraulic_conductivities_layer2[y, x] = hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run]
                hydraulic_conductivities_layer3[y, x] = hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run]
                hydraulic_conductivities_layer4[y, x] = hydraulic_conductivities_layer4[y, x] * fudge_parameters["kf_riv"].values[model_run]
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer4[y, x] * fudge_parameters["kf_riv"].values[model_run]) / 86400
            else:
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run]
                hydraulic_conductivities_layer2[y, x] = hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run]
                hydraulic_conductivities_layer3[y, x] = hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run]
                hydraulic_conductivities_layer4[y, x] = hydraulic_conductivities_layer4[y, x] * fudge_parameters["kf_riv"].values[model_run]
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer4[y, x] * fudge_parameters["kf_riv"].values[model_run]) / 86400

    hydraulic_conductivities_layer1[~mask] = np.nan
    hydraulic_conductivities_layer2[~mask] = np.nan
    hydraulic_conductivities_layer3[~mask] = np.nan
    hydraulic_conductivities_layer4[~mask] = np.nan

    # fudge streambed conductivity
    cond = (reaches["kf"] >= 10e-6)
    reaches.loc[cond, "rhk"] = reaches.loc[cond, "rhk"] * fudge_parameters["rhkp"].values[model_run]
    cond = (reaches["kf"] < 10e-6)
    reaches.loc[cond, "rhk"] = reaches.loc[cond, "rhk"] * fudge_parameters["rhkf"].values[model_run]
    reaches["man"] = reaches["man"] * fudge_parameters["man"].values[model_run]
    # cond = (reaches["rhk"] > 1)
    # reaches.loc[cond, "rhk"] = reaches.loc[cond, "rhk"] * 1.0

    hydraulic_conductivities = np.array([hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4])

    specific_yield_layer1 = recalc_specific_yield(hydraulic_conductivities_layer1)
    specific_yield_layer2 = recalc_specific_yield(hydraulic_conductivities_layer2)
    specific_yield_layer3 = recalc_specific_yield(hydraulic_conductivities_layer3)
    specific_yield_layer4 = recalc_specific_yield(hydraulic_conductivities_layer4)

    # modify specific yield
    cond1 = (hydraulic_conductivities_layer1_ < 10.0e-07)
    cond2 = (hydraulic_conductivities_layer2_ < 10.0e-07)
    specific_yield_layer2[cond2] = 0.05
    cond3 = (hydraulic_conductivities_layer3_ < 10.0e-07)
    specific_yield_layer3[cond3] = 0.02
    cond4 = (hydraulic_conductivities_layer4_ < 10.0e-07)
    specific_yield_layer4[cond4] = 0.01
    specific_yield_layer1[np.isnan(specific_yield_layer1)] = 0
    specific_yield_layer2[np.isnan(specific_yield_layer2)] = 0
    specific_yield_layer3[np.isnan(specific_yield_layer3)] = 0
    specific_yield_layer4[np.isnan(specific_yield_layer4)] = 0
    _specific_yield_layer1 = sp.ndimage.gaussian_filter(specific_yield_layer1, [1.5, 1.5],  mode="constant")
    _specific_yield_layer2 = sp.ndimage.gaussian_filter(specific_yield_layer2, [1.5, 1.5],  mode="constant")
    _specific_yield_layer3 = sp.ndimage.gaussian_filter(specific_yield_layer3, [1.5, 1.5],  mode="constant")
    _specific_yield_layer4 = sp.ndimage.gaussian_filter(specific_yield_layer4, [1.5, 1.5],  mode="constant")
    specific_yield_layer1[cond1] = _specific_yield_layer1[cond1]
    specific_yield_layer2[cond2] = _specific_yield_layer2[cond2]
    specific_yield_layer3[cond3] = _specific_yield_layer3[cond3]
    specific_yield_layer4[cond4] = _specific_yield_layer4[cond4]
    specific_yield_layer1[~mask] = np.nan
    specific_yield_layer2[~mask] = np.nan
    specific_yield_layer3[~mask] = np.nan
    specific_yield_layer4[~mask] = np.nan

    specific_yield_layers = [specific_yield_layer1, specific_yield_layer2, specific_yield_layer3, specific_yield_layer4]

    # load observed groundwater heads (average values of the observation wells)
    path = base_path.parent / "observations" / "observed_groundwater_heads_avg.csv"
    observed_groundwater_heads = pd.read_csv(path, sep=";", skiprows=0)

    # load observed streamflow
    path = base_path.parent / "observations" / "observed_streamflow.csv"
    observed_streamflow = pd.read_csv(path, sep=";", skiprows=0, index_col=0)

    # load interpolated groundwater heads
    base_path = Path(__file__).parent
    src = rasterio.open(str(base_path.parent / "input" / "groundwater_heads_interpolated_50m.tif"))
    gw_heads_interpolated = src.read(1)

    grid_extent = (ds_mf.lon.values[0] / 1000, ds_mf.lon.values[-1] / 1000, ds_mf.lat.values[-1] / 1000, ds_mf.lat.values[0] / 1000)
    fig, axes = plt.subplots(figsize=(4, 4))
    gw_heads_interpolated[~mask] = np.nan
    plt.imshow(gw_heads_interpolated, cmap='terrain', aspect='equal', extent=grid_extent)
    plt.colorbar(label='[m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / "groundwater_heads_interpolated.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    gw_depth_interpolated = topography - gw_heads_interpolated
    plt.imshow(gw_depth_interpolated, cmap='viridis', aspect='equal', vmin=0, vmax=20, extent=grid_extent)
    plt.colorbar(label='[m]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / "groundwater_depths_interpolated.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    # load observed groundwater heads
    rows = observed_groundwater_heads.iloc[:, -2].values  # row IDs of the observation wells
    cols = observed_groundwater_heads.iloc[:, -3].values  # column IDs of the observation wells
    obs_depths = topography[rows, cols].flatten() - observed_groundwater_heads.iloc[:, -1].values  # observed groundwater depths
    obs = observed_groundwater_heads.iloc[:, -1].values

    dict_obs_stage_id = modflow_config["dict_obs_stage_rnos"]
    dict_obs_flow_id = modflow_config["dict_obs_flow_rnos"]
    sfr_obs_points = [key.split('_')[0] for key in dict_obs_stage_id.keys()]

    dict_obs_stage_id_inv = {v: k for k, v in dict_obs_stage_id.items()}
    dict_obs_flow_id_inv = {v: k for k, v in dict_obs_flow_id.items()}

    df_sfr = pd.DataFrame(index=sfr_obs_points, columns=["rno", "layer", "x", "y", "rlen", "rwid", "rtp", "rgrd", "man", "rhk", "water_head", "water_depth", "flow", "kf", "topo", "elev_layer1", "elev_layer2", "gw_head", "topo-rtp", "topo1-rtp", "gw-sw", "sw-gw_flux"])
    df_sfr["rno"] = [key for key in dict_obs_stage_id.values()]
    for rno in df_sfr["rno"].values:
        z = int(reaches.loc[reaches["rno"] == rno, "k"].values[0])
        y = int(reaches.loc[reaches["rno"] == rno, "i"].values[0])
        x = int(reaches.loc[reaches["rno"] == rno, "j"].values[0])
        kf = hydraulic_conductivities[z, y, x] / 86400
        df_sfr.loc[df_sfr["rno"] == rno, "layer"] = reaches.loc[reaches["rno"] == rno, "k"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "y"] = reaches.loc[reaches["rno"] == rno, "i"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "x"] = reaches.loc[reaches["rno"] == rno, "j"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rlen"] = reaches.loc[reaches["rno"] == rno, "rlen"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rwid"] = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rtp"] = reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rgrd"] = reaches.loc[reaches["rno"] == rno, "rgrd"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "rhk"] = reaches.loc[reaches["rno"] == rno, "rhk"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "man"] = reaches.loc[reaches["rno"] == rno, "man"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "kf"] = kf
        rwidth = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
        stage_depth = df_sfr_.loc[0, dict_obs_stage_id_inv[rno]] - reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
        df_sfr.loc[df_sfr["rno"] == rno, "water_head"] = df_sfr_.loc[0, dict_obs_stage_id_inv[rno]]
        df_sfr.loc[df_sfr["rno"] == rno, "water_depth"] = stage_depth
        flow = (df_sfr_.loc[0, dict_obs_flow_id_inv[rno]] * (-1)) / 86400
        if stage_depth < 0:
            flow = 0
            stage_depth = 0
        df_sfr.loc[df_sfr["rno"] == rno, "flow"] = flow
        df_sfr.loc[df_sfr["rno"] == rno, "gw_head"] = ds_mf["head"].values[0, z, y, x]
        df_sfr.loc[df_sfr["rno"] == rno, "gw-sw"] = ds_mf["head"].values[0, z, y, x] - df_sfr_.loc[0, dict_obs_stage_id_inv[rno]]
        df_sfr.loc[df_sfr["rno"] == rno, "sw-gw_flux"] = gw_sw[y, x]
        df_sfr.loc[df_sfr["rno"] == rno, "topo"] = topography[y, x]
        df_sfr.loc[df_sfr["rno"] == rno, "elev_layer1"] = elevation_bottom_layer1[y, x]
        df_sfr.loc[df_sfr["rno"] == rno, "elev_layer2"] = elevation_bottom_layer2[y, x] 
    
    df_sfr.loc[:, "topo-rtp"] = df_sfr.loc[:, "topo"] - df_sfr.loc[:, "rtp"]
    df_sfr.loc[:, "topo1-rtp"] = df_sfr.loc[:, "elev_layer1"] - df_sfr.loc[:, "rtp"]

    file = base_path / "output" / f"dmn_run_{model_run}_sfr.csv"
    df_sfr.to_csv(file, sep=";")

    obs_flow_stage = ["EBNET", "OBERAMBRINGEN", "WIESNECK"]
    df_sfr = df_sfr.loc[obs_flow_stage, :]
    sim_water_depth = df_sfr["water_depth"].values
    sim_water_depth[sim_water_depth < 0] = 0
    obs_water_depth = observed_streamflow["WDavg"].values
    diff_sim_obs_water_depth = sim_water_depth - obs_water_depth
    obs_streamflow = observed_streamflow["Qavg"].values
    sim_streamflow = df_sfr["flow"].values
    diff_sim_obs_streamflow = sim_streamflow - obs_streamflow

    groundwater_heads[groundwater_heads > topography] = topography[groundwater_heads > topography]
    # extract the simulated groundwater heads at the location of the observation wells
    sim_depths = topography[rows, cols].flatten() - groundwater_heads[rows, cols].flatten()
    sim_depths = np.where(sim_depths < 0, 0, sim_depths)
    sim = groundwater_heads[rows, cols].flatten()

    interp_depths = topography[rows, cols].flatten() - gw_heads_interpolated[rows, cols].flatten()
    observed_groundwater_heads["sim-obs"] = sim - obs
    observed_groundwater_heads["sim_head"] = groundwater_heads[rows, cols].flatten()
    observed_groundwater_heads["topo"] = topography[rows, cols].flatten()
    observed_groundwater_heads["obs_depths"] = obs_depths
    observed_groundwater_heads["sim_depths"] = sim_depths
    observed_groundwater_heads["sim-int"] = sim_depths - interp_depths
    observed_groundwater_heads["int-obs"] = interp_depths - obs_depths
    observed_groundwater_heads["kf_fudged_layer1"] = hydraulic_conductivities_layer1[rows, cols].flatten() / 86400
    observed_groundwater_heads["kf_fudged_layer2"] = hydraulic_conductivities_layer2[rows, cols].flatten() / 86400
    observed_groundwater_heads["kf_fudged_layer3"] = hydraulic_conductivities_layer3[rows, cols].flatten() / 86400
    observed_groundwater_heads["kf_fudged_layer4"] = hydraulic_conductivities_layer4[rows, cols].flatten() / 86400
    observed_groundwater_heads["kf_layer1"] = hydraulic_conductivities_layer1_[rows, cols].flatten()
    observed_groundwater_heads["kf_layer2"] = hydraulic_conductivities_layer2_[rows, cols].flatten()
    observed_groundwater_heads["kf_layer3"] = hydraulic_conductivities_layer3_[rows, cols].flatten()
    observed_groundwater_heads["kf_layer4"] = hydraulic_conductivities_layer4_[rows, cols].flatten()
    observed_groundwater_heads.to_csv(base_path / "output" / f"groundwater_heads_{model_run}.csv", sep=";", index=False)

    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    gauge_obs_y = observed_streamflow["y-coordinate"].values / 1000  # row IDs of the observation wells
    gauge_obs_x = observed_streamflow["x-coordinate"].values / 1000  # column IDs of the observation wells
    plt.scatter(gauge_obs_x, gauge_obs_y, c=diff_sim_obs_water_depth, s=5, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"difference_sim_obs_water_depth_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    gauge_obs_y = observed_streamflow["y-coordinate"].values / 1000  # row IDs of the observation wells
    gauge_obs_x = observed_streamflow["x-coordinate"].values / 1000  # column IDs of the observation wells
    plt.scatter(gauge_obs_x, gauge_obs_y, c=diff_sim_obs_streamflow, s=5, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='[$m^3$/s]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"difference_sim_obs_streamflow_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    # calculate mean error
    print(np.mean(sim[:-2] - obs[:-2]))
    # calculate mean absolute error
    print(np.mean(np.abs(sim[:-2] - obs[:-2])))
    print(sp.stats.spearmanr(sim_depths[:-2], obs_depths[:-2])[0])

    diff_sim_obs = sim - obs
    cm = plt.get_cmap('PuOr')
    cm.set_bad(color='grey')
    grid_extent = (ds_mf.lon.values[0] / 1000, ds_mf.lon.values[-1] / 1000, ds_mf.lat.values[-1] / 1000, ds_mf.lat.values[0] / 1000)
    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    # wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]) * 50
    # wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]) * 50
    # plt.scatter(wells_x, wells_y, marker='x', s=5, c='black')
    wells_obs_y = observed_groundwater_heads["y-coordinate"].values / 1000   # row IDs of the observation wells
    wells_obs_x = observed_groundwater_heads["x-coordinate"].values / 1000   # column IDs of the observation wells
    plt.scatter(wells_obs_x, wells_obs_y, c=diff_sim_obs, s=5, cmap=cm, vmin=-5, vmax=5)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"difference_sim_obs_{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    grid_extent = (ds_mf.lon.values[0] / 1000, ds_mf.lon.values[-1] / 1000, ds_mf.lat.values[-1] / 1000, ds_mf.lat.values[0] / 1000)
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(groundwater_heads[:, :] - gw_heads_interpolated, cmap='PuOr', aspect='equal', vmin=-10, vmax=10, extent=grid_extent)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / f"difference_sim_{model_run}_int.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    grid_extent = (ds_mf.lon.values[0] / 1000, ds_mf.lon.values[-1] / 1000, ds_mf.lat.values[-1] / 1000, ds_mf.lat.values[0] / 1000)
    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]) * 50
    wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]) * 50
    plt.scatter(wells_x, wells_y, marker='x', s=5, c='black')
    wells_obs_y = observed_groundwater_heads["y-coordinate"].values   # row IDs of the observation wells
    wells_obs_x = observed_groundwater_heads["x-coordinate"].values   # column IDs of the observation wells
    plt.scatter(wells_obs_x, wells_obs_y, c=sim, s=5, cmap="viridis", vmin=150, vmax=400)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / f"groundwater_heads_sim{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    grid_extent = (ds_mf.lon.values[0] / 1000, ds_mf.lon.values[-1] / 1000, ds_mf.lat.values[-1] / 1000, ds_mf.lat.values[0] / 1000)
    fig, axes = plt.subplots(figsize=(4, 4))
    topography[~mask] = np.nan
    wells_y = np.array([266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]) * 50
    wells_x = np.array([66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]) * 50
    plt.scatter(wells_x, wells_y, marker='x', s=5, c='black')
    wells_obs_y = observed_groundwater_heads["y-coordinate"].values   # row IDs of the observation wells
    wells_obs_x = observed_groundwater_heads["x-coordinate"].values   # column IDs of the observation wells
    plt.scatter(wells_obs_x, wells_obs_y, c=obs, s=5, cmap="viridis", vmin=150, vmax=400)
    plt.colorbar(label='[m]', shrink=0.45)
    plt.imshow(topography, cmap='terrain', aspect='equal', alpha=0.5, extent=grid_extent)
    plt.grid(zorder=0)
    plt.xlabel('x-coordinate [km]')
    plt.ylabel('y-coordinate [km]')
    plt.tight_layout()
    file = Path(__file__).parent / "figures" / "observed_groundwater_heads.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(obs_water_depth, sim_water_depth, marker='.', s=8, c='black')
    axes.set_ylabel('Simulated water depth [m]')
    axes.set_xlabel('Observed water depth [m]')
    axes.set_xlim(-0.1, 1)
    axes.set_ylim(-0.1, 1)
    axes.plot(axes.get_xlim(), axes.get_ylim(), ls="--", c=".3", zorder=1, alpha=0.5)
    # axes.text(axes.get_xlim()[0] + 0.1, axes.get_ylim()[1] - 0.1, f"ME: {df_params_metrics.loc[model_run, 'ME']:.2f} m")
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"scatter_obs_sim_water_depth{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(obs_streamflow, sim_streamflow, marker='.', s=8, c='black')
    axes.set_ylabel('Simulated streamflow [$m^3$/s]')
    axes.set_xlabel('Observed streamflow [$m^3$/s]')
    axes.set_xlim(0, 8)
    axes.set_ylim(0, 8)
    axes.plot(axes.get_xlim(), axes.get_ylim(), ls="--", c=".3", zorder=1, alpha=0.5)
    # axes.text(axes.get_xlim()[0] + 0.1, axes.get_ylim()[1] - 0.1, f"ME: {df_params_metrics.loc[model_run, 'ME']:.2f} m")
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"scatter_obs_sim_streamflow{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(obs_depths[:-2], sim_depths[:-2], marker='.', s=5, c='black')
    axes.set_ylabel('Simulated groundwater depth [m]')
    axes.set_xlabel('Observed groundwater depth [m]')
    axes.set_xlim(np.nanmin(sim_depths[:-2]) - 1, np.nanmax(sim_depths[:-2]) + 1)
    axes.set_ylim(np.nanmin(sim_depths[:-2]) - 1, np.nanmax(sim_depths[:-2]) + 1)
    axes.plot(axes.get_xlim(), axes.get_ylim(), ls="--", c=".3", zorder=1, alpha=0.5)
    # axes.text(axes.get_xlim()[0] + 0.1, axes.get_ylim()[1] - 0.1, f"ME: {df_params_metrics.loc[model_run, 'ME']:.2f} m")
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"scatter_obs_sim{model_run}_.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(obs_depths[:-2], sim_depths[:-2], marker='.', s=5, c='black')
    axes.set_ylabel('Simulated groundwater depth [m]')
    axes.set_xlabel('Observed groundwater depth [m]')
    axes.set_xlim(0, 30)
    axes.set_ylim(0, 30)
    axes.plot(axes.get_xlim(), axes.get_ylim(), ls="--", c=".3", zorder=1, alpha=0.5)
    # axes.text(axes.get_xlim()[0] + 0.1, axes.get_ylim()[1] - 0.1, f"ME: {df_params_metrics.loc[model_run, 'ME']:.2f} m")
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"scatter_obs_sim{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


    fig, axes = plt.subplots(figsize=(4, 4))
    axes.scatter(range(len(diff_sim_obs[:-2])), diff_sim_obs[:-2], marker='.', s=5, c='black')
    axes.set_ylabel('Bias [m]')
    axes.set_xlabel('# Observation well')
    axes.axhline(y=0, color='grey', linestyle='--', zorder=1, alpha=0.5)
    fig.tight_layout()
    file = Path(__file__).parent / "figures" / f"scatter_diff_obs_sim{model_run}.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


    for layer in range(4):
        fig, axes = plt.subplots(figsize=(4, 4))
        plt.imshow(ds_mf['head'].isel(Time=0, layer=layer).values, extent=grid_extent, cmap='viridis', aspect='equal')
        plt.colorbar(label='groundwater head \n[m a.s.l.]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.tight_layout()
        i = layer + 1
        file = Path(__file__).parent / "figures" / f"gw_head_steady_state_layer{i}_grid_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        elevation_bottom_layer = elevation_bottom_layers[layer]
        gw_thickness = ds_mf['head'].isel(Time=0, layer=layer).values - elevation_bottom_layer
        gw_thickness[gw_thickness <= 0] = 0
        plt.imshow(gw_thickness, extent=grid_extent, cmap='viridis', aspect='equal')
        plt.colorbar(label='groundwater thickness [m]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.tight_layout()
        i = layer + 1
        file = Path(__file__).parent / "figures" / f"gw_thickness_steady_state_layer{i}_grid_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        gw_depth = topography - ds_mf['head'].isel(Time=0, layer=layer).values
        plt.imshow(gw_depth, extent=grid_extent, cmap='viridis', aspect='equal', vmin=0, vmax=20)
        plt.colorbar(label='groundwater depth [m]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.tight_layout()
        i = layer + 1
        file = Path(__file__).parent / "figures" / f"gw_depth_steady_state_layer{i}_grid_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        fig, axes = plt.subplots(figsize=(4, 4))
        flow_residuals = ds_mf['flow_residual'].isel(Time=0, layer=layer).values
        mask1 = (flow_residuals <= 1) & (flow_residuals >= -1)
        flow_residuals[mask1] = np.nan
        minmax = np.nanmax(np.abs(flow_residuals))
        plt.imshow(flow_residuals, extent=grid_extent, cmap='PuOr', aspect='equal', vmin=-minmax, vmax=minmax)
        plt.colorbar(label='groundwater flow residuals [m/day]', shrink=0.5)
        plt.grid(zorder=0)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.tight_layout()
        i = layer + 1
        file = Path(__file__).parent / "figures" / f"gw_flow_residuals_steady_state_layer{i}_grid_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close("all")

        print(f"Layer {layer} flow residual (min, max): {np.nanmin(flow_residuals):.2f}, {np.nanmax(flow_residuals):.2f} m/day")


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
        file = Path(__file__).parent / "figures" / f"hydraulic_conductivity_layer_{i}_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

    for i, specific_yield_layer in enumerate(specific_yield_layers):
        fig, axes = plt.subplots(figsize=(4, 4))
        specific_yield_layer[~mask] = np.nan
        plt.imshow(specific_yield_layer, extent=grid_extent, cmap='Blues', aspect='equal', vmin=0.01, vmax=0.35)
        cbar = plt.colorbar(label='$sy$ [-]', shrink=0.48)
        plt.grid(zorder=0)
        plt.xlabel('x-coordinate [km]')
        plt.ylabel('y-coordinate [km]')
        plt.tight_layout()
        file = Path(__file__).parent / "figures" / f"specific_yield_layer_{i}_{model_run}.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

    return

if __name__ == "__main__":
    main()
