from pathlib import Path
import rasterio
import xarray as xr
import pandas as pd
import geoxarray
import numpy as np
import datetime
import flopy
import scipy

import click

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

@click.option("-mr", "--model-run", type=int, default=8304)
@click.command("main")
def main(model_run):
    base_path = Path(__file__).parent

    sim = flopy.mf6.MFSimulation.load(
        sim_ws=base_path / "output",
        exe_name="mf6",
        version="mf6",
        verbosity_level=0,
    )

    ml = sim.get_model(f"dmn_run_{model_run}")

    # load MODFLOW parameters
    path = Path(__file__).parent.parent / "input" / "parameters_modflow.nc"
    ds_params = xr.open_dataset(path, engine="h5netcdf")
    topography = ds_params['elevations'].isel(z=0).values
    spatial_ref = ds_params.spatial_ref

    output_file = Path(__file__).parent / "output" / f"modflow_output_run_{model_run}_pre.nc"
    ds_mf_pre = xr.open_dataset(output_file, engine="h5netcdf")

    output_file = Path(__file__).parent / "output" / f"modflow_output_run_{model_run}_pre1.nc"
    ds_mf_pre1 = xr.open_dataset(output_file, engine="h5netcdf")

    groundwater_extraction = pd.read_csv(base_path.parent / "input" / "groundwater_extraction.csv", sep=";")
    groundwater_extraction["cell_y"] = groundwater_extraction["cell_y"].values - 1
    groundwater_extraction["cell_x"] = groundwater_extraction["cell_x"].values - 1
    groundwater_extraction["layer"] = groundwater_extraction["layer"].values - 1

    gw_extraction = np.zeros(ds_params['kf'].values.shape)
    gw_extraction[:, :, :] = np.nan
    for idx, row in groundwater_extraction.iterrows():
        gw_extraction[row["layer"], row["cell_y"], row["cell_x"]] = row["annual_average"]

    # load the netcdf file
    output_file = Path(__file__).parent / "output" / f"modflow_output_run_{model_run}.nc"
    ds_mf = xr.open_dataset(output_file, engine="h5netcdf")
    gw_depth = topography[np.newaxis, :, :] - ds_mf['head'].isel(Time=0).values
    cond = (gw_depth < 0)
    gw_depth[cond] = 0

    thickness_layer1 = ds_params['elevations'].isel(z=0).values - ds_params['elevations'].isel(z=1).values
    thickness_layer2 = ds_params['elevations'].isel(z=1).values - ds_params['elevations'].isel(z=2).values
    thickness_layer3 = ds_params['elevations'].isel(z=2).values - ds_params['elevations'].isel(z=3).values
    thickness_layer4 = ds_params['elevations'].isel(z=3).values - ds_params['elevations'].isel(z=4).values
    thickness = np.array([thickness_layer1, thickness_layer2, thickness_layer3, thickness_layer4])

    # load the boundary conditions
    path = Path(__file__).parent.parent / "input" / "boundary_conditions.nc"
    ds_bc = xr.open_dataset(path, engine="h5netcdf")
    direct_recharge = ds_bc['recharge'].values * ((50 * 50)/ 1000)

    mask = np.isfinite(topography)
    # set Schoenberg to inactive
    mask_schoenberg = (ds_params["mask_schoenberg"].values == 1)
    mask = np.where(mask_schoenberg, False, mask)
    mask_boundary_condition_schoenberg = ds_bc["mask_schoenberg_bc"].values
    mask = np.where(mask_boundary_condition_schoenberg, True, mask)
    mask_custom_hausen1 = (ds_params["mask_kf_18e_3_lower_moehlin"].values == 1)
    mask_custom_hausen2 = (ds_params["mask_kf_2e_7_lower_moehlin_and_dreisam"].values == 1)

    # read SFR data
    reaches = pd.read_csv(base_path.parent / 'input' / 'sfr_packagedata_modified.csv', sep=';')
    reaches.iloc[:, 0] = reaches.iloc[:, 0].astype(int) - 1  # convert to zero-based indexing
    reaches.iloc[:, 1] = reaches.iloc[:, 1].astype(int) - 1
    reaches.iloc[:, 2] = reaches.iloc[:, 2].astype(int) - 1  # convert to zero-based indexing
    reaches.iloc[:, 3] = reaches.iloc[:, 3].astype(int) - 1  # convert to zero-based indexing
    reaches.iloc[:, 4] = reaches.iloc[:, 4].astype(float) 
    reaches.iloc[:, 5] = reaches.iloc[:, 5].astype(int)
    reaches.iloc[:, 6] = reaches.iloc[:, 6].astype(float)
    reaches.iloc[:, 7] = reaches.iloc[:, 7].astype(float)
    reaches.iloc[:, 8] = reaches.iloc[:, 8].astype(float)
    reaches.iloc[:, 9] = reaches.iloc[:, 9].astype(float)
    reaches.iloc[:, 10] = reaches.iloc[:, 10].astype(float)
    reaches.iloc[:, 11] = reaches.iloc[:, 11].astype(int)
    reaches.iloc[:, 12] = reaches.iloc[:, 12].astype(float)
    reaches.iloc[:, 13] = reaches.iloc[:, 13].astype(int)

    rwid = np.empty_like(topography)
    rwid[:, :] = np.nan
    rlen = np.empty_like(topography)
    rlen[:, :] = np.nan
    rhk = np.empty_like(topography)
    rhk[:, :] = np.nan
    man = np.empty_like(topography)
    man[:, :] = np.nan

    for rno, x, y in zip(reaches.iloc[:, 0], reaches.iloc[:, 2], reaches.iloc[:, 3]):
        rwid[x, y] = reaches.loc[rno, 'rwid']
        rlen[x, y] = reaches.loc[rno, 'rlen']
        man[x, y] = reaches.loc[rno, 'man']
        rhk[x, y] = reaches.loc[rno, 'rhk']

    indirect_recharge = np.nanmean(ds_mf['gw_sw'].isel(Time=0).values, axis=0)

    # fudge parameters
    path = base_path / "fudge_parameters_modflow.csv"
    fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)

    # Create the node property flow package with hydraulic conducitivities
    hydraulic_conductivities_layer1 = np.copy(ds_params["kf"].isel(layer=0).values)
    hydraulic_conductivities_layer2 = np.copy(ds_params["kf"].isel(layer=1).values)
    hydraulic_conductivities_layer3 = np.copy(ds_params["kf"].isel(layer=2).values)
    hydraulic_conductivities_layer4 = np.copy(ds_params["kf"].isel(layer=3).values)

    specific_yield_layer1 = recalc_specific_yield(hydraulic_conductivities_layer1)
    specific_yield_layer2 = recalc_specific_yield(hydraulic_conductivities_layer2)
    specific_yield_layer3 = recalc_specific_yield(hydraulic_conductivities_layer3)
    specific_yield_layer4 = recalc_specific_yield(hydraulic_conductivities_layer4)

    sy = np.array([specific_yield_layer1, specific_yield_layer2, specific_yield_layer3, specific_yield_layer4])

    hydraulic_conductivities_layer1_ = np.copy(ds_params["kf"].isel(layer=0).values) / 86400
    hydraulic_conductivities_layer2_ = np.copy(ds_params["kf"].isel(layer=1).values) / 86400
    hydraulic_conductivities_layer3_ = np.copy(ds_params["kf"].isel(layer=2).values) / 86400
    hydraulic_conductivities_layer4_ = np.copy(ds_params["kf"].isel(layer=3).values) / 86400
    
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
    hydraulic_conductivities_layer1[mask81] = hydraulic_conductivities_layer1[mask81] * fudge_parameters["-8_1"].values[model_run]

    hydraulic_conductivities_layer1[mask71] = hydraulic_conductivities_layer1[mask71] * fudge_parameters["-7_1"].values[model_run]
    hydraulic_conductivities_layer2[mask72] = hydraulic_conductivities_layer2[mask72] * fudge_parameters["-7_2"].values[model_run]
    hydraulic_conductivities_layer3[mask73] = hydraulic_conductivities_layer3[mask73] * fudge_parameters["-7_3"].values[model_run]
    hydraulic_conductivities_layer4[mask74] = hydraulic_conductivities_layer4[mask74] * fudge_parameters["-7_4"].values[model_run]

    hydraulic_conductivities_layer1[mask61] = hydraulic_conductivities_layer1[mask61] * fudge_parameters["-6_1"].values[model_run]

    hydraulic_conductivities_layer1[mask51] = hydraulic_conductivities_layer1[mask51] * fudge_parameters["-5_1"].values[model_run]
    hydraulic_conductivities_layer2[mask52] = hydraulic_conductivities_layer2[mask52] * fudge_parameters["-5_2"].values[model_run]
    hydraulic_conductivities_layer3[mask53] = hydraulic_conductivities_layer3[mask53] * fudge_parameters["-5_3"].values[model_run]
    hydraulic_conductivities_layer4[mask54] = hydraulic_conductivities_layer4[mask54] * fudge_parameters["-5_4"].values[model_run]

    hydraulic_conductivities_layer2[mask42] = hydraulic_conductivities_layer2[mask42] * fudge_parameters["-4_2"].values[model_run]
    hydraulic_conductivities_layer3[mask43] = hydraulic_conductivities_layer3[mask43] * fudge_parameters["-4_3"].values[model_run]
    hydraulic_conductivities_layer4[mask44] = hydraulic_conductivities_layer4[mask44] * fudge_parameters["-4_4"].values[model_run]

    hydraulic_conductivities_layer2[mask132] = hydraulic_conductivities_layer2[mask132] * fudge_parameters["1-3_2"].values[model_run]
    hydraulic_conductivities_layer3[mask133] = hydraulic_conductivities_layer3[mask133] * fudge_parameters["1-3_3"].values[model_run]

    hydraulic_conductivities_layer2[mask232] = hydraulic_conductivities_layer2[mask232] * fudge_parameters["1.8-3_2"].values[model_run]
    hydraulic_conductivities_layer3[mask233] = hydraulic_conductivities_layer3[mask233] * fudge_parameters["1.8-3_3"].values[model_run]
    hydraulic_conductivities_layer4[mask234] = hydraulic_conductivities_layer4[mask234] * fudge_parameters["1.8-3_4"].values[model_run]

    hydraulic_conductivities_layer2[mask332] = hydraulic_conductivities_layer2[mask332] * fudge_parameters["3-3_2"].values[model_run]
    hydraulic_conductivities_layer3[mask333] = hydraulic_conductivities_layer3[mask333] * fudge_parameters["3-3_3"].values[model_run]

    hydraulic_conductivities_layer2[mask432] = hydraulic_conductivities_layer2[mask432] * fudge_parameters["4-3_2"].values[model_run]
    hydraulic_conductivities_layer3[mask433] = hydraulic_conductivities_layer3[mask433] * fudge_parameters["4-3_3"].values[model_run]

    # adjust hydraulic conductivities
    hydraulic_conductivities_layer2[mask232 & mask_custom_hausen1] = hydraulic_conductivities_layer2[mask232 & mask_custom_hausen1] * fudge_parameters["hausen1_re"].values[model_run]
    hydraulic_conductivities_layer3[mask233 & mask_custom_hausen1] = hydraulic_conductivities_layer2[mask233 & mask_custom_hausen1] * fudge_parameters["hausen1_re"].values[model_run]

    hydraulic_conductivities_layer2[mask72 & mask_custom_hausen2] = hydraulic_conductivities_layer2[mask72 & mask_custom_hausen2] * fudge_parameters["hausen2_re"].values[model_run]
    hydraulic_conductivities_layer3[mask73 & mask_custom_hausen2] = hydraulic_conductivities_layer3[mask73 & mask_custom_hausen2] * fudge_parameters["hausen2_re"].values[model_run]

    # smooth transition between fissured and porous aquifers
    hydraulic_conductivities_layer1[np.isnan(hydraulic_conductivities_layer1)] = 0
    hydraulic_conductivities_layer2[np.isnan(hydraulic_conductivities_layer2)] = 0
    hydraulic_conductivities_layer3[np.isnan(hydraulic_conductivities_layer3)] = 0
    hydraulic_conductivities_layer4[np.isnan(hydraulic_conductivities_layer4)] = 0
    _hydraulic_conductivities_layer1 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer1, [1.5, 1.5], mode="constant")
    _hydraulic_conductivities_layer2 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer2, [1.5, 1.5], mode="constant")
    _hydraulic_conductivities_layer3 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer3, [1.5, 1.5], mode="constant")
    _hydraulic_conductivities_layer4 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer4, [1.5, 1.5], mode="constant")
    cond1 = (hydraulic_conductivities_layer1_ <= 10.0e-07)
    cond2 = (hydraulic_conductivities_layer2_ <= 10.0e-07)
    cond3 = (hydraulic_conductivities_layer3_ <= 10.0e-07)
    cond4 = (hydraulic_conductivities_layer4_ <= 10.0e-07)
    hydraulic_conductivities_layer1[cond1] = _hydraulic_conductivities_layer1[cond1]
    hydraulic_conductivities_layer2[cond2] = _hydraulic_conductivities_layer2[cond2]
    hydraulic_conductivities_layer3[cond3] = _hydraulic_conductivities_layer3[cond3]
    hydraulic_conductivities_layer4[cond4] = _hydraulic_conductivities_layer4[cond4]

    gw_depth_layer2 = topography - ds_mf_pre['head'].isel(Time=0, layer=1).values
    gw_depth_layer3 = topography - ds_mf_pre['head'].isel(Time=0, layer=2).values
    gw_depth_layer4 = topography - ds_mf_pre['head'].isel(Time=0, layer=3).values

    cond2 = (gw_depth_layer2 >= 0)
    cond3 = (gw_depth_layer3 >= 0)
    cond4 = (gw_depth_layer4 >= 0)

    gw_depth_layer2[cond2] = 0
    gw_depth_layer3[cond3] = 0
    gw_depth_layer4[cond4] = 0

    gw_depth_min_layer2 = np.nanmin(gw_depth_layer2)
    gw_depth_min_layer3 = np.nanmin(gw_depth_layer3)
    gw_depth_min_layer4 = np.nanmin(gw_depth_layer4)

    scale2 = np.abs(gw_depth_layer2) / np.abs(gw_depth_min_layer2)
    scale3 = np.abs(gw_depth_layer3) / np.abs(gw_depth_min_layer3)
    scale4 = np.abs(gw_depth_layer4) / np.abs(gw_depth_min_layer4)

    cond2 = (gw_depth_layer2 < 0) & (hydraulic_conductivities_layer2_ <= 10.0e-07)
    cond3 = (gw_depth_layer3 < 0) & (hydraulic_conductivities_layer3_ <= 10.0e-07)
    cond4 = (gw_depth_layer4 < 0) & (hydraulic_conductivities_layer4_ <= 10.0e-07)
    hydraulic_conductivities_layer2[cond2] = hydraulic_conductivities_layer2[cond2] * (fudge_parameters["-7_2_re"].values[model_run] * scale2[cond2])
    hydraulic_conductivities_layer3[cond3] = hydraulic_conductivities_layer3[cond3] * (fudge_parameters["-7_3_re"].values[model_run] * scale3[cond3])
    hydraulic_conductivities_layer4[cond4] = hydraulic_conductivities_layer4[cond4] * (fudge_parameters["-7_4_re"].values[model_run] * scale4[cond4])

    gw_depth_layer2 = topography - ds_mf_pre1['head'].isel(Time=0, layer=1).values
    gw_depth_layer3 = topography - ds_mf_pre1['head'].isel(Time=0, layer=2).values
    gw_depth_layer4 = topography - ds_mf_pre1['head'].isel(Time=0, layer=3).values

    cond2 = (gw_depth_layer2 >= 0)
    cond3 = (gw_depth_layer3 >= 0)
    cond4 = (gw_depth_layer4 >= 0)

    gw_depth_layer2[cond2] = 0
    gw_depth_layer3[cond3] = 0
    gw_depth_layer4[cond4] = 0

    gw_depth_min_layer2 = np.nanmin(gw_depth_layer2)
    gw_depth_min_layer3 = np.nanmin(gw_depth_layer3)
    gw_depth_min_layer4 = np.nanmin(gw_depth_layer4)

    scale2 = np.abs(gw_depth_layer2) / np.abs(gw_depth_min_layer2)
    scale3 = np.abs(gw_depth_layer3) / np.abs(gw_depth_min_layer3)
    scale4 = np.abs(gw_depth_layer4) / np.abs(gw_depth_min_layer4)

    cond2 = (gw_depth_layer2 < 0) & (hydraulic_conductivities_layer2_ <= 10.0e-07)
    cond3 = (gw_depth_layer3 < 0) & (hydraulic_conductivities_layer3_ <= 10.0e-07)
    cond4 = (gw_depth_layer4 < 0) & (hydraulic_conductivities_layer4_ <= 10.0e-07)
    hydraulic_conductivities_layer2[cond2] = hydraulic_conductivities_layer2[cond2] * (1 + fudge_parameters["-7_2_re1"].values[model_run] * scale2[cond2])
    hydraulic_conductivities_layer3[cond3] = hydraulic_conductivities_layer3[cond3] * (1 + fudge_parameters["-7_3_re1"].values[model_run] * scale3[cond3])
    hydraulic_conductivities_layer4[cond4] = hydraulic_conductivities_layer4[cond4] * (1 + fudge_parameters["-7_4_re1"].values[model_run] * scale4[cond4])

    reaches["indirect_recharge"] = np.nan
    # increase the hydraulic conductivities of the reach cell by a factor of xx
    reaches["kf"] = np.nan
    c_fissured = 1  # factor to increase the hydraulic conductivity in fissured layers
    for rno, z, y, x in zip(reaches.loc[:, "rno"], reaches.loc[:, "k"], reaches.loc[:, "i"], reaches.loc[:, "j"]):
        reaches.loc[rno, "indirect_recharge"] = indirect_recharge[y, x] * (-1)
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
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run]
                hydraulic_conductivities_layer2[y, x] = hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run]
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run]) / 86400
        elif z == 2:
            kf_riv = hydraulic_conductivities_layer3[y, x] / 86400
            if kf_riv < 10e-6:
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer2[y, x] = hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer3[y, x] = hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured) / 86400
            else:
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run]
                hydraulic_conductivities_layer2[y, x] = hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run]
                hydraulic_conductivities_layer3[y, x] = hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run]
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run]) / 86400
        elif z == 3:
            kf_riv = hydraulic_conductivities_layer4[y, x] / 86400
            if kf_riv < 10e-6:
                hydraulic_conductivities_layer1[y, x] = hydraulic_conductivities_layer1[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer2[y, x] = hydraulic_conductivities_layer2[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer3[y, x] = hydraulic_conductivities_layer3[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                hydraulic_conductivities_layer4[y, x] = hydraulic_conductivities_layer4[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured
                reaches.loc[rno, "kf"] = (hydraulic_conductivities_layer4[y, x] * fudge_parameters["kf_riv"].values[model_run] * c_fissured) / 86400
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


    kf_fudged = np.array([hydraulic_conductivities_layer1 / 86400,
                          hydraulic_conductivities_layer2 / 86400,
                          hydraulic_conductivities_layer3 / 86400,
                          hydraulic_conductivities_layer4 / 86400])
    
    # create the storage package
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
    _specific_yield_layer1 = scipy.ndimage.gaussian_filter(specific_yield_layer1, [1.5, 1.5], mode="constant")
    _specific_yield_layer2 = scipy.ndimage.gaussian_filter(specific_yield_layer2, [1.5, 1.5], mode="constant")
    _specific_yield_layer3 = scipy.ndimage.gaussian_filter(specific_yield_layer3, [1.5, 1.5], mode="constant")
    _specific_yield_layer4 = scipy.ndimage.gaussian_filter(specific_yield_layer4, [1.5, 1.5], mode="constant")
    specific_yield_layer1[cond1] = _specific_yield_layer1[cond1]
    specific_yield_layer2[cond2] = _specific_yield_layer2[cond2]
    specific_yield_layer3[cond3] = _specific_yield_layer3[cond3]
    specific_yield_layer4[cond4] = _specific_yield_layer4[cond4]
    specific_yield_layer1[~mask] = np.nan
    specific_yield_layer2[~mask] = np.nan
    specific_yield_layer3[~mask] = np.nan
    specific_yield_layer4[~mask] = np.nan

    sy_fudged = np.array([specific_yield_layer1,
                          specific_yield_layer2,
                          specific_yield_layer3,
                          specific_yield_layer4])

    # fudge streambed conductivity
    cond = (reaches["kf"] >= 10e-6)
    reaches.loc[cond, "rhk"] = reaches.loc[cond, "rhk"] * fudge_parameters["rhkp"].values[model_run]
    cond = (reaches["kf"] < 10e-6)
    reaches.loc[cond, "rhk"] = reaches.loc[cond, "rhk"] * fudge_parameters["rhkf"].values[model_run]
    reaches["man"] = reaches["man"] * fudge_parameters["man"].values[model_run]

    # adjust streambed conductance for reaches with leakage
    cond = (reaches["indirect_recharge"] < 0)
    reaches.loc[cond, "rhk"] = 1.0e-08

    rhk_fudged = np.empty_like(topography)
    rhk_fudged[:, :] = np.nan
    man_fudged = np.empty_like(topography)
    man_fudged[:, :] = np.nan

    for rno, x, y in zip(reaches.iloc[:, 0], reaches.iloc[:, 2], reaches.iloc[:, 3]):
        man_fudged[x, y] = reaches.loc[rno, 'man']
        rhk_fudged[x, y] = reaches.loc[rno, 'rhk']

    attrs = dict(
            date_created=datetime.datetime.today().isoformat(),
            title="MODFLOW steady-state simulation of the Dreisam-Moehlin-Neumagen catchment",
            institution="University of Freiburg, Chair of Hydrology",
            flopy_version=f"{flopy.__version__}",
            modflow_version=f"{ml.version}",
        )
    coords = {
            "lon": ("lon", ds_mf.lon.values),  # x
            "lat": ("lat", ds_mf.lat.values),  # y
            "layer": ("layer", ds_mf.layer.values),
            "z": ("z", ds_params['z'].values),
            "Time": ("Time", [1]),
        }
    data_vars=dict(
            mask=(["lat", "lon"], mask),
            gw_head=(["Time", "layer", "lat", "lon"], ds_mf['head'].values),
            gw_depth=(["Time", "layer", "lat", "lon"], gw_depth[np.newaxis, :, :, :]),
            indirect_recharge=(["Time", "lat", "lon"], indirect_recharge[np.newaxis, :, :] * (-1)),
            direct_recharge=(["Time", "lat", "lon"], direct_recharge[np.newaxis, :, :]),
            elevations=(["z", "lat", "lon"], ds_params['elevations'].values),
            thickness=(["layer", "lat", "lon"], thickness),
            kf=(["layer", "lat", "lon"], ds_params['kf'].values / 86400),
            sy=(["layer", "lat", "lon"], sy),
            kf_fudged=(["layer", "lat", "lon"], kf_fudged),
            sy_fudged=(["layer", "lat", "lon"], sy_fudged),
            rhk=(["lat", "lon"], rhk),
            rhk_fudged=(["lat", "lon"], rhk_fudged),
            man=(["lat", "lon"], man),
            man_fudged=(["lat", "lon"], man_fudged),
            rwid=(["lat", "lon"], rwid),
            rlen=(["lat", "lon"], rlen),
            rgrd=(["lat", "lon"], ds_mf['sfr_gradient'].values),
            sfr_stage=(["Time", "lat", "lon"], ds_mf['sfr_stage'].values[np.newaxis, :, :]),
            sfr_head=(["Time", "lat", "lon"], ds_mf['sfr_head'].values[np.newaxis, :, :]),
            sfr_flow=(["Time", "lat", "lon"], ds_mf['sfr_flow'].values[np.newaxis, :, :]),
            delta_sw_gw_head=(["Time", "lat", "lon"], ds_mf['delta_sw_gw_head'].values[np.newaxis, :, :]),
            delta_rtp_gw_head=(["Time", "lat", "lon"], ds_mf['delta_rtp_gw_head'].values[np.newaxis, :, :]),
            gw_extraction=(["Time", "layer", "lat", "lon"], gw_extraction[np.newaxis, :, :, :]),
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    ds["mask"].attrs["long_name"] = "Mask of the catchment"
    ds["indirect_recharge"].attrs["units"] = "m3/day"
    ds["indirect_recharge"].attrs["long_name"] = "Recharge from surface water. Negative values indicate surface water leakage into the groundwater."
    ds["direct_recharge"].attrs["units"] = "m3/day"
    ds["direct_recharge"].attrs["long_name"] = "Recharge from percolation"
    ds["elevations"].attrs["units"] = "m a.s.l."
    ds["elevations"].attrs["long_name"] = "layer elevations of the top"
    ds["thickness"].attrs["units"] = "m"
    ds["thickness"].attrs["long_name"] = "layer thickness"
    ds["kf"].attrs["units"] = "m/s"
    ds["kf"].attrs["long_name"] = "hydraulic conductivity (original)"
    ds["kf_fudged"].attrs["units"] = "m/s"
    ds["kf_fudged"].attrs["long_name"] = "hydraulic conductivity (fudged)"
    ds["sy"].attrs["units"] = "m/s"
    ds["sy"].attrs["long_name"] = "specific yield (original; calculated using formula of Marotz 1968)"
    ds["sy_fudged"].attrs["units"] = "m/s"
    ds["sy_fudged"].attrs["long_name"] = "specific yield (fudged)"
    ds["gw_head"].attrs["units"] = "m a.s.l."
    ds["gw_head"].attrs["long_name"] = "Groundwater head"
    ds["gw_depth"].attrs["units"] = "m"
    ds["gw_depth"].attrs["long_name"] = "Groundwater depth"
    ds["sfr_stage"].attrs["units"] = "m"
    ds["sfr_stage"].attrs["long_name"] = "Streamflow depth"
    ds["sfr_head"].attrs["units"] = "m a.s.l."
    ds["sfr_head"].attrs["long_name"] = "Streamflow head"
    ds["sfr_flow"].attrs["units"] = "m3/s"
    ds["sfr_flow"].attrs["long_name"] = "Streamflow"
    ds["rwid"].attrs["units"] = "m"
    ds["rwid"].attrs["long_name"] = "Reach width"
    ds["rlen"].attrs["units"] = "m"
    ds["rlen"].attrs["long_name"] = "Reach length"
    ds["man"].attrs["units"] = ""
    ds["man"].attrs["long_name"] = "Reach Manning's roughness coefficient"
    ds["rhk"].attrs["units"] = "m/s"
    ds["rhk"].attrs["long_name"] = "Reach hydraulic conductivity"
    ds["rgrd"].attrs["units"] = ""
    ds["rgrd"].attrs["long_name"] = "Reach gradient"
    ds["delta_sw_gw_head"].attrs["units"] = "m"
    ds["delta_sw_gw_head"].attrs["long_name"] = "Difference between surface water head and groundwater head"
    ds["delta_rtp_gw_head"].attrs["units"] = "m"
    ds["delta_rtp_gw_head"].attrs["long_name"] = "Difference between elevation of the riverbed and groundwater head"
    ds["gw_extraction"].attrs["units"] = "m3/day"
    ds["gw_extraction"].attrs["long_name"] = "Groundwater extraction"
    # create spatial reference
    ds = ds.geo.write_crs("EPSG:25832")
    ds["spatial_ref"] = spatial_ref
    file = base_path / "output" / "dreisam_moehlin_neumagen.nc"
    comp = dict(zlib=True, complevel=1)  # compress data to save storage
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
    return


if __name__ == "__main__":
    main()