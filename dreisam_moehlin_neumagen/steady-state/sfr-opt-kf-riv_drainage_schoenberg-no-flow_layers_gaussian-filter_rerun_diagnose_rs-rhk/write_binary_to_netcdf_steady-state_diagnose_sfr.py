import sys
import os
from pathlib import Path
import flopy
import xarray as xr
import geoxarray
import numpy as np
import datetime
import flopy.utils.binaryfile as bf
import pandas as pd
import scipy
import yaml

import click

@click.option("-mr", "--model-run", type=int, default=8304)
@click.option("-c", "--converged", type=int, default=1)
@click.command("main")
def main(model_run, converged):
    if converged == 1:
        # try:
        print(sys.version)
        print(f"flopy version: {flopy.__version__}")

        base_path = Path(__file__).parent

        sim = flopy.mf6.MFSimulation.load(
            sim_ws=base_path / "output",
            exe_name="mf6",
            version="mf6",
            verbosity_level=0,
        )

        ml = sim.get_model(f"dmn_run_{model_run}")
        nlayers = np.arange(ml.modelgrid.nlay)

        # load spatial reference and coordinates
        with xr.open_dataset(base_path.parent / "input" / "parameters_modflow.nc") as ds:
            topography = ds['elevations'].isel(z=0).values
            kf_layers = ds['kf'].values
            spatial_ref = ds.spatial_ref
            xcoords = ds.x.values
            ycoords = ds.y.values

        # load MODFLOW parameters
        path = Path(__file__).parent.parent / "input" / "parameters_modflow.nc"
        ds_params = xr.open_dataset(path, engine="h5netcdf")

        path = Path(__file__).parent.parent / "input" / "boundary_conditions.nc"
        ds_bc = xr.open_dataset(path, engine="h5netcdf")

        # load the previous MODFLOW output
        output_file = base_path / "output" / f"modflow_output_run_{model_run}_pre.nc"
        ds_mf_pre = xr.open_dataset(output_file, engine="h5netcdf")

        # load the previous MODFLOW output
        output_file = base_path / "output" / f"modflow_output_run_{model_run}_pre1.nc"
        ds_mf_pre1 = xr.open_dataset(output_file, engine="h5netcdf")

        # load the modflow config file
        file_config = base_path.parent / "config.yml"
        with open(file_config, "r") as file:
            modflow_config = yaml.safe_load(file)

        dict_obs_stage_id = modflow_config["dict_obs_stage_rnos"]
        dict_obs_flow_id = modflow_config["dict_obs_flow_rnos"]

        dict_obs_stage_id_inv = {v: k for k, v in dict_obs_stage_id.items()}
        dict_obs_flow_id_inv = {v: k for k, v in dict_obs_flow_id.items()}

        # load the fugde parameters
        path = base_path / "fudge_parameters_modflow.csv"
        fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)

        # load the SFR reaches
        reaches = pd.read_csv(base_path.parent / 'input' / 'sfr_packagedata_modified.csv', sep=';')
        reaches["rno"] = reaches["rno"] - 1
        reaches["k"] = reaches["k"] - 1
        reaches["i"] = reaches["i"] - 1
        reaches["j"] = reaches["j"] - 1
        reaches["man_fudged"] = reaches["man"].astype(float)
        reaches["rhk_fudged"] = reaches["rhk"].astype(float)

        mask = np.isfinite(topography)
        # set Schoenberg to inactive
        mask_schoenberg = (ds_params["mask_schoenberg"].values == 1)
        mask = np.where(mask_schoenberg, False, mask)
        mask_boundary_condition_schoenberg = ds_bc["mask_schoenberg_bc"].values
        mask = np.where(mask_boundary_condition_schoenberg, True, mask)
        mask_drainage_area = (ds_params["mask_drainage"].values == 1)
        mask_custom_hausen1 = (ds_params["mask_kf_18e_3_lower_moehlin"].values == 1)
        mask_custom_hausen2 = (ds_params["mask_kf_2e_7_lower_moehlin_and_dreisam"].values == 1)

        # Create the node property flow package with hydraulic conducitivities
        hydraulic_conductivities_layer1 = ds_params["kf"].isel(layer=0).values
        hydraulic_conductivities_layer2 = ds_params["kf"].isel(layer=1).values
        hydraulic_conductivities_layer3 = ds_params["kf"].isel(layer=2).values
        hydraulic_conductivities_layer4 = ds_params["kf"].isel(layer=3).values

        hydraulic_conductivities_layer1_ = ds_params["kf"].isel(layer=0).values / 86400
        hydraulic_conductivities_layer2_ = ds_params["kf"].isel(layer=1).values / 86400
        hydraulic_conductivities_layer3_ = ds_params["kf"].isel(layer=2).values / 86400
        hydraulic_conductivities_layer4_ = ds_params["kf"].isel(layer=3).values / 86400
        
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
        hydraulic_conductivities_layer2[cond2] = hydraulic_conductivities_layer2[cond2] * (1 + fudge_parameters["-7_2_re"].values[model_run] * scale2[cond2])
        hydraulic_conductivities_layer3[cond3] = hydraulic_conductivities_layer3[cond3] * (1 + fudge_parameters["-7_3_re"].values[model_run] * scale3[cond3])
        hydraulic_conductivities_layer4[cond4] = hydraulic_conductivities_layer4[cond4] * (1 + fudge_parameters["-7_4_re"].values[model_run] * scale4[cond4])

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
        hydraulic_conductivities_layer2[cond2] = hydraulic_conductivities_layer2[cond2] * (1 + fudge_parameters["-7_2_re"].values[model_run] * scale2[cond2])
        hydraulic_conductivities_layer3[cond3] = hydraulic_conductivities_layer3[cond3] * (1 + fudge_parameters["-7_3_re"].values[model_run] * scale3[cond3])
        hydraulic_conductivities_layer4[cond4] = hydraulic_conductivities_layer4[cond4] * (1 + fudge_parameters["-7_4_re"].values[model_run] * scale4[cond4])

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


        # fudge streambed conductivity
        cond = (reaches["kf"] >= 10e-6)
        reaches.loc[cond, "rhk_fudged"] = reaches.loc[cond, "rhk_fudged"] * fudge_parameters["rhkp"].values[model_run]
        cond = (reaches["kf"] < 10e-6)
        reaches.loc[cond, "rhk_fudged"] = reaches.loc[cond, "rhk_fudged"] * fudge_parameters["rhkf"].values[model_run]
        reaches["man_fudged"] = reaches["man_fudged"] * fudge_parameters["man"].values[model_run]

        # write groundwater head to netcdf
        fhead = base_path / "output" / f"dmn_run_{model_run}.hds"
        hds = flopy.utils.HeadFile(fhead)

        fbudget = base_path / "output" / f"dmn_run_{model_run}.cbc"
        cbb = flopy.utils.CellBudgetFile(fbudget)

        grb_file = base_path / "output" / f"dmn_run_{model_run}.dis.grb"
        flowja = ml.oc.output.budget().get_data(text="FLOW-JA-FACE", kstpkper=(0, 0))[0]
        residual = flopy.mf6.utils.get_residuals(flowja, grb_file=grb_file)

        # load the SFR output file
        output_file = base_path / "output" / f"dmn_run_{model_run}_sfr.obs.csv"
        df_sfr_ = pd.read_csv(output_file, sep=",")

        gw_head = np.where(hds.get_data()[np.newaxis, :, :, :] > 10000, np.nan, hds.get_data()[np.newaxis, :, :, :])
        gw_sw = cbb.get_data(text="SFR", kstpkper=(0, 0), full3D=True)[0].filled(fill_value=np.nan)[np.newaxis, :, :, :]
        gw_sw = np.nansum(gw_sw[0, ...], axis=0)

        rwid = np.nan * np.ones(topography.shape)
        man = np.nan * np.ones(topography.shape)
        rhk = np.nan * np.ones(topography.shape)
        rgrd = np.nan * np.ones(topography.shape)
        sfr_head = np.nan * np.ones(topography.shape)
        sfr_stage = np.nan * np.ones(topography.shape)
        sfr_flow = np.nan * np.ones(topography.shape)
        delta_sw_gw_head = np.nan * np.ones(topography.shape)
        delta_rtp_gw_head = np.nan * np.ones(topography.shape)

        df_sfr = pd.DataFrame(columns=["rno", "layer", "x", "y", "rlen", "rwid", "rtp", "rgrd", "man", "man_fudged", "rhk", "rhk_fudged", "sfr_head", "sfr_stage", "sfr_flow", "kf", "kf_fudged", "topo", "gw_head", "gw-sw", "sw-gw_flux", "sfr"])
        df_sfr["rno"] = reaches["rno"].values
        for rno in df_sfr["rno"].values:
            z = reaches.loc[reaches["rno"] == rno, "k"].values[0]
            y = reaches.loc[reaches["rno"] == rno, "i"].values[0]
            x = reaches.loc[reaches["rno"] == rno, "j"].values[0]
            kf = kf_layers[z, y, x] / 86400
            kf_fudged[z, y, x]
            df_sfr.loc[df_sfr["rno"] == rno, "layer"] = reaches.loc[reaches["rno"] == rno, "k"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "y"] = reaches.loc[reaches["rno"] == rno, "i"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "x"] = reaches.loc[reaches["rno"] == rno, "j"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rlen"] = reaches.loc[reaches["rno"] == rno, "rlen"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rwid"] = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rtp"] = reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rgrd"] = reaches.loc[reaches["rno"] == rno, "rgrd"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rhk"] = reaches.loc[reaches["rno"] == rno, "rhk"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "man"] = reaches.loc[reaches["rno"] == rno, "man"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rhk_fudged"] = reaches.loc[reaches["rno"] == rno, "rhk_fudged"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "man_fudged"] = reaches.loc[reaches["rno"] == rno, "man_fudged"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "kf"] = kf
            df_sfr.loc[df_sfr["rno"] == rno, "kf_fudged"] = kf_fudged[z, y, x]
            rwidth = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
            stage = df_sfr_.loc[0, dict_obs_stage_id_inv[rno]] - reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "sfr_head"] = df_sfr_.loc[0, dict_obs_stage_id_inv[rno]]
            df_sfr.loc[df_sfr["rno"] == rno, "sfr_stage"] = stage
            flow = (df_sfr_.loc[0, dict_obs_flow_id_inv[rno]] * (-1)) / 86400
            df_sfr.loc[df_sfr["rno"] == rno, "sfr_flow"] = flow
            df_sfr.loc[df_sfr["rno"] == rno, "gw_head"] = gw_head[0, z, y, x]
            df_sfr.loc[df_sfr["rno"] == rno, "gw-sw"] = gw_head[0, z, y, x] - df_sfr_.loc[0, dict_obs_stage_id_inv[rno]]
            df_sfr.loc[df_sfr["rno"] == rno, "sw-gw_flux"] = gw_sw[y, x] 
            df_sfr.loc[df_sfr["rno"] == rno, "sfr"] = df_sfr_.loc[0, f"{rno}_SFR"]
            df_sfr.loc[df_sfr["rno"] == rno, "depth"] = df_sfr_.loc[0, f"{rno}_DEPTH"]
            df_sfr.loc[df_sfr["rno"] == rno, "topo"] = topography[y, x]
            rwid[y, x] = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
            rhk[y, x] = reaches.loc[reaches["rno"] == rno, "rhk"].values[0]
            rgrd[y, x] = reaches.loc[reaches["rno"] == rno, "rgrd"].values[0]
            sfr_head[y, x] = df_sfr.loc[df_sfr["rno"] == rno, "sfr_head"].values[0]
            sfr_stage[y, x] = df_sfr.loc[df_sfr["rno"] == rno, "sfr_stage"].values[0]
            sfr_flow[y, x] = df_sfr.loc[df_sfr["rno"] == rno, "sfr_flow"].values[0]
            man[y, x] = reaches.loc[reaches["rno"] == rno, "man"].values[0]
            delta_sw_gw_head[y, x] = df_sfr_.loc[0, dict_obs_stage_id_inv[rno]] - gw_head[0, z, y, x]
            delta_rtp_gw_head[y, x] = reaches.loc[reaches["rno"] == rno, "rtp"].values[0] - gw_head[0, z, y, x]

        file = base_path / "output" / f"dmn_run_{model_run}_sfr_.csv"
        df_sfr.to_csv(file, sep=";")

        # create xarray dataset
        attrs = dict(
                date_created=datetime.datetime.today().isoformat(),
                title="MODFLOW steady-state simulations of the Dreisam-Moehlin-Neumagen catchment",
                institution="University of Freiburg, Chair of Hydrology",
                flopy_version=f"{flopy.__version__}",
                modflow_version=f"{ml.version}",
            )
        coords = {
                "lon": ("lon", xcoords),  # x
                "lat": ("lat", ycoords),  # y
                "layer": ("layer", nlayers),
                "Time": ("Time", [1]),
            }
        data_vars=dict(
                head=(["Time", "layer", "lat", "lon"], np.where(hds.get_data()[np.newaxis, :, :, :] > 10000, np.nan, hds.get_data()[np.newaxis, :, :, :])),
                depth=(["Time", "layer", "lat", "lon"], np.where(hds.get_data()[np.newaxis, :, :, :] > 10000, np.nan, np.where(topography[np.newaxis, np.newaxis, :, :] - hds.get_data()[np.newaxis, :, :, :] > 0, topography[np.newaxis, np.newaxis, :, :] - hds.get_data()[np.newaxis, :, :, :], 0))),
                flow_residual=(["Time", "layer", "lat", "lon"], residual[np.newaxis, :, :, :]),
                gw_sw=(["Time", "layer", "lat", "lon"], cbb.get_data(text="SFR", kstpkper=(0, 0), full3D=True)[0].filled(fill_value=np.nan)[np.newaxis, :, :, :]),
                gw_sw_=(["Time", "lat", "lon"], np.nansum(cbb.get_data(text="SFR", kstpkper=(0, 0), full3D=True)[0].filled(fill_value=np.nan), axis=0)[np.newaxis, :, :]),
                sfr_stage=(["lat", "lon"], sfr_stage),
                sfr_head=(["lat", "lon"], sfr_head),
                sfr_flow=(["lat", "lon"], sfr_flow),
                sfr_width=(["lat", "lon"], rwid),
                sfr_manning_coefficient=(["lat", "lon"], man),
                sfr_hydraulic_conductivity=(["lat", "lon"], rhk),
                sfr_gradient=(["lat", "lon"], rgrd),
                delta_sw_gw_head=(["lat", "lon"], delta_sw_gw_head),
                delta_rtp_gw_head=(["lat", "lon"], delta_rtp_gw_head)
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        ds["head"].attrs["units"] = "m a.s.l."
        ds["head"].attrs["long_name"] = "Groundwater head"
        ds["depth"].attrs["units"] = "m"
        ds["depth"].attrs["long_name"] = "Groundwater depth"
        ds["flow_residual"].attrs["units"] = "m/day"
        ds["flow_residual"].attrs["long_name"] = "Flow residuals"
        ds["gw_sw"].attrs["units"] = "m3/day"
        ds["gw_sw"].attrs["long_name"] = "Groundwater-surface water flux"
        ds["gw_sw_"].attrs["units"] = "m3/day"
        ds["gw_sw_"].attrs["long_name"] = "Groundwater-surface water flux"
        ds["sfr_stage"].attrs["units"] = "m"
        ds["sfr_stage"].attrs["long_name"] = "Streamflow depth"
        ds["sfr_head"].attrs["units"] = "m a.s.l."
        ds["sfr_head"].attrs["long_name"] = "Streamflow head"
        ds["sfr_flow"].attrs["units"] = "m3/s"
        ds["sfr_flow"].attrs["long_name"] = "Streamflow"
        ds["sfr_width"].attrs["units"] = "m"
        ds["sfr_width"].attrs["long_name"] = "Reach width"
        ds["sfr_manning_coefficient"].attrs["units"] = ""
        ds["sfr_manning_coefficient"].attrs["long_name"] = "Reach Manning's roughness coefficient"
        ds["sfr_hydraulic_conductivity"].attrs["units"] = "m/s"
        ds["sfr_hydraulic_conductivity"].attrs["long_name"] = "Reach hydraulic conductivity"
        ds["sfr_gradient"].attrs["units"] = ""
        ds["sfr_gradient"].attrs["long_name"] = "Reach gradient"
        ds["delta_sw_gw_head"].attrs["units"] = "m"
        ds["delta_sw_gw_head"].attrs["long_name"] = "Difference between surface water and groundwater head"
        ds["delta_rtp_gw_head"].attrs["units"] = "m"
        ds["delta_rtp_gw_head"].attrs["long_name"] = "Difference between elevation of the riverbed and groundwater head"

        # create spatial reference
        ds = ds.geo.write_crs("EPSG:25832")
        ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
        file = base_path / "output" / f"modflow_output_run_{model_run}.nc"
        comp = dict(zlib=True, complevel=1)  # compress data to save storage
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)

        # # write to csv
        # if "steady-state" == "steady-state":
        #     hds_data = hds.get_data()
        #     for i in range(4):
        #         file = base_path / "output" / "steady-state" / f"groundwater_heads_layer{i+1}.csv"
        #         hds_data_layer = hds_data[i, ...]
        #         mask = (hds_data_layer > 1200) | (hds_data_layer < -100)
        #         hds_data_layer[mask] = np.nan
        #         np.savetxt(file, hds_data_layer, delimiter=";")

        # except:
        #     pass
    return


if __name__ == "__main__":
    main()