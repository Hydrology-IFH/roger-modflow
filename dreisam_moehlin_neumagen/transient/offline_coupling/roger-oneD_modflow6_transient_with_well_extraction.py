from time import time
from pathlib import Path
import os
import numpy as np
from xmipy import XmiWrapper
import flopy
from flopy.utils import Raster
import platform
import yaml
import xarray as xr
import xesmf as xe
import pandas as pd
import scipy
import shutil
import click

base_path = Path(__file__).parent

file_config = base_path.parent / "config_modflow.yml"
with open(file_config, "r") as file:
    config_modflow = yaml.safe_load(file)

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


def aggregate_to_coarser_resolution(vals, res_fine, res_coarse, method="sum", x_origin=0, y_origin=0):
    """Aggregate raster data to a coarser resolution.
    
    Args
    ----
    vals : numpy.ndarray
        2D array with the raster data.

    res_fine : int
        spatial resolution of the fine grid in meters.

    res_coarse : int
        spatial resolution of the coarse grid in meters.

    method : str
        Method to aggregate the data. Options are "sum" and "average".

    x_origin : int
        x-coordinate of the grid origin (lower left corner).

    y_origin : int
        y-coordinate of the grid origin (lower left corner).  
    """
    ny_fine, nx_fine = vals.shape[0], vals.shape[1]
    nlat_coarse, nlon_coarse = int(res_coarse / res_fine), int(res_coarse / res_fine)
    meters_to_latlon = 111195
    lat_fine = np.linspace(y_origin, y_origin + ny_fine*(res_fine/meters_to_latlon), ny_fine)/meters_to_latlon  # boundaries
    lon_fine = np.linspace(x_origin, x_origin + nx_fine*(res_fine/meters_to_latlon), nx_fine)/meters_to_latlon  # boundaries

    arr_fine = xr.DataArray(vals, coords={"lat": lat_fine, "lon": lon_fine}, dims=["lat", "lon"])

    if method == "sum":
        arr_coarse = xr.DataArray(
            arr_fine.coarsen(lat=nlat_coarse, lon=nlon_coarse).sum().values,
            dims=("lat", "lon"),
        )
        
    elif method == "average":
        arr_coarse = xr.DataArray(
            arr_fine.coarsen(lat=nlat_coarse, lon=nlon_coarse).mean().values,
            dims=("lat", "lon"),
        )
    return arr_coarse.values


def aggregate_to_finer_resolution(vals, res_coarse, res_fine, method="keep", x_origin=0, y_origin=0):
    """Aggregate raster data to a finer resolution.
        
    Args
    ----
    vals : numpy.ndarray
        2D array with the raster data.

    res_coarse : int
        spatial resolution of the coarse grid in meters.

    res_fine : int
        spatial resolution of the fine grid in meters.

    method : str
        Method to aggregate the data. Options are "keep", "interpolate" and "conservative".

    x_origin : int
        x-coordinate of the grid origin (lower left corner).

    y_origin : int
        y-coordinate of the grid origin (lower left corner).  
    """
    ny_coarse, nx_coarse = vals.shape[0], vals.shape[1]
    nx_fine = int(nx_coarse * (res_coarse / res_fine))
    ny_fine = int(ny_coarse * (res_coarse / res_fine))
    meters_to_latlon = 111195
    if method == "keep":
        lat_fine = np.linspace(y_origin, y_origin + ny_fine*(res_fine/meters_to_latlon), ny_fine)/meters_to_latlon  # boundaries
        lon_fine = np.linspace(x_origin, x_origin + nx_fine*(res_fine/meters_to_latlon), nx_fine)/meters_to_latlon  # boundaries
        lat_coarse = np.linspace(y_origin, y_origin + ny_coarse*(res_coarse/meters_to_latlon), ny_coarse)/meters_to_latlon  # boundaries
        lon_coarse = np.linspace(x_origin, x_origin + nx_coarse*(res_coarse/meters_to_latlon), nx_coarse)/meters_to_latlon  # boundaries
        grid_coarse = {"lon": lon_coarse, "lat": lat_coarse}
        grid_fine = {"lon": lon_fine, "lat": lat_fine}
        regridder = xe.Regridder(grid_coarse, grid_fine, "nearest_s2d")
    elif method == "interpolate":
        lat_fine = np.linspace(y_origin, y_origin + ny_fine*(res_fine/meters_to_latlon), ny_fine)/meters_to_latlon  # boundaries
        lon_fine = np.linspace(x_origin, x_origin + nx_fine*(res_fine/meters_to_latlon), nx_fine)/meters_to_latlon  # boundaries
        lat_coarse = np.linspace(y_origin, y_origin + ny_coarse*(res_coarse/meters_to_latlon), ny_coarse)/meters_to_latlon  # boundaries
        lon_coarse = np.linspace(x_origin, x_origin + nx_coarse*(res_coarse/meters_to_latlon), nx_coarse)/meters_to_latlon  # boundaries
        grid_coarse = {"lon": lon_coarse, "lat": lat_coarse}
        grid_fine = {"lon": lon_fine, "lat": lat_fine}
        regridder = xe.Regridder(grid_coarse, grid_fine, "bilinear")
    elif method == "conservative":
        lat_fine = np.linspace(y_origin, y_origin + (ny_fine + 1) * (res_fine/meters_to_latlon), ny_fine + 1)/meters_to_latlon  # boundaries
        lon_fine = np.linspace(x_origin, x_origin + (nx_fine + 1) * (res_fine/meters_to_latlon), nx_fine + 1)/meters_to_latlon  # boundaries
        lat_coarse = np.linspace(y_origin, y_origin + (ny_coarse + 1) * (res_coarse/meters_to_latlon), ny_coarse + 1)/meters_to_latlon  # boundaries
        lon_coarse = np.linspace(x_origin, x_origin + (nx_coarse + 1) * (res_coarse/meters_to_latlon), nx_coarse + 1)/meters_to_latlon  # boundaries
        lat_fine_centers = (0.5 * (lat_fine[1] - lat_fine[0])) + lat_fine[:-1]  # centers
        lat_coarse_centers = (0.5 * (lat_coarse[1] + lat_coarse[0])) + lat_coarse[:-1]  # centers
        lon_fine_centers = (0.5 * (lon_fine[1] + lon_fine[0])) + lon_fine[:-1]  # centers
        lon_coarse_centers = (0.5 * (lon_coarse[1] + lon_coarse[0])) + lon_coarse[:-1]  # centers
        grid_coarse = {"lon": lon_coarse_centers, "lon_b": lon_coarse, "lat": lat_coarse_centers, "lat_b": lat_coarse}
        grid_fine = {"lon": lon_fine_centers, "lon_b": lon_fine, "lat": lat_fine_centers, "lat_b": lat_fine}
        regridder = xe.Regridder(grid_coarse, grid_fine, "conservative")

    data = regridder(vals)
    return data


base_path = Path(__file__).parent

class ModFlowSimulation:
    def __init__(
        self,
        name,
        stress_test_scenario,
        folder,
        time_origin,
        ndays,
        nlay,
        nrow,
        ncol,
        rowsize,
        colsize,
        model_run=1,
        verbose=False
    ):
        self.name = name.upper()  # MODFLOW requires the name to be uppercase
        self.stress_test_scenario = stress_test_scenario
        self.folder = folder
        self.nrow = nrow
        self.ncol = ncol
        self.rowsize = rowsize
        self.colsize = colsize
        self.working_directory = os.path.join(folder, f"output/{stress_test_scenario}")
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.verbose = verbose

        groundwater_extraction = pd.read_csv(base_path.parent / "input" / "groundwater_extraction.csv", sep=";")
        groundwater_extraction["cell_y"] = groundwater_extraction["cell_y"].values - 1
        groundwater_extraction["cell_x"] = groundwater_extraction["cell_x"].values - 1
        groundwater_extraction["layer"] = groundwater_extraction["layer"].values - 1

        # load MODFLOW parameters
        path = Path(__file__).parent.parent / "input" / "parameters_modflow.nc"
        ds_params = xr.open_dataset(path, engine="h5netcdf")

        path = Path(__file__).parent.parent / "input" / "boundary_conditions.nc"
        ds_bc = xr.open_dataset(path, engine="h5netcdf")

        path = Path(__file__).parent.parent / "input" / "initial_conditions.nc"
        ds_ic = xr.open_dataset(path, engine="h5netcdf")

        path = base_path.parent / "fudge_parameters_modflow.csv"
        fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)

        # Temporal discretization (TDIS)
        # One or more models (GWF is the only model supported at present)
        # Zero or more exchanges (instructions for how models are coupled)
        # Solutions
        #
        # For this simple hillslope example, the simulation consists of the temporal discretization (TDIS) package (TDIS), a groundwater flow (GWF) model, and an iterative model solution (IMS), which controls how the GWF model is solved.

        # Create the Flopy simulation object
        sim = flopy.mf6.MFSimulation(
            sim_name=name, exe_name="mf6", version="mf6", sim_ws=self.working_directory,
        )

        # Create the Flopy temporal discretization object
        tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
            sim, pname="tdis", time_units="DAYS", start_date_time=time_origin, nper=2, perioddata=[(0.0, 1.0, 1.0), (float(ndays), int(ndays), 1.0)]
        )

        # Create the Flopy groundwater flow (gwf) model object
        model_nam_file = "{}.nam".format(name)
        gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file, save_flows=True, newtonoptions="NEWTON")

        # Create the Flopy iterative model solver (ims) Package object
        ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", print_option="summary", complexity="COMPLEX", no_ptcrecord="NO_PTC_ALL")

        # Now that the overall simulation is set up, we can focus on building the groundwater flow model.  The groundwater flow model will be built by adding packages to it that describe the model characteristics.
        #
        # Define the discretization of the model. All layers are given equal thickness. The `bot` array is build from `H` and the `Nlay` values to indicate top and bottom of each layer, and `delrow` and `delcol` are computed from model size `L` and number of cells `N`. Once these are all computed, the Discretization file is built.

        # Create the discretization package
        # load elevation data of the layers
        topography = ds_params["elevations"].isel(z=0).values
        elevation_bottom_layer1 = ds_params["elevations"].isel(z=1).values
        elevation_bottom_layer2 = ds_params["elevations"].isel(z=2).values
        elevation_bottom_layer3 = ds_params["elevations"].isel(z=3).values
        elevation_bottom_layer4 = ds_params["elevations"].isel(z=4).values
        elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

        mask = np.isfinite(topography)
        # set Schoenberg to inactive
        mask_schoenberg = (ds_params["mask_schoenberg"].values == 1)
        mask = np.where(mask_schoenberg, False, mask)
        mask_boundary_condition_schoenberg = ds_bc["mask_schoenberg_bc"].values
        mask = np.where(mask_boundary_condition_schoenberg, True, mask)
        mask_drainage_area = (ds_params["mask_drainage"].values == 1)
        domain = np.empty_like(topography)
        domain[mask] = 1
        domain[~mask] = -1
        self.modflow_basin = mask
        self.n_active_cells_per_layer = np.nansum(self.modflow_basin)
        domain_layers = [domain, domain, domain, domain]
        dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
            gwf,
            pname="dis",
            nlay=nlay,
            nrow=self.nrow,
            ncol=self.ncol,
            delr=self.rowsize, 
            delc=self.colsize,
            length_units="METERS",
            top=topography,
            botm=elevation_bottom_layers,
            idomain=domain_layers,
        )

        # Create the initial conditions package
        # use interpolated groundwater head values at start date as initial conditions
        gw_heads_interpolated = ds_params["gw_heads_interpolated"].values - 2.0
        gw_heads_interpolated[~mask] = np.nan
        gw_heads_interpolated[gw_heads_interpolated > topography] = topography[gw_heads_interpolated > topography]
        initial_conditions_layers = [gw_heads_interpolated, gw_heads_interpolated, gw_heads_interpolated, gw_heads_interpolated]
        ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=initial_conditions_layers)

        # initial_conditions_layer1 = ds_ic['initial_head'].values
        # initial_conditions_layer2 = ds_ic['initial_head'].values
        # initial_conditions_layer3 = ds_ic['initial_head'].values
        # initial_conditions_layer4 = ds_ic['initial_head'].values
        # initial_conditions_layer1[~mask] = np.nan
        # initial_conditions_layer2[~mask] = np.nan
        # initial_conditions_layer3[~mask] = np.nan
        # initial_conditions_layer4[~mask] = np.nan
        # initial_conditions_layers = [initial_conditions_layer1, initial_conditions_layer2, initial_conditions_layer3, initial_conditions_layer4]
        # ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=initial_conditions_layers)

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

        # constrain hydraulic conductivities to a reasonable range to avoid numerical instabilities
        hydraulic_conductivities_layer1[hydraulic_conductivities_layer1 > 1000] = 1000
        hydraulic_conductivities_layer2[hydraulic_conductivities_layer2 > 1000] = 1000
        hydraulic_conductivities_layer3[hydraulic_conductivities_layer3 > 1000] = 1000
        hydraulic_conductivities_layer4[hydraulic_conductivities_layer4 > 1000] = 1000
        hydraulic_conductivities_layer1[hydraulic_conductivities_layer1 < 10e-6] = 10e-6
        hydraulic_conductivities_layer2[hydraulic_conductivities_layer2 < 10e-6] = 10e-6
        hydraulic_conductivities_layer3[hydraulic_conductivities_layer3 < 10e-6] = 10e-6
        hydraulic_conductivities_layer4[hydraulic_conductivities_layer4 < 10e-6] = 10e-6

        # prepare SFR data
        reaches = pd.read_csv(base_path.parent / "input" / "sfr_packagedata_modified.csv", sep=";")
        reaches.iloc[:, 0] = reaches.iloc[:, 0].astype(int) - 1  # convert to zero-based indexing
        reaches.iloc[:, 1] = reaches.iloc[:, 1].astype(int) - 1 # convert to zero-based indexing
        reaches.iloc[:, 2] = reaches.iloc[:, 2].astype(int) - 1  # convert to zero-based indexing
        reaches.iloc[:, 3] = reaches.iloc[:, 3].astype(int) - 1  # convert to zero-based indexing
        reaches.iloc[:, 4] = reaches.iloc[:, 4].astype(float) 
        reaches.iloc[:, 5] = reaches.iloc[:, 5].astype(int)
        reaches.iloc[:, 6] = reaches.iloc[:, 6].astype(float)
        reaches.iloc[:, 7] = reaches.iloc[:, 7].astype(float)
        reaches.iloc[:, 8] = reaches.iloc[:, 8].astype(float)
        reaches.iloc[:, 9] = reaches.iloc[:, 9].astype(float) * 86400  # convert from m/s to m/day
        reaches.iloc[:, 10] = reaches.iloc[:, 10].astype(float)
        reaches.iloc[:, 11] = reaches.iloc[:, 11].astype(int)
        reaches.iloc[:, 12] = reaches.iloc[:, 12].astype(float)
        reaches.iloc[:, 13] = reaches.iloc[:, 13].astype(int)
        reaches.iloc[:, 15] = reaches.iloc[:, 15].astype(float)
        reaches.iloc[:, 16] = reaches.iloc[:, 16].astype(int)

        # smooth transition between fissured and porous aquifers
        hydraulic_conductivities_layer1[np.isnan(hydraulic_conductivities_layer1)] = 0
        hydraulic_conductivities_layer2[np.isnan(hydraulic_conductivities_layer2)] = 0
        hydraulic_conductivities_layer3[np.isnan(hydraulic_conductivities_layer3)] = 0
        hydraulic_conductivities_layer4[np.isnan(hydraulic_conductivities_layer4)] = 0
        _hydraulic_conductivities_layer1 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer1, [1.5, 1.5],  mode="constant")
        _hydraulic_conductivities_layer2 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer2, [1.5, 1.5],  mode="constant")
        _hydraulic_conductivities_layer3 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer3, [1.5, 1.5],  mode="constant")
        _hydraulic_conductivities_layer4 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer4, [1.5, 1.5],  mode="constant")
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

        # fudge streambed conductivity
        cond = (reaches["kf"] >= 10e-6)
        reaches.loc[cond, "rhk"] = reaches.loc[cond, "rhk"] * fudge_parameters["rhkp"].values[model_run]
        cond = (reaches["kf"] < 10e-6)
        reaches.loc[cond, "rhk"] = reaches.loc[cond, "rhk"] * fudge_parameters["rhkf"].values[model_run]
        reaches["man"] = reaches["man"] * fudge_parameters["man"].values[model_run]

        # modify the manning"s n and hydraulic conductivity of the streambed based on the degree of alteration (5=partly, 6=strongly, 7=very strongly)
        cond = (reaches["ss"] == 5)
        reaches.loc[cond, "rhk"] = 50e-7 * 86400
        cond = (reaches["ss"] == 6)
        reaches.loc[cond, "rhk"] = 10e-9 * 86400
        cond = (reaches["ss"] == 7)
        reaches.loc[cond, "rhk"] = 50e-10 * 86400

        diversions = pd.read_csv(base_path.parent / "input" / "sfr_diversions.csv", sep=";")
        diversions.iloc[:, 0] = diversions.iloc[:, 0].astype(int) - 1  # convert to zero-based indexing
        diversions.iloc[:, 1] = diversions.iloc[:, 1].astype(int) - 1
        diversions.iloc[:, 2] = diversions.iloc[:, 2].astype(int) - 1  # convert to zero-based indexing
        diversions.iloc[:, 3] = diversions.iloc[:, 3].astype(str)
        diversiondata = []
        for i in range(len(diversions)):
            diversion = diversions.iloc[i, :].to_list()
            diversiondata.append(diversion)

        # condition of diversions
        cond_diversions = reaches.iloc[:, 0].isin(diversions.iloc[:, 0])
        reaches.loc[cond_diversions, "ustrf"] = 1.0
        reaches.loc[cond_diversions, "ndv"] = 1
        reaches.loc[cond_diversions, "ncon"] = reaches.loc[cond_diversions, "ncon"] + 1
        cond_diversions1 = reaches.iloc[:, 0].isin(diversions.iloc[:, 2])
        reaches.loc[cond_diversions1, "ustrf"] = 0.0
        reaches.loc[cond_diversions1, "ncon"] = reaches.loc[cond_diversions1, "ncon"] + 1

        packagedata = []
        for i in range(len(reaches)):
            reach = []
            for j in range(len(reaches.columns)-3):
                reach.append(reaches.iloc[i, j])
            packagedata.append(reach)
        
        nstrm = len(packagedata)  # number of reaches

        connections = pd.read_csv(base_path.parent / "input" / "sfr_connectiondata.csv", sep=";", header=None)
        for i in range(len(diversions)):
            rno_up = diversions.iloc[i, 0]
            rno_down = diversions.iloc[i, 2] 
            connections.iloc[rno_up, -1] = -(diversions.iloc[i, 2] + 1) # set diversion number
            connections.iloc[rno_down, -1] = diversions.iloc[i, 0] + 1

        connectiondata = []
        for i in range(len(connections)):
            _connection = []
            connection = connections.iloc[i, :].values
            cond = np.isfinite(connection)
            connection = connection[cond]
            cond_pos = (connection > 0)
            connection_up = connection[cond_pos] - 1  # convert to zero-based indexing
            _connection.extend(connection_up.astype(int))
            cond_neg = (connection < 0)
            connection_down = connection[cond_neg] + 1  # convert to zero-based indexing
            _connection.extend(connection_down.astype(int))
            connectiondata.append(_connection)

        hydraulic_conductivities_layers = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]
        npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
            gwf, pname="npf", icelltype=1, k=hydraulic_conductivities_layers, wetdry=0.5, save_flows=True, save_specific_discharge="budget save file"
        )

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
        _specific_yield_layer1 = scipy.ndimage.gaussian_filter(specific_yield_layer1, [1.5, 1.5],  mode="constant")
        _specific_yield_layer2 = scipy.ndimage.gaussian_filter(specific_yield_layer2, [1.5, 1.5],  mode="constant")
        _specific_yield_layer3 = scipy.ndimage.gaussian_filter(specific_yield_layer3, [1.5, 1.5],  mode="constant")
        _specific_yield_layer4 = scipy.ndimage.gaussian_filter(specific_yield_layer4, [1.5, 1.5],  mode="constant")
        specific_yield_layer1[cond1] = _specific_yield_layer1[cond1]
        specific_yield_layer2[cond2] = _specific_yield_layer2[cond2]
        specific_yield_layer3[cond3] = _specific_yield_layer3[cond3]
        specific_yield_layer4[cond4] = _specific_yield_layer4[cond4]
        specific_yield_layer1[~mask] = np.nan
        specific_yield_layer2[~mask] = np.nan
        specific_yield_layer3[~mask] = np.nan
        specific_yield_layer4[~mask] = np.nan
        specific_yield = flopy.mf6.ModflowGwfsto.sy.empty(gwf, layered=True)
        specific_yield[0]["data"] = specific_yield_layer1
        specific_yield[1]["data"] = specific_yield_layer2
        specific_yield[2]["data"] = specific_yield_layer3
        specific_yield[3]["data"] = specific_yield_layer4

        specific_storage = flopy.mf6.ModflowGwfsto.ss.empty(
            gwf, layered=True
        )
        thickness_layer1 = topography - elevation_bottom_layer1
        thickness_layer2 = elevation_bottom_layer1 - elevation_bottom_layer2
        thickness_layer3 = elevation_bottom_layer2 - elevation_bottom_layer3
        thickness_layer4 = elevation_bottom_layer3 - elevation_bottom_layer4
        specific_storage_layer1 = specific_yield_layer1 * thickness_layer1
        specific_storage_layer2 = specific_yield_layer2 * thickness_layer2
        specific_storage_layer3 = specific_yield_layer3 * thickness_layer3
        specific_storage_layer4 = specific_yield_layer4 * thickness_layer4
        specific_storage[0]["data"] = specific_storage_layer1
        specific_storage[1]["data"] = specific_storage_layer2
        specific_storage[2]["data"] = specific_storage_layer3
        specific_storage[3]["data"] = specific_storage_layer4

        sto = flopy.mf6.ModflowGwfsto(gwf, pname="sto",
            iconvert=1, ss=specific_storage, sy=specific_yield, steady_state={0: True}, transient={1: True}, save_flows=False)

        # Create the constant head package (Dirichlet boundary condition i.e. first type)
        mask_boundary_condition_porous_aquifer = ds_bc["mask_porous_aquifer_bc"].values
        index = np.where(mask_boundary_condition_porous_aquifer == 1)
        rows_bc = index[0]
        cols_bc = index[1]

        chd_rec = []
        for ii in range(0, len(rows_bc)):
            constant_head = ds_bc["constant_head_porous_aquifer"].values[rows_bc[ii], cols_bc[ii]] - fudge_parameters["offset"].values[model_run]
            if (constant_head <= topography[rows_bc[ii], cols_bc[ii]]) and (constant_head > elevation_bottom_layer1[rows_bc[ii], cols_bc[ii]]):
                layer = 0
            elif (constant_head <= elevation_bottom_layer1[rows_bc[ii], cols_bc[ii]]) and (constant_head > elevation_bottom_layer2[rows_bc[ii], cols_bc[ii]]):
                layer = 1
            elif (constant_head <= elevation_bottom_layer2[rows_bc[ii], cols_bc[ii]]) and (constant_head > elevation_bottom_layer3[rows_bc[ii], cols_bc[ii]]):
                layer = 2
            elif (constant_head <= elevation_bottom_layer3[rows_bc[ii], cols_bc[ii]]) and (constant_head > elevation_bottom_layer4[rows_bc[ii], cols_bc[ii]]):
                layer = 3
            chd_rec.append(((layer, rows_bc[ii], cols_bc[ii]), constant_head))

        chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
            gwf,
            pname="chd",
            maxbound=len(chd_rec),
            stress_period_data=chd_rec,
            save_flows=True,
        )
            
        # Recharge package (Neumann boundary condition i.e. second type)
        recharge = np.zeros((self.modflow_basin.sum(), 4), dtype=np.int32)
        recharge_locations = np.where(self.modflow_basin == True)  # only set recharge where modflow_basin is True
        # 0: layer, 1: y-idx, 2: x-idx, 3: rate
        recharge[:, 1] = recharge_locations[0]
        recharge[:, 2] = recharge_locations[1]
        recharge = recharge.tolist()

        recharge = flopy.mf6.ModflowGwfrch(gwf, fixed_cell=False,
                            print_input=False, print_flows=False,
                            save_flows=False, boundnames=None,
                            maxbound=self.modflow_basin.sum(), stress_period_data=recharge)
        
        # Evapotranspiration package (Neumann boundary condition i.e. second type) to represent capillary fringe and ectraction of irrigation water by plants
        cpr_irr_spd = np.zeros((self.modflow_basin.sum(), 9), dtype=np.int32)
        cpr_irr_locations = np.where(self.modflow_basin == True)  # only set recharge where modflow_basin is True
        # 0: layer, 1: y-idx, 2: x-idx, 3: rate
        cpr_irr_spd[:, 1] = cpr_irr_locations[0]
        cpr_irr_spd[:, 2] = cpr_irr_locations[1]
        cpr_irr_spd = cpr_irr_spd.tolist()
        cpr_irr = flopy.mf6.ModflowGwfevt(gwf, fixed_cell=True,
                                          print_input=False, print_flows=False,
                                          save_flows=True, boundnames=None,
                                          maxbound=self.modflow_basin.sum(), stress_period_data=cpr_irr_spd)

        # streamflow routing package (SFR)
        ls_obs = [(str(key), str(config_modflow["sfr_obs"][key][0]), (int(config_modflow["sfr_obs"][key][1]),)) for key in config_modflow["sfr_obs"].keys()]
        obs_dict = {
            (f"{name}_sfr.obs.csv", "binary"): ls_obs
        }
        sfr = flopy.mf6.modflow.mfgwfsfr.ModflowGwfsfr(gwf, pname="sfr",
            time_conversion=86400, length_conversion=1.0, nreaches=nstrm, packagedata=packagedata, 
            connectiondata=connectiondata, diversions=diversiondata, save_flows=True,
            maximum_depth_change=0.001, maximum_iterations=500, observations=obs_dict)
        # Create the drainage package (Neumann boundary condition i.e. second type)
        for x, y in zip(reaches.iloc[:, 2], reaches.iloc[:, 3]):
            if mask_drainage_area[x, y]:
                mask_drainage_area[x, y] = False  # set drainage area to inactive if there is a reach

        index = np.where(mask_drainage_area)
        rows_drainage = index[0]
        cols_drainage = index[1]
        drn_spd = []
        for ii in range(0, len(rows_drainage)):
            elev_drn = topography[rows_drainage[ii], cols_drainage[ii]] - 0.5 * (topography[rows_drainage[ii], cols_drainage[ii]] - elevation_bottom_layer1[rows_drainage[ii], cols_drainage[ii]])
            slope = 0.01
            length = 50
            drainage_area = 0.3**2 * np.pi  # drainage pipe with 0.3 m diameter per grid cell
            kf = 0.1 * 86400
            conductance = kf * drainage_area * length * slope
            drn_spd.append(((0, rows_drainage[ii], cols_drainage[ii]), elev_drn, conductance))

        drn = flopy.mf6.ModflowGwfdrn(
            gwf,
            pname="drn",
            maxbound=len(drn_spd),
            boundnames=False,
            mover=False,
            stress_period_data=drn_spd,
        )

        # Well package
        groundwater_extraction = pd.read_csv(base_path.parent / "input" / "groundwater_extraction.csv", sep=";")

        n_wells = len(groundwater_extraction)
        wells = np.zeros((n_wells, 4), dtype=np.int32)
        # 0: layer, 1: y-idx, 2: x-idx, 3: rate
        wells[:, 0] = groundwater_extraction["layer"].values.astype(np.int32) - 1
        wells[:, 1] = groundwater_extraction["cell_y"].values.astype(np.int32) - 1
        wells[:, 2] = groundwater_extraction["cell_x"].values.astype(np.int32) - 1
        wells = wells.tolist()

        wells = flopy.mf6.ModflowGwfwel(gwf, print_input=False, print_flows=False, save_flows=False,
                                    maxbound=n_wells, stress_period_data=wells,
                                    boundnames=False, auto_flow_reduce=0.1)

        # Create the output control package
        headfile = "{}.hds".format(name)
        head_filerecord = [headfile]
        budgetfile = "{}.cbc".format(name)
        budget_filerecord = [budgetfile]
        saverecord = [("HEAD", "FIRST"), ("HEAD", "FREQUENCY", 1), ("HEAD", "LAST"), ("BUDGET", "FIRST"), ("BUDGET", "FREQUENCY", 1), ("BUDGET", "LAST")]
        oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(
            gwf,
            pname="oc",
            saverecord=saverecord,
            head_filerecord=head_filerecord,
            budget_filerecord=budget_filerecord,
        )

        # Create the MODFLOW 6 Input Files and Run the Model
        # Once all the flopy objects are created, it is very easy to create all of the input files and run the model.
        sim.write_simulation()  # write the MODFLOW6 files

        self.load_bmi()

    def bmi_return(self, success, model_ws):
        """
        parse libmf6.so and libmf6.dll stdout file
        """
        fpth = os.path.join(model_ws, "mfsim.stdout")
        if os.path.exists(fpth):
            lines = open(fpth).readlines()
        else:
            lines = None
        return success, lines

    def load_bmi(self):
        """Load the Basic Model Interface"""
        success = False
        
        
        if platform.system() == "Windows":
            libary_name = "libmf6.dll"
        elif platform.system() == "Linux":
            libary_name = "libmf6.so"
        elif platform.system() == "Darwin":
            libary_name = "libmf6.dylib"
        else:
            raise ValueError(f"Platform {platform.system()} not recognized.")

        # modflow requires the real path (no symlinks etc.)
        library_path = self.folder.parent.parent.parent / "bin" / libary_name
        try:
            self.mf6 = XmiWrapper(str(library_path), working_directory=self.working_directory)
        except Exception as e:
            click.echo(f"Failed to load {library_path}")
            click.echo("with message: " + str(e))
            return self.bmi_return(success, self.working_directory)

        # modflow requires the real path (no symlinks etc.)
        config_file = self.folder / "output" / self.stress_test_scenario / "mfsim.nam"
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found on disk. Did you create the model first (load_from_disk = False)?")

        try:
            # initialize the model
            self.mf6.initialize(str(config_file))
        except:
            return self.bmi_return(success, str(self.folder / "output"))

        if self.verbose:
            click.echo("MODFLOW model initialized")

        
        self.end_time = self.mf6.get_end_time()

        mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
        self.max_iter = self.mf6.get_value_ptr(mxit_tag)[0]

        self.prepare_time_step()

    def prepare_time_step(self):
        dt = self.mf6.get_time_step()
        self.mf6.prepare_time_step(dt)

    def set_recharge(self, recharge):
        """Set recharge, value in m/day"""
        recharge_tag = self.mf6.get_var_address("RECHARGE", self.name, "RCH_0")
        mask = self.modflow_basin.flatten()  # mask of active cells
        self.mf6.set_value(recharge_tag, recharge[mask])

    def set_cpr_irr_rate(self, cpr_irr):
        """Set capillary rise (+ extracted irrigation water), value in m/day"""
        cpr_tag = self.mf6.get_var_address("RATE", self.name, "EVT_0")
        self.mf6.set_value(cpr_tag, cpr_irr)

    def set_cpr_irr_surface(self, val):
        """Set elevation of the ET surface, value in m"""
        tag = self.mf6.get_var_address("SURFACE", self.name, "EVT_0")
        self.mf6.set_value(tag, val)

    def set_cpr_irr_depth(self, val):
        """Set ET extinction depth, value in m"""
        tag = self.mf6.get_var_address("DEPTH", self.name, "EVT_0")
        self.mf6.set_value(tag, val)

    def set_well_rate(self, well_rate):
        """Set pumping rate, value in m/day"""
        well_tag = self.mf6.get_var_address("Q", self.name, "WEL_0")
        self.mf6.set_value(well_tag, well_rate)

    def set_sfr_inflow(self, sfr_inflow):
        """Set surface water inflow rate, value in m3/day"""
        sfr_tag = self.mf6.get_var_address("inflow", self.name, "SFR")
        self.mf6.set_value(sfr_tag, sfr_inflow)

    def set_sfr_stage(self, sfr_stage):
        """Set surface water stage, value in m"""
        sfr_tag = self.mf6.get_var_address("stage", self.name, "SFR")
        self.mf6.set_value(sfr_tag, sfr_stage)

    def get_groundwater_head(self, groundwater_head):
        """Get groundwater head from second layer, value in m"""
        head_tag = self.mf6.get_var_address("X", self.name)
        mask = self.modflow_basin.flatten()  # mask of active cells
        groundwater_head[mask] = self.mf6.get_value_ptr(head_tag)[self.n_active_cells_per_layer:self.n_active_cells_per_layer*2]

    def get_recharge(self, recharge):
        """Get recharge, value in m/day"""
        recharge_tag = self.mf6.get_var_address("RECHARGE", self.name, "RCH_0")
        mask = self.modflow_basin.flatten() 
        recharge[mask] = self.mf6.get_value_ptr(recharge_tag)

    def get_well_rate(self, well_rate):
        """Get well rate, value in m/day"""
        well_tag = self.mf6.get_var_address("Q", self.name, "WEL_0")
        well_rate = self.mf6.get_value_ptr(well_tag)

    def get_sfr_inflow(self, sfr_inflow):
        """Get surface water inflow rate, value in m3/day"""
        sfr_tag = self.mf6.get_var_address("inflow", self.name, "SFR")
        sfr_inflow = self.mf6.get_value_ptr(sfr_tag)

    def step(self):
        if self.mf6.get_current_time() > self.end_time:
            raise StopIteration("MODFLOW used all iteration steps. Consider increasing `ndays`")

        t0 = time()
        # loop over subcomponents
        n_solutions = self.mf6.get_subcomponent_count()
        for solution_id in range(1, n_solutions + 1):

            # convergence loop
            kiter = 0
            self.mf6.prepare_solve(solution_id)
            while kiter < self.max_iter:
                has_converged = self.mf6.solve(solution_id)
                kiter += 1

                if has_converged:
                    break

            self.mf6.finalize_solve(solution_id)

        self.mf6.finalize_time_step()

        if self.verbose:
            click.echo(f'MODFLOW timestep {int(self.mf6.get_current_time())} converged in {round(time() - t0, 2)} seconds')
        
        # If next step exists, prepare timestep. Otherwise the data set through the bmi
        # will be overwritten when preparing the next timestep.
        if self.mf6.get_current_time() < self.end_time:
            self.prepare_time_step()

    def finalize(self):
        self.mf6.finalize()

@click.option("-stm", "--stress-test-meteo", type=click.Choice(["base", "base_2000-2024", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]), default="base", help="Type of meteorological stress test")
@click.option("-stmm", "--stress-test-meteo-magnitude", type=click.Choice([0, 1, 2]), default=0, help="Magnitude of meteorological stress test")
@click.option("-stmd", "--stress-test-meteo-duration", type=click.Choice([0, 2, 3]), default=0, help="Duration of meteorological stress test in consecutive years")
@click.option("-irr", "--irrigation", type=click.Choice(["no-irrigation", "irrigation"]), default="no-irrigation", help="Enable irrigation")
@click.option("-ym", "--yellow-mustard", type=click.Choice(["no-yellow-mustard", "yellow-mustard"]), default="no-yellow-mustard", help="Enable catch crop using yellow mustard")
@click.option("-sc", "--soil-compaction", type=click.Choice(["no-soil-compaction", "soil-compaction"]), default="soil-compaction", help="Enable soil compaction")
@click.option("-gco", "--grain-corn-only", type=click.Choice(["no-grain-corn-only", "grain-corn-only"]), default="no-grain-corn-only", help="Enable grain corn monoculture (no crop rotation)")
@click.option("-stwe", "--stress-test-well-extraction", type=click.Choice(["no-stress", "ta-dependent-20", "ta-dependent-40"]), default="no-stress", help="Enable stress test for well extraction")
@click.command("main", short_help="Run MODFLOW in transient mode coupled with RoGeR.")
def main(stress_test_meteo, stress_test_meteo_magnitude, stress_test_meteo_duration, irrigation, yellow_mustard, soil_compaction, grain_corn_only, stress_test_well_extraction):
    if stress_test_meteo == "base_2000-2024":
        time_origin = "2000-01-01"
        date_time = pd.date_range(start="2000-01-01", end="2024-12-31", freq="D")
    else:
        time_origin = "2013-01-01"
        date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")

    if grain_corn_only == "no-grain-corn-only":
        _grain_corn_only = ""
    else:
        _grain_corn_only = "_grain-corn-only"
    
    # load the soil depth from the RoGeR parameters file
    ds_roger_parameters = xr.open_dataset(base_path.parent / "input" / "parameters_roger.nc")  
    soildepth = ds_roger_parameters["GRUND"].values.flatten() / 100  # convert from cm to m

    # load the topography of the model domain and aggregate it to the resolution of RoGeR
    path = Path(__file__).parent.parent / "input" / "parameters_modflow.nc"
    with xr.open_dataset(path, engine="h5netcdf") as ds_params_modflow:
        topography_ = ds_params_modflow["elevations"].values[0, :, :]  # m a.s.l.
        topography = aggregate_to_finer_resolution(topography_, config_modflow['dx'], 25, method="keep")

    # get number of reaches from the modified sfr_packagedata.csv
    sfr_packagedata = pd.read_csv(base_path.parent / "input" / "sfr_packagedata_modified.csv", sep=";")
    n_reaches = sfr_packagedata.shape[0]

    sfr_connections = pd.read_csv(base_path.parent / "input" / "sfr_connectiondata.csv", sep=";", header=None)
    # get rows if last three columns have nan values (i.e. no connections and no diversions)
    source_rnos = sfr_connections[sfr_connections.iloc[:, -3:].isna().all(axis=1)].iloc[:, 0].tolist()

    # load groundwater extraction data
    groundwater_extraction = pd.read_csv(base_path.parent / "input" / "groundwater_extraction.csv", sep=";")
    groundwater_extraction["cell_y"] = groundwater_extraction["cell_y"].astype(int)
    groundwater_extraction["cell_x"] = groundwater_extraction["cell_x"].astype(int)
    groundwater_extraction["layer"] = groundwater_extraction["layer"].astype(int)
    groundwater_extraction["purpose"] = groundwater_extraction["purpose"].astype(str)
    groundwater_extraction["cell_y"] = groundwater_extraction["cell_y"].values - 1
    groundwater_extraction["cell_x"] = groundwater_extraction["cell_x"].values - 1
    groundwater_extraction["layer"] = groundwater_extraction["layer"].values - 1
    n_wells = len(groundwater_extraction)
    cond_drinking_water_supply = groundwater_extraction["purpose"].isin(['Badenova WW Ebnet', 'Badenova WW Hausen', 'Eigenwasserversorgung', 'oeffentliche Wasserversorgung']).values

    # load daily weights for drinking water supply wells to scale the pumping rates of the drinking water supply wells in the well package
    daily_weights_drinking_water_supply = pd.read_csv(base_path.parent / "input" / "daily_weights_drinking_water_supply.csv", sep=";", index_col=0)

    # get number of days in the simulation which also used as number of time steps in MODFLOW
    NDAYS = len(date_time)
    doys = date_time.dayofyear.values
    years = date_time.year.values

    # initialize the MODFLOW model using XMI
    modflow_interface = ModFlowSimulation(
        "dmn_run_944",
        f"{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}",
        base_path,
        time_origin=time_origin,
        ndays=float(NDAYS),
        nlay=4,
        nrow=config_modflow["ny"],
        ncol=config_modflow["nx"],
        rowsize=config_modflow["dx"],
        colsize=config_modflow["dy"],
        model_run=944,
        verbose=True
    )

    # initialize the model running in steady-state mode
    year = years[0]
    doy = doys[0]
    daily_weights_drinking_water_supply_year_doy = daily_weights_drinking_water_supply.loc[int(year), f"{int(doy)}"]

    # load recharge data of the current year
    file = f"recharge_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
    path = Path(__file__).parent.parent / "input" / file
    with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as ds_recharge:
        recharge_year = ds_recharge["recharge"].values
        recharge_year[recharge_year < 0] = 0  # set negative recharge to zero

    # load capillary rise data of the current year
    file = f"capillary_rise_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
    path = Path(__file__).parent.parent / "input" / file
    with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as ds_capillary_rise:
        capillary_rise_year = ds_capillary_rise["capillary_rise"].values
        capillary_rise_year[capillary_rise_year < 0] = 0  # set negative capillary rise to zero
    
    if irrigation == "irrigation":
        # load irrigation data of the current year
        file = f"irrigation_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
        path = Path(__file__).parent.parent / "input" / file
        with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as ds_irrigation:
            irrigation_year = ds_irrigation["irrigation"].values
            irrigation_year[irrigation_year < 0] = 0  # set negative irrigation to zero

    # update groundwater head
    groundwater_head = np.zeros(config_modflow['ny'] * config_modflow['nx'])
    modflow_interface.get_groundwater_head(groundwater_head)
    groundwater_head = groundwater_head.reshape(config_modflow['ny'], config_modflow['nx'])
    click.echo(groundwater_head[214, 450])
    # aggregate groundwater head to the resolution of RoGeR
    groundwater_head = aggregate_to_finer_resolution(groundwater_head, config_modflow['dx'], 25, method="keep")
    # RoGeR requires depth of groundwater head (in meters)
    groundwater_depth = topography.flatten() - groundwater_head.flatten()
    groundwater_depth[(groundwater_depth <= soildepth)] = soildepth[(groundwater_depth <= soildepth)] + 0.05

    # update recharge and pass it to MODFLOW
    recharge_ = recharge_year[1, :, :]
    recharge = recharge_.flatten()
    recharge[(groundwater_depth <= soildepth)] = 0 # constrain recharge to zero where groundwater depth is equal to soil depth
    recharge = recharge.reshape(config_modflow['ny'] * 2, config_modflow['nx'] * 2).astype(np.float64) / 1000  # mm/day to m/day
    recharge_vertical = aggregate_to_coarser_resolution(recharge, 25, config_modflow['dx'], method="average")
    recharge = recharge_vertical.flatten()
    recharge[recharge > 0.1] = 0.1  # constrain recharge to 0.1 m/day
    modflow_interface.set_recharge(recharge)

    # set ET extinction depth to 3 m for the entire model domain
    extinction_depth = np.zeros((len(recharge),), dtype=np.float64) + 3
    modflow_interface.set_cpr_irr_depth(extinction_depth)

    # update well rate and pass it to MODFLOW
    well_extraction_rate = np.zeros((n_wells,), dtype=np.float64)
    well_extraction_rate[:] = groundwater_extraction[f"{year}"].values.astype(np.float64)
    well_extraction_rate[cond_drinking_water_supply] = well_extraction_rate[cond_drinking_water_supply] * daily_weights_drinking_water_supply_year_doy
    well_extraction_rate[~cond_drinking_water_supply] = well_extraction_rate[~cond_drinking_water_supply] / 365.25
    well_extraction_rate[:] = -well_extraction_rate[:]  # extraction is negative
    modflow_interface.set_well_rate(well_extraction_rate)

    # update intital SFR stage of source reaches and pass it to MODFLOW
    sfr_stage = np.zeros((n_reaches,), dtype=np.float64)
    for i, rno in enumerate(source_rnos):
        sfr_stage[rno - 1] = sfr_packagedata.loc[sfr_packagedata["rno"] == rno, "rtp"].values[0] + 0.1

    # Ebnet
    sfr_stage[18598] = 307.8
    # Wiesneck
    sfr_stage[12409] = 432.6
    # Falkensteig
    sfr_stage[9094] = 487.8
    # St. Wilhelm
    sfr_stage[8165] = 542.6
    # Oberried
    sfr_stage[8803] = 432.6
    # Untermuenstertal
    sfr_stage[13128] = 331.85
    # Oberambringen
    sfr_stage[21173] = 242.05
    modflow_interface.set_sfr_stage(sfr_stage)

    # run MODFLOW for one timestep
    modflow_interface.step()
    click.echo("MODFLOW (initial steady-state) finalized")

    # run the transient simulation
    for i in range(NDAYS):
        year = years[i]
        doy = doys[i]

        # load recharge data of the current year
        file = f"recharge_{stress_test_meteo}-magnitude{stress_test_meteo_magnitude}-duration{stress_test_meteo_duration}_{irrigation}_{yellow_mustard}_{soil_compaction}{_grain_corn_only}_year{year}.nc"
        path = Path(__file__).parent.parent / "input" / file
        if i == 0:
            with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as ds_recharge:
                recharge_year = ds_recharge["recharge"].values
                recharge_year[recharge_year < 0] = 0  # set negative recharge to zero
        elif  years[i - 1] < year and i > 0:
            with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as ds_recharge:
                recharge_year = ds_recharge["recharge"].values
                recharge_year[recharge_year < 0] = 0  # set negative recharge to zero
                    
        # update groundwater head
        groundwater_head = np.zeros(config_modflow['ny'] * config_modflow['nx'])
        modflow_interface.get_groundwater_head(groundwater_head)
        groundwater_head = groundwater_head.reshape(config_modflow['ny'], config_modflow['nx'])
        click.echo(groundwater_head[214, 450])
        # aggregate groundwater head to the resolution of RoGeR
        groundwater_head = aggregate_to_finer_resolution(groundwater_head, config_modflow['dx'], 25, method="keep")
        # RoGeR requires depth of groundwater head (in meters)
        groundwater_depth = topography.flatten() - groundwater_head.flatten()
        groundwater_depth[(groundwater_depth <= soildepth)] = soildepth[(groundwater_depth <= soildepth)] + 0.05

        # update recharge and pass it to MODFLOW
        try:
            recharge_ = recharge_year[doy - 1, :, :]
        except IndexError:
            click.echo(f"IndexError: doy {doy} of year {year} is out of bounds for recharge. Setting recharge to zero for this timestep.")
            recharge_ = np.zeros((config_modflow['ny'], config_modflow['nx'])) 
        recharge = recharge_.flatten()
        recharge[(groundwater_depth <= soildepth)] = 0 # constrain recharge to zero where groundwater depth is equal to soil depth
        recharge = recharge.reshape(config_modflow['ny'] * 2, config_modflow['nx'] * 2).astype(np.float64) / 1000  # mm/day to m/day
        recharge_vertical = aggregate_to_coarser_resolution(recharge, 25, config_modflow['dx'], method="average")
        recharge = recharge_vertical.flatten()
        recharge[recharge > 0.1] = 0.1  # constrain recharge to 0.1 m/day
        modflow_interface.set_recharge(recharge)

        # update capillary rise and pass it to MODFLOW
        try:
            capillary_rise_ = capillary_rise_year[doy - 1, :, :]
        except IndexError:
            click.echo(f"IndexError: doy {doy} of year {year} is out of bounds for capillary rise. Setting capillary rise to zero for this timestep.")
            capillary_rise_ = np.zeros((config_modflow['ny'], config_modflow['nx']))
        capillary_rise = capillary_rise_.flatten()
        capillary_rise = capillary_rise.reshape(config_modflow['ny'] * 2, config_modflow['nx'] * 2).astype(np.float64) / 1000  # mm/day to m/day
        capillary_rise = aggregate_to_coarser_resolution(capillary_rise, 25, config_modflow['dx'], method="average")
        capillary_rise[capillary_rise > 0.003] = 0.003  # constrain capillary rise to 0.0031 m/day
        # set ET surface to the current groundwater head for the entire model domain
        modflow_interface.set_cpr_irr_surface(groundwater_head.flatten())

        if irrigation == "irrigation":
            # update irrigation and pass it to MODFLOW as capillary rise (i.e. evapotranspiration) since the water is extracted from the groundwater by plants
            irrigation_ = irrigation_year[doy - 1, :, :]
            irrigation = irrigation_.flatten()
            irrigation = irrigation.reshape(config_modflow['ny'] * 2, config_modflow['nx'] * 2).astype(np.float64) / 1000  # mm/day to m/day
            irrigation = aggregate_to_coarser_resolution(irrigation, 25, config_modflow['dx'], method="average")
            irrigation[irrigation > 0.03] = 0.03  # constrain irrigation to 0.03 m/day
            capillary_rise_irrigation = capillary_rise.flatten() + irrigation.flatten()
        else:
            capillary_rise_irrigation = capillary_rise.flatten()

        modflow_interface.set_cpr_irr_rate(capillary_rise_irrigation)

        # update well rate and pass it to MODFLOW
        well_extraction_rate = np.zeros((n_wells,), dtype=np.float64)
        well_extraction_rate[:] = groundwater_extraction[f"{year}"].values.astype(np.float64)
        well_extraction_rate[cond_drinking_water_supply] = well_extraction_rate[cond_drinking_water_supply] * daily_weights_drinking_water_supply_year_doy
        well_extraction_rate[~cond_drinking_water_supply] = well_extraction_rate[~cond_drinking_water_supply] / 365.25
        well_extraction_rate[:] = -well_extraction_rate[:]  # extraction is negative
        modflow_interface.set_well_rate(well_extraction_rate)

        # update SFR inflow and pass it to MODFLOW
        sfr_stage = np.zeros((n_reaches,), dtype=np.float64)
        modflow_interface.set_sfr_stage(sfr_stage)

        # run MODFLOW for one timestep
        modflow_interface.step()

    modflow_interface.finalize()
    click.echo("MODFLOW (transient) finalized")

if __name__ == "__main__":
    main()