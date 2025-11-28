from time import time
from pathlib import Path
import os
import numpy as np
from xmipy import XmiWrapper
import flopy
from flopy.utils import Raster
import scipy
import platform
import yaml
import xarray as xr
import xesmf as xe
import click
import pandas as pd
import subprocess

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
        folder,
        ndays,
        nlay,
        nrow,
        ncol,
        rowsize,
        colsize,
        model_run=5,
        verbose=False
    ):
        self.name = name.upper()  # MODFLOW requires the name to be uppercase
        self.folder = folder
        self.nrow = nrow
        self.ncol = ncol
        self.rowsize = rowsize
        self.colsize = colsize
        self.working_directory = os.path.join(folder, "output")
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.verbose = verbose

        # load MODFLOW parameters
        path = Path(__file__).parent / "input" / "parameters_modflow.nc"
        ds_params_modflow = xr.open_dataset(path, engine="h5netcdf")

        path = Path(__file__).parent / "input" / "boundary_conditions.nc"
        ds_bc = xr.open_dataset(path, engine="h5netcdf")

        path = base_path / "fudge_parameters_modflow.csv"
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
            sim, pname="tdis", time_units="DAYS", start_date_time='2019-11-01', nper=ndays, perioddata=[(1.0, 1, 1)] * ndays
        )

        # Create the Flopy groundwater flow (gwf) model object
        model_nam_file = "{}.nam".format(name)
        gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file, save_flows=True, newtonoptions="NEWTON")

        # Create the Flopy iterative model solver (ims) Package object
        ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", print_option="all", complexity="COMPLEX", no_ptcrecord="NO_PTC_ALL")

        # Now that the overall simulation is set up, we can focus on building the groundwater flow model.  The groundwater flow model will be built by adding packages to it that describe the model characteristics.
        #
        # Define the discretization of the model. All layers are given equal thickness. The `bot` array is build from `H` and the `Nlay` values to indicate top and bottom of each layer, and `delrow` and `delcol` are computed from model size `L` and number of cells `N`. Once these are all computed, the Discretization file is built.

        # Create the discretization package
        # load elevation data of the layers
        topography = ds_params_modflow["elevations"].isel(z=0).values
        elevation_bottom_layer1 = ds_params_modflow["elevations"].isel(z=1).values
        elevation_bottom_layer2 = ds_params_modflow["elevations"].isel(z=2).values
        elevation_bottom_layer3 = ds_params_modflow["elevations"].isel(z=3).values
        elevation_bottom_layer4 = ds_params_modflow["elevations"].isel(z=4).values
        elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

        mask = np.isfinite(topography)
        # set Schoenberg to inactive
        mask_schoenberg = (ds_params_modflow["mask_schoenberg"].values == 1)
        mask = np.where(mask_schoenberg, False, mask)
        mask_boundary_condition_schoenberg = ds_bc["mask_schoenberg_bc"].values
        mask = np.where(mask_boundary_condition_schoenberg, True, mask)
        mask_drainage_area = (ds_params_modflow["mask_drainage"].values == 1)
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
        gw_heads_interpolated = ds_params_modflow["gw_heads_interpolated"].values - 1
        gw_heads_interpolated[~mask] = np.nan
        initial_conditions_layers = [gw_heads_interpolated, gw_heads_interpolated, gw_heads_interpolated, gw_heads_interpolated]
        ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=initial_conditions_layers)

        # Create the node property flow package with hydraulic conducitivities
        hydraulic_conductivities_layer1 = ds_params_modflow["kf"].isel(layer=0).values
        hydraulic_conductivities_layer2 = ds_params_modflow["kf"].isel(layer=1).values
        hydraulic_conductivities_layer3 = ds_params_modflow["kf"].isel(layer=2).values
        hydraulic_conductivities_layer4 = ds_params_modflow["kf"].isel(layer=3).values

        hydraulic_conductivities_layer1_ = ds_params_modflow["kf"].isel(layer=0).values / 86400
        hydraulic_conductivities_layer2_ = ds_params_modflow["kf"].isel(layer=1).values / 86400
        hydraulic_conductivities_layer3_ = ds_params_modflow["kf"].isel(layer=2).values / 86400
        hydraulic_conductivities_layer4_ = ds_params_modflow["kf"].isel(layer=3).values / 86400
        
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

        # smooth transition between fissured and porous aquifers
        hydraulic_conductivities_layer1[np.isnan(hydraulic_conductivities_layer1)] = 0
        hydraulic_conductivities_layer2[np.isnan(hydraulic_conductivities_layer2)] = 0
        hydraulic_conductivities_layer3[np.isnan(hydraulic_conductivities_layer3)] = 0
        hydraulic_conductivities_layer4[np.isnan(hydraulic_conductivities_layer4)] = 0
        hydraulic_conductivities_layer1 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer1, [1.5, 1.5], mode="constant")
        hydraulic_conductivities_layer2 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer2, [1.5, 1.5], mode="constant")
        hydraulic_conductivities_layer3 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer3, [1.5, 1.5], mode="constant")
        hydraulic_conductivities_layer4 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer4, [1.5, 1.5], mode="constant")

        hydraulic_conductivities_layer1[~mask] = np.nan
        hydraulic_conductivities_layer2[~mask] = np.nan
        hydraulic_conductivities_layer3[~mask] = np.nan
        hydraulic_conductivities_layer4[~mask] = np.nan

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
        _specific_yield_layer1 = scipy.ndimage.gaussian_filter(specific_yield_layer1, [1.0, 1.0], mode="constant")
        _specific_yield_layer2 = scipy.ndimage.gaussian_filter(specific_yield_layer2, [1.0, 1.0], mode="constant")
        _specific_yield_layer3 = scipy.ndimage.gaussian_filter(specific_yield_layer3, [1.0, 1.0], mode="constant")
        _specific_yield_layer4 = scipy.ndimage.gaussian_filter(specific_yield_layer4, [1.0, 1.0], mode="constant")
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
        specific_storage[0]["data"] = specific_yield[0]["data"] * thickness_layer1
        specific_storage[1]["data"] = specific_yield[1]["data"] * thickness_layer2
        specific_storage[2]["data"] = specific_yield[2]["data"] * thickness_layer3
        specific_storage[3]["data"] = specific_yield[3]["data"] * thickness_layer4

        sto = flopy.mf6.ModflowGwfsto(gwf, pname="sto",
            iconvert=1, ss=specific_storage, sy=specific_yield, steady_state=False)

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
            
        # Recharge package
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

        # Create the output control package
        headfile = "{}.hds".format(name)
        head_filerecord = [headfile]
        budgetfile = "{}.cbc".format(name)
        budget_filerecord = [budgetfile]
        saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
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
        fpth = os.path.join(model_ws, 'mfsim.stdout')
        if os.path.exists(fpth):
            lines = open(fpth).readlines()
        else:
            lines = None
        return success, lines

    def load_bmi(self):
        """Load the Basic Model Interface"""
        success = False
        
        
        if platform.system() == 'Windows':
            libary_name = 'libmf6.dll'
        elif platform.system() == 'Linux':
            libary_name = 'libmf6.so'
        elif platform.system() == 'Darwin':
            libary_name = 'libmf6.dylib'
        else:
            raise ValueError(f'Platform {platform.system()} not recognized.')

        # modflow requires the real path (no symlinks etc.)
        library_path = self.folder.parent.parent / "bin" / libary_name
        try:
            self.mf6 = XmiWrapper(str(library_path), working_directory=self.working_directory)
        except Exception as e:
            print(f"Failed to load {library_path}")
            print("with message: " + str(e))
            return self.bmi_return(success, self.working_directory)

        # modflow requires the real path (no symlinks etc.)
        config_file = self.folder / 'output' / 'mfsim.nam'
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"config_roger file {config_file} not found on disk. Did you create the model first (load_from_disk = False)?")

        # initialize the model
        try:
            self.mf6.initialize(str(config_file))
        except:
            return self.bmi_return(success, str(self.folder / 'output'))

        if self.verbose:
            print("MODFLOW model initialized")

        
        self.end_time = self.mf6.get_end_time()

        recharge_tag = self.mf6.get_var_address("RECHARGE", self.name, "RCH_0")
        # there seems to be a bug in xmipy where the size of the pointer to RCHA is
        # is the size of the entire modflow area, including modflow_basined cells. Only the first
        # part of the array is actually used, when a part of the area is modflow_basined. Since
        # numpy returns a view of the array when the array[]-syntax is used, we can simply
        # use the view of the first part of the array up to the number of active
        # (non-modflow_basined) cells
        self.recharge = self.mf6.get_value_ptr(recharge_tag)
        
        head_tag = self.mf6.get_var_address("X", self.name)
        self.head = self.mf6.get_value_ptr(head_tag)

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

    def get_groundwater_head(self, groundwater_head):
        """Get groundwater head from upper layer, value in m"""
        head_tag = self.mf6.get_var_address("X", self.name)
        mask = self.modflow_basin.flatten()  # mask of active cells
        groundwater_head[mask] = self.mf6.get_value_ptr(head_tag)[:self.n_active_cells_per_layer]

    def get_recharge(self, recharge):
        """Get recharge, value in m/day"""
        recharge_tag = self.mf6.get_var_address("RECHARGE", self.name, "RCH_0")
        mask = self.modflow_basin.flatten() 
        recharge[mask] = self.mf6.get_value_ptr(recharge_tag)

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
            print(f'MODFLOW timestep {int(self.mf6.get_current_time())} converged in {round(time() - t0, 2)} seconds')
        
        # If next step exists, prepare timestep. Otherwise the data set through the bmi
        # will be overwritten when preparing the next timestep.
        if self.mf6.get_current_time() < self.end_time:
            self.prepare_time_step()

    def finalize(self):
        self.mf6.finalize()

@click.option("-b", "--backend", type=click.Choice(["numpy", "jax"]), default="numpy", help="Computational backend of RoGeR")
@click.option("-ft", "--float_type", type=click.Choice(["float32", "float64"]), default="float32", help="Float type of RoGeR")
@click.command("main", short_help="Run MODFLOW in transient mode coupled with RoGeR.")
def main(backend, float_type):
    from roger import runtime_settings
    runtime_settings.update(
    backend=backend,
    float_type=float_type,
    )
    from bmiroger import BmiRoger
    from roger.bmimodels.svat_dist import SVATDISTSetup
    from roger.tools.setup import write_forcing_distributed
    file_config = base_path / "config_roger.yml"
    with open(file_config, "r") as file:
        config_roger = yaml.safe_load(file)

    file_config = base_path / "config_modflow.yml"
    with open(file_config, "r") as file:
        config_modflow = yaml.safe_load(file)

    # define the output variables of RoGeR
    config_roger['OUTPUT_COLLECT'] = ["theta", "z_gw"]
    config_roger['OUTPUT_RATE'] = ["q_hof", "q_ss"]

    # file = base_path / "write_parameters_to_csv_for_SVAT.py"
    # subprocess.run(["python", str(file)], check=True, timeout=20)

    # set the number of grid cells in x and y direction from the parameters file
    file = base_path / "input" / "parameters_roger_25m.nc"
    with xr.open_dataset(file) as ds:
        config_roger['nx'] = ds.sizes['y']
        config_roger['ny'] = ds.sizes['x']

    # save the updated config file
    file_config = base_path / "config_roger.yml"
    with open(file_config, "w") as file:
        yaml.dump(config_roger, file)

    # load the parameters of MODFLOW
    path = Path(__file__).parent / "input" / "parameters_modflow.nc"
    ds_params_modflow = xr.open_dataset(path, engine="h5netcdf")

    path = Path(__file__).parent / "input" / "boundary_conditions.nc"
    ds_bc = xr.open_dataset(path, engine="h5netcdf")

    # load the spatial domain and elevation data
    topography = ds_params_modflow["elevations"].isel(z=0).values
    mask = np.isfinite(topography)
    # set Schoenberg to inactive
    mask_schoenberg = (ds_params_modflow["mask_schoenberg"].values == 1)
    mask = np.where(mask_schoenberg, False, mask)
    mask_boundary_condition_schoenberg = ds_bc["mask_schoenberg_bc"].values
    mask = np.where(mask_boundary_condition_schoenberg, True, mask)
    domain = np.empty_like(topography)
    domain[mask] = 1
    domain[~mask] = -1
    topography = aggregate_to_finer_resolution(topography, config_modflow['dx'], config_roger['dx'], method="keep")

    # initialize the SVAT model of RoGeR using BMI
    model = SVATDISTSetup(base_path)
    write_forcing_distributed(base_path / "input")
    roger_interface = BmiRoger(model=model)
    roger_interface._model._output_dir = base_path / "output"
    roger_interface.initialize(base_path)
    print("RoGeR model initialized")
    NDAYS = int(roger_interface.get_end_time() / (60 * 60 * 24))  # seconds to days

    soildepth = np.zeros(roger_interface.get_grid_node_count())
    roger_interface.get_value("z_soil", soildepth)
    soildepth = soildepth / 1000  # mm to m
    roger_mask = np.empty(roger_interface.get_grid_node_count(), dtype=bool)
    roger_interface.get_value("maskCatch", roger_mask)
    roger_mask = roger_mask.reshape(config_roger['nx'], config_roger['ny'])
    
    # initialize the MODFLOW model using XMI
    modflow_interface = ModFlowSimulation(
        config_roger['identifier'],
        base_path,
        ndays=NDAYS,
        nlay=config_modflow['nz'],
        nrow=config_modflow['nx'],
        ncol=config_modflow['ny'],
        rowsize=config_modflow['dx'],
        colsize=config_modflow['dy'],
        verbose=True
    )
    for _ in range(NDAYS):
        # update groundwater head
        groundwater_head = np.zeros(config_modflow['nx'] * config_modflow['ny'])
        modflow_interface.get_groundwater_head(groundwater_head)
        groundwater_head = groundwater_head.reshape(config_modflow['nx'], config_modflow['ny'])
        # aggregate groundwater head to the resolution of RoGeR
        groundwater_head = aggregate_to_finer_resolution(groundwater_head, config_modflow['dx'], config_roger['dx'], method="keep")
        # RoGeR requires depth of groundwater head (in meters)
        groundwater_depth = topography.flatten() - groundwater_head.flatten()
        groundwater_depth[(groundwater_depth <= soildepth)] = soildepth[(groundwater_depth <= soildepth)] + 0.05  # constrain groundwater depth to soil depth
        with roger_interface._model.state.variables.unlock():
            roger_interface.set_value("z_gw", groundwater_depth)

        # run RoGeR for one timestep
        roger_interface.update_until(roger_interface._model._config["OUTPUT_FREQUENCY"])

        # update recharge and pass it to MODFLOW
        recharge = np.zeros(roger_interface.get_grid_node_count())
        roger_interface.get_value("q_ss", recharge)
        recharge[(groundwater_depth <= soildepth)] = 0 # constrain recharge to zero where groundwater depth is equal to soil depth
        recharge = recharge.reshape(config_roger['nx'], config_roger['ny']).astype(np.float64) / 1000  # mm/day to m/day
        recharge = aggregate_to_coarser_resolution(recharge, config_roger['dx'], config_modflow['dx'], method="average")
        recharge[~mask] = np.nan
        recharge = recharge.flatten()
        modflow_interface.set_recharge(recharge)
    
        # run MODFLOW for one timestep
        modflow_interface.step()

    roger_interface.finalize()
    modflow_interface.finalize()
    print("RoGeR and MODFLOW (transient) finalized")

if __name__ == "__main__":
    main()

# import matplotlib.pyplot as plt
# grid_extent = (0, 404 * 25, 0, 356 * 25)

# fig, axes = plt.subplots(figsize=(4, 4))
# topography[~roger_mask] = np.nan
# plt.imshow(topography, extent=grid_extent, cmap='viridis', aspect='equal', vmin=200, vmax=1200)
# plt.colorbar(label='[m a.s.l.]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path / "figures" / "debug_topo.png"
# fig.savefig(file, dpi=300)

# fig, axes = plt.subplots(figsize=(4, 4))
# groundwater_head[~roger_mask] = np.nan
# plt.imshow(groundwater_head, extent=grid_extent, cmap='viridis', aspect='equal', vmin=200, vmax=1200)
# plt.colorbar(label='[m a.s.l.]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path / "figures" / "debug_gw_head.png"
# fig.savefig(file, dpi=300)

# fig, axes = plt.subplots(figsize=(4, 4))
# plt.imshow(topography - groundwater_head, extent=grid_extent, cmap='viridis', aspect='equal')
# plt.colorbar(label='[m a.s.l.]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path / "figures" / "debug_gw_depth.png"
# fig.savefig(file, dpi=300)

# fig, axes = plt.subplots(figsize=(4, 4))
# recharge = recharge.reshape(config_modflow['nx'], config_modflow['ny']) * 1000
# recharge[~mask] = np.nan
# plt.imshow(recharge, extent=grid_extent, cmap='viridis', aspect='equal')
# plt.colorbar(label='[mm/day]', shrink=0.5)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
# plt.tight_layout()
# file = base_path / "figures" / "debug_recharge.png"
# fig.savefig(file, dpi=300)

# plt.close("all")
