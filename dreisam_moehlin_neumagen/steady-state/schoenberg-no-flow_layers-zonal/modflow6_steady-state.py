from time import time
from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
from xmipy import XmiWrapper
import flopy
import platform
import yaml
import click
import signal

def handler(signum, frame):
    raise TimeoutError("Function execution timed out")

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

base_path = Path(__file__).parent

class ModFlowSimulation:
    def __init__(
        self,
        name,
        folder,
        nlay,
        nrow,
        ncol,
        rowsize,
        colsize,
        model_run=1,
        verbose=False
    ):
        self.name = name.upper()  # MODFLOW requires the name to be uppercase
        self.folder = folder
        self.nrow = nrow
        self.ncol = ncol
        self.rowsize = rowsize
        self.colsize = colsize
        self.working_directory = os.path.join(folder, 'output')
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.verbose = verbose

        # load MODFLOW parameters
        path = Path(__file__).parent / "parameters_modflow.nc"
        ds_params = xr.open_dataset(path, engine="h5netcdf")

        path = Path(__file__).parent / "boundary_conditions.nc"
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
            sim, pname="tdis", time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1)]
        )

        # Create the Flopy groundwater flow (gwf) model object
        model_nam_file = "{}.nam".format(name)
        gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file)

        # Create the Flopy iterative model solver (ims) Package object
        ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="COMPLEX",
                                                 outer_maximum=300, inner_maximum=750)

        # Now that the overall simulation is set up, we can focus on building the groundwater flow model.  The groundwater flow model will be built by adding packages to it that describe the model characteristics.
        #
        # Define the discretization of the model. All layers are given equal thickness. The `bot` array is build from `H` and the `Nlay` values to indicate top and bottom of each layer, and `delrow` and `delcol` are computed from model size `L` and number of cells `N`. Once these are all computed, the Discretization file is built.

        # Create the discretization package
        # load elevation data of the layers
        topography = ds_params['elevations'].isel(z=0).values
        elevation_bottom_layer1 = ds_params['elevations'].isel(z=1).values
        elevation_bottom_layer2 = ds_params['elevations'].isel(z=2).values
        elevation_bottom_layer3 = ds_params['elevations'].isel(z=3).values
        elevation_bottom_layer4 = ds_params['elevations'].isel(z=4).values
        elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]

        mask = np.isfinite(topography)
        # set Schoenberg to inactive
        mask_schoenberg = (ds_params['mask_schoenberg'].values == 1)
        mask = np.where(mask_schoenberg, False, mask)
        mask_boundary_condition_schoenberg = ds_bc['mask_schoenberg_bc'].values
        mask = np.where(mask_boundary_condition_schoenberg, True, mask)
        mask_drainage_area = (ds_params['mask_drainage'].values == 1)
        domain = np.empty_like(topography)
        domain[mask] = 1
        domain[~mask] = -1
        self.modflow_basin = mask
        self.n_active_cells = np.nansum(self.modflow_basin)
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
        initial_conditions_layer1 = (topography - elevation_bottom_layer1) * 0.5 + elevation_bottom_layer1
        initial_conditions_layer2 = (elevation_bottom_layer1 - elevation_bottom_layer2) * 0.5 + elevation_bottom_layer2
        initial_conditions_layer3 = (elevation_bottom_layer2 - elevation_bottom_layer3) * 0.5 + elevation_bottom_layer3
        initial_conditions_layer4 = (elevation_bottom_layer3 - elevation_bottom_layer4) * 0.5 + elevation_bottom_layer4
        initial_conditions_layers = [initial_conditions_layer1, initial_conditions_layer2, initial_conditions_layer3, initial_conditions_layer4]
        ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=initial_conditions_layers)

        # Create the node property flow package with hydraulic conducitivities
        hydraulic_conductivities_layer1 = ds_params['kf'].isel(layer=0).values
        hydraulic_conductivities_layer2 = ds_params['kf'].isel(layer=1).values
        hydraulic_conductivities_layer3 = ds_params['kf'].isel(layer=2).values
        hydraulic_conductivities_layer4 = ds_params['kf'].isel(layer=3).values

        hydraulic_conductivities_layer1_ = ds_params['kf'].isel(layer=0).values / 86400
        hydraulic_conductivities_layer2_ = ds_params['kf'].isel(layer=1).values / 86400
        hydraulic_conductivities_layer3_ = ds_params['kf'].isel(layer=2).values / 86400
        hydraulic_conductivities_layer4_ = ds_params['kf'].isel(layer=3).values / 86400
        
        # fudge parameters
        # zone 1
        mask811 = ((hydraulic_conductivities_layer1_ == 1.1574075e-08) | (hydraulic_conductivities_layer1_ == 2.7777778e-08)) & (topography < 300)

        mask711 = ((hydraulic_conductivities_layer1_ == 1.9722222e-07) | (hydraulic_conductivities_layer1_ == 2.3055554e-07) | (hydraulic_conductivities_layer1_ == 5.7777777e-07)) & (topography < 300)
        mask721 = (hydraulic_conductivities_layer2_ == 1.9722222e-07) & (topography < 300)
        mask731 = (hydraulic_conductivities_layer3_ == 1.9722222e-07) & (topography < 300)
        mask741 = (hydraulic_conductivities_layer4_ == 1.9722222e-07) & (topography < 300)

        mask611 = ((hydraulic_conductivities_layer1_ >= 1.1583334e-06) & (hydraulic_conductivities_layer1_ <= 8.1027783e-06)) & (topography < 300)

        mask511 = ((hydraulic_conductivities_layer1_ == 1.1575000e-05) | (hydraulic_conductivities_layer1_ == 1.8181944e-04)) & (topography < 300)
        mask521 = (hydraulic_conductivities_layer2_ == 1.8180555e-05) & (topography < 300)
        mask531 = (hydraulic_conductivities_layer3_ == 1.8180555e-05) & (topography < 300)
        mask541 = (hydraulic_conductivities_layer4_ == 1.8180555e-05) & (topography < 300)

        mask421 = (hydraulic_conductivities_layer2_ == 1.8181944e-04) & (topography < 300)
        mask431 = (hydraulic_conductivities_layer3_ == 1.8181944e-04) & (topography < 300)
        mask441 = (hydraulic_conductivities_layer4_ == 1.8181944e-04) & (topography < 300)

        mask1321 = (hydraulic_conductivities_layer2_ == 1.0000000e-03) & (topography < 300)
        mask1331 = (hydraulic_conductivities_layer3_ == 1.0000000e-03) & (topography < 300)

        mask2321 = (hydraulic_conductivities_layer2_ == 1.8181807e-03) & (topography < 300)
        mask2331 = (hydraulic_conductivities_layer3_ == 1.8181807e-03) & (topography < 300)
        mask2341 = (hydraulic_conductivities_layer4_ == 1.8181807e-03) & (topography < 300)

        mask3321 = (hydraulic_conductivities_layer2_ == 3.0000000e-03) & (topography < 300)
        mask3331 = (hydraulic_conductivities_layer3_ == 3.0000000e-03) & (topography < 300)

        mask4321 = (hydraulic_conductivities_layer2_ == 4.0000002e-03) & (topography < 300)
        mask4331 = (hydraulic_conductivities_layer3_ == 4.0000002e-03) & (topography < 300)

        # fudge parameters for hydraulic conductivities in zone 2
        mask812 = ((hydraulic_conductivities_layer1_ == 1.1574075e-08) | (hydraulic_conductivities_layer1_ == 2.7777778e-08)) & (topography >= 300)

        mask712 = ((hydraulic_conductivities_layer1_ == 1.9722222e-07) | (hydraulic_conductivities_layer1_ == 2.3055554e-07) | (hydraulic_conductivities_layer1_ == 5.7777777e-07)) & (topography >= 300)
        mask722 = (hydraulic_conductivities_layer2_ == 1.9722222e-07) & (topography >= 300)
        mask732 = (hydraulic_conductivities_layer3_ == 1.9722222e-07) & (topography >= 300)
        mask742 = (hydraulic_conductivities_layer4_ == 1.9722222e-07) & (topography >= 300)

        mask612 = ((hydraulic_conductivities_layer1_ >= 1.1583334e-06) & (hydraulic_conductivities_layer1_ <= 8.1027783e-06)) & (topography >= 300)

        mask512 = ((hydraulic_conductivities_layer1_ == 1.1575000e-05) | (hydraulic_conductivities_layer1_ == 1.8181944e-04)) & (topography >= 300)
        mask522 = (hydraulic_conductivities_layer2_ == 1.8180555e-05) & (topography >= 300)
        mask532 = (hydraulic_conductivities_layer3_ == 1.8180555e-05) & (topography >= 300)
        mask542 = (hydraulic_conductivities_layer4_ == 1.8180555e-05) & (topography >= 300)

        mask422 = (hydraulic_conductivities_layer2_ == 1.8181944e-04) & (topography >= 300)
        mask432 = (hydraulic_conductivities_layer3_ == 1.8181944e-04) & (topography >= 300)
        mask442 = (hydraulic_conductivities_layer4_ == 1.8181944e-04) & (topography >= 300)

        mask1322 = (hydraulic_conductivities_layer2_ == 1.0000000e-03) & (topography >= 300)
        mask1332 = (hydraulic_conductivities_layer3_ == 1.0000000e-03) & (topography >= 300)

        mask2322 = (hydraulic_conductivities_layer2_ == 1.8181807e-03) & (topography >= 300)
        mask2332 = (hydraulic_conductivities_layer3_ == 1.8181807e-03) & (topography >= 300)
        mask2342 = (hydraulic_conductivities_layer4_ == 1.8181807e-03) & (topography >= 300)

        mask3322 = (hydraulic_conductivities_layer2_ == 3.0000000e-03) & (topography >= 300)
        mask3332 = (hydraulic_conductivities_layer3_ == 3.0000000e-03) & (topography >= 300)

        mask4322 = (hydraulic_conductivities_layer2_ == 4.0000002e-03) & (topography >= 300)
        mask4332 = (hydraulic_conductivities_layer3_ == 4.0000002e-03) & (topography >= 300)

        # fudge parameters in zone 1
        hydraulic_conductivities_layer1[mask811] = hydraulic_conductivities_layer1[mask811] * fudge_parameters['-8_1_1'].values[model_run]

        hydraulic_conductivities_layer1[mask711] = hydraulic_conductivities_layer1[mask711] * fudge_parameters['-7_1_1'].values[model_run]
        hydraulic_conductivities_layer2[mask721] = hydraulic_conductivities_layer2[mask721] * fudge_parameters['-7_2_1'].values[model_run] * 0.75
        hydraulic_conductivities_layer3[mask731] = hydraulic_conductivities_layer3[mask731] * fudge_parameters['-7_3_1'].values[model_run] * 0.5
        hydraulic_conductivities_layer4[mask741] = hydraulic_conductivities_layer4[mask741] * fudge_parameters['-7_4_1'].values[model_run] * 0.25

        hydraulic_conductivities_layer1[mask611] = hydraulic_conductivities_layer1[mask611] * fudge_parameters['-6_1_1'].values[model_run]

        hydraulic_conductivities_layer1[mask511] = hydraulic_conductivities_layer1[mask511] * fudge_parameters['-5_1_1'].values[model_run]
        hydraulic_conductivities_layer2[mask521] = hydraulic_conductivities_layer2[mask521] * fudge_parameters['-5_2_1'].values[model_run]
        hydraulic_conductivities_layer3[mask531] = hydraulic_conductivities_layer3[mask531] * fudge_parameters['-5_3_1'].values[model_run]
        hydraulic_conductivities_layer4[mask541] = hydraulic_conductivities_layer4[mask541] * fudge_parameters['-5_4_1'].values[model_run]

        hydraulic_conductivities_layer2[mask421] = hydraulic_conductivities_layer2[mask421] * fudge_parameters['-4_2_1'].values[model_run]
        hydraulic_conductivities_layer3[mask431] = hydraulic_conductivities_layer3[mask431] * fudge_parameters['-4_3_1'].values[model_run]
        hydraulic_conductivities_layer4[mask441] = hydraulic_conductivities_layer4[mask441] * fudge_parameters['-4_4_1'].values[model_run]

        hydraulic_conductivities_layer2[mask1321] = hydraulic_conductivities_layer2[mask1321] * fudge_parameters['1-3_2_1'].values[model_run]
        hydraulic_conductivities_layer3[mask1331] = hydraulic_conductivities_layer3[mask1331] * fudge_parameters['1-3_3_1'].values[model_run]

        hydraulic_conductivities_layer2[mask2321] = hydraulic_conductivities_layer2[mask2321] * fudge_parameters['1.8-3_2_1'].values[model_run]
        hydraulic_conductivities_layer3[mask2331] = hydraulic_conductivities_layer3[mask2331] * fudge_parameters['1.8-3_3_1'].values[model_run]
        hydraulic_conductivities_layer4[mask2341] = hydraulic_conductivities_layer4[mask2341] * fudge_parameters['1.8-3_4_1'].values[model_run]

        hydraulic_conductivities_layer2[mask3321] = hydraulic_conductivities_layer2[mask3321] * fudge_parameters['3-3_2_1'].values[model_run]
        hydraulic_conductivities_layer3[mask3331] = hydraulic_conductivities_layer3[mask3331] * fudge_parameters['3-3_3_1'].values[model_run]

        hydraulic_conductivities_layer2[mask4321] = hydraulic_conductivities_layer2[mask4321] * fudge_parameters['4-3_2_1'].values[model_run]
        hydraulic_conductivities_layer3[mask4331] = hydraulic_conductivities_layer3[mask4331] * fudge_parameters['4-3_3_1'].values[model_run]

        # fudge parameters in zone 2
        hydraulic_conductivities_layer1[mask812] = hydraulic_conductivities_layer1[mask812] * fudge_parameters['-8_1_2'].values[model_run]

        hydraulic_conductivities_layer1[mask712] = hydraulic_conductivities_layer1[mask712] * fudge_parameters['-7_1_2'].values[model_run]
        hydraulic_conductivities_layer2[mask722] = hydraulic_conductivities_layer2[mask722] * fudge_parameters['-7_2_2'].values[model_run]
        hydraulic_conductivities_layer3[mask732] = hydraulic_conductivities_layer3[mask732] * fudge_parameters['-7_3_2'].values[model_run] * 0.666
        hydraulic_conductivities_layer4[mask742] = hydraulic_conductivities_layer4[mask742] * fudge_parameters['-7_4_2'].values[model_run] * 0.333

        hydraulic_conductivities_layer1[mask612] = hydraulic_conductivities_layer1[mask612] * fudge_parameters['-6_1_2'].values[model_run]

        hydraulic_conductivities_layer1[mask512] = hydraulic_conductivities_layer1[mask512] * fudge_parameters['-5_1_2'].values[model_run]
        hydraulic_conductivities_layer2[mask522] = hydraulic_conductivities_layer2[mask522] * fudge_parameters['-5_2_2'].values[model_run]
        hydraulic_conductivities_layer3[mask532] = hydraulic_conductivities_layer3[mask532] * fudge_parameters['-5_3_2'].values[model_run]
        hydraulic_conductivities_layer4[mask542] = hydraulic_conductivities_layer4[mask542] * fudge_parameters['-5_4_2'].values[model_run]

        hydraulic_conductivities_layer2[mask422] = hydraulic_conductivities_layer2[mask422] * fudge_parameters['-4_2_2'].values[model_run]
        hydraulic_conductivities_layer3[mask432] = hydraulic_conductivities_layer3[mask432] * fudge_parameters['-4_3_2'].values[model_run]
        hydraulic_conductivities_layer4[mask442] = hydraulic_conductivities_layer4[mask442] * fudge_parameters['-4_4_2'].values[model_run]

        hydraulic_conductivities_layer2[mask1322] = hydraulic_conductivities_layer2[mask1322] * fudge_parameters['1-3_2_2'].values[model_run]
        hydraulic_conductivities_layer3[mask1332] = hydraulic_conductivities_layer3[mask1332] * fudge_parameters['1-3_3_2'].values[model_run]

        hydraulic_conductivities_layer2[mask2322] = hydraulic_conductivities_layer2[mask2322] * fudge_parameters['1.8-3_2_2'].values[model_run]
        hydraulic_conductivities_layer3[mask2332] = hydraulic_conductivities_layer3[mask2332] * fudge_parameters['1.8-3_3_2'].values[model_run]
        hydraulic_conductivities_layer4[mask2342] = hydraulic_conductivities_layer4[mask2342] * fudge_parameters['1.8-3_4_2'].values[model_run]

        hydraulic_conductivities_layer2[mask3322] = hydraulic_conductivities_layer2[mask3322] * fudge_parameters['3-3_2_2'].values[model_run]
        hydraulic_conductivities_layer3[mask3332] = hydraulic_conductivities_layer3[mask3332] * fudge_parameters['3-3_3_2'].values[model_run]

        hydraulic_conductivities_layer2[mask4322] = hydraulic_conductivities_layer2[mask4322] * fudge_parameters['4-3_2_2'].values[model_run]
        hydraulic_conductivities_layer3[mask4332] = hydraulic_conductivities_layer3[mask4332] * fudge_parameters['4-3_3_2'].values[model_run]

        hydraulic_conductivities_layers = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]
        npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
            gwf, pname="npf", icelltype=0, k=hydraulic_conductivities_layers, save_flows=True, wetdry=0.5
        )

        # create the storage package
        specific_yield_layer1 = recalc_specific_yield(hydraulic_conductivities_layer1)
        specific_yield_layer2 = recalc_specific_yield(hydraulic_conductivities_layer2)
        specific_yield_layer3 = recalc_specific_yield(hydraulic_conductivities_layer3)
        specific_yield_layer4 = recalc_specific_yield(hydraulic_conductivities_layer4)
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
            iconvert=1, ss=specific_storage, sy=specific_yield,  steady_state=True)

        # Create the constant head package (Dirichlet boundary condition i.e. first type)
        mask_boundary_condition_porous_aquifer = ds_bc['mask_porous_aquifer_bc'].values
        index = np.where(mask_boundary_condition_porous_aquifer == 1)
        rows_bc = index[0]
        cols_bc = index[1]

        chd_rec = []
        for ii in range(0, len(rows_bc)):
            constant_head = ds_bc['constant_head_porous_aquifer'].values[rows_bc[ii], cols_bc[ii]] - fudge_parameters['offset'].values[model_run]
            if (constant_head <= topography[rows_bc[ii], cols_bc[ii]]) and (constant_head > elevation_bottom_layer1[rows_bc[ii], cols_bc[ii]]):
                layer = 0
            elif (constant_head <= elevation_bottom_layer1[rows_bc[ii], cols_bc[ii]]) and (constant_head > elevation_bottom_layer2[rows_bc[ii], cols_bc[ii]]):
                layer = 1
            elif (constant_head <= elevation_bottom_layer2[rows_bc[ii], cols_bc[ii]]) and (constant_head > elevation_bottom_layer3[rows_bc[ii], cols_bc[ii]]):
                layer = 2
            elif (constant_head <= elevation_bottom_layer3[rows_bc[ii], cols_bc[ii]]) and (constant_head > elevation_bottom_layer4[rows_bc[ii], cols_bc[ii]]):
                layer = 3
            chd_rec.append(((layer, rows_bc[ii], cols_bc[ii]), constant_head))
            if (constant_head > elevation_bottom_layer1[rows_bc[ii], cols_bc[ii]]):
                layer = 1
                chd_rec.append(((layer, rows_bc[ii], cols_bc[ii]), elevation_bottom_layer1[rows_bc[ii], cols_bc[ii]]))
            if (constant_head > elevation_bottom_layer2[rows_bc[ii], cols_bc[ii]]):
                layer = 2
                chd_rec.append(((layer, rows_bc[ii], cols_bc[ii]), elevation_bottom_layer2[rows_bc[ii], cols_bc[ii]]))
            if (constant_head > elevation_bottom_layer3[rows_bc[ii], cols_bc[ii]]):
                layer = 3
                chd_rec.append(((layer, rows_bc[ii], cols_bc[ii]), elevation_bottom_layer3[rows_bc[ii], cols_bc[ii]]))

        chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
            gwf,
            pname="chd",
            maxbound=len(chd_rec),
            stress_period_data=chd_rec,
            save_flows=True,
        )
            
        # Recharge package (Neumann boundary condition i.e. second type)
        recharge = ds_bc['recharge'].values / 1000  # convert mm/day to m/day
        rcha = flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge * fudge_parameters['rch'].values[model_run], fixed_cell=True)

        # Create the well package (Neumann boundary condition i.e. second type)
        # pumping rate in m3/day
        wells_q = [5727, 5822, 3494, 4315, 4525, 2899, 6401, 7024, 3160, 1117, 920, 1340, 729]
        # location of the wells
        wells_y = [266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]
        wells_x = [66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]
        wel_rec = []
        for i in range(len(wells_x)):
            if i <= 5:
                wel_rec.append((1, wells_y[i], wells_x[i], -wells_q[i]))
            else:
                wel_rec.append((2, wells_y[i], wells_x[i], -wells_q[i]))  # extraction from layer 3

        wel = flopy.mf6.ModflowGwfwel(
            gwf,
            pname="wel",
            maxbound=len(wel_rec),
            stress_period_data=wel_rec,
        )

        # Create the output control package
        headfile = "{}.hds".format(name)
        head_filerecord = [headfile]
        budgetfile = "{}.cbb".format(name)
        budget_filerecord = [budgetfile]
        saverecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
        printrecord = [("HEAD", "ALL"), ("BUDGET", "ALL")]
        oc = flopy.mf6.modflow.mfgwfoc.ModflowGwfoc(
            gwf,
            pname="oc",
            saverecord=saverecord,
            head_filerecord=head_filerecord,
            budget_filerecord=budget_filerecord,
            printrecord=printrecord,
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
        library_path = self.folder.parent.parent.parent / "bin" / libary_name
        try:
            self.mf6 = XmiWrapper(str(library_path), working_directory=self.working_directory)
        except Exception as e:
            print(f"Failed to load {library_path}")
            print("with message: " + str(e))
            return self.bmi_return(success, self.working_directory)

        # modflow requires the real path (no symlinks etc.)
        config_file = self.folder / 'output' / 'mfsim.nam'
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found on disk. Did you create the model first (load_from_disk = False)?")

        try:
            # initialize the model
            self.mf6.initialize(str(config_file))
        except:
            return self.bmi_return(success, str(self.folder / 'output'))

        if self.verbose:
            print("MODFLOW model initialized")

        
        self.end_time = self.mf6.get_end_time()

        mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
        self.max_iter = self.mf6.get_value_ptr(mxit_tag)[0]

        self.prepare_time_step()

    def prepare_time_step(self):
        dt = self.mf6.get_time_step()
        self.mf6.prepare_time_step(dt)

    def step(self):
        if self.mf6.get_current_time() > self.end_time:
            raise StopIteration("MODFLOW used all iteration steps. Consider increasing `ndays`")

        t0 = time()
        self.mf6.prepare_solve(1)

        # limit the execution time of the numerical solver
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(60)  # Set the timeout duration to 60 seconds

        complete = 0
        self.mf6.prepare_solve(1)
        try:
            # convergence loop
            for _ in range(self.max_iter):
                has_converged = self.mf6.solve(1)

                if has_converged:
                    complete = 1
                    break
        except TimeoutError:
            has_converged = False
            print("MODFLOW numerical solver timed out")
        finally:
            signal.alarm(0)  # Reset the alarm
            self.mf6.finalize_solve(1)

        self.mf6.finalize_time_step()

        if self.verbose:
            print(f'MODFLOW timestep {int(self.mf6.get_current_time())} converged in {round(time() - t0, 2)} seconds')
        
        # If next step exists, prepare timestep. Otherwise the data set through the bmi
        # will be overwritten when preparing the next timestep.
        if self.mf6.get_current_time() < self.end_time:
            self.prepare_time_step()

        return complete

    def finalize(self):
        self.mf6.finalize()

@click.option("-mr", "--model-run", type=int, default=27)
@click.command("main", short_help="Run MODFLOW in steady-state mode")
def main(model_run):
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        roger_config = yaml.safe_load(file)

    res_modflow = 50  # spatial resolution of MODFLOW in meters

    modflow_config = {
        'dx': res_modflow,
        'dy': res_modflow,
        'nx': 621,
        'ny': 777,
        'nz': 4,
    }
    
    # initialize the MODFLOW model using XMI
    modflow_interface = ModFlowSimulation(
        f"dmn_run_{model_run}",
        base_path,
        nlay=4,
        nrow=modflow_config['nx'],
        ncol=modflow_config['ny'],
        rowsize=modflow_config['dx'],
        colsize=modflow_config['dy'],
        model_run=model_run,
        verbose=True
    )
    # run MODFLOW for one timestep
    complete = modflow_interface.step()
    
    modflow_interface.finalize()
    print("MODFLOW (steady-state) finalized")

    # update the fudge parameters if the parameter set produces a useful simulation
    path = base_path / "fudge_parameters_modflow.csv"
    fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)
    fudge_parameters.loc[fudge_parameters.index[model_run], "complete"] = complete
    fudge_parameters.columns = [
        ["[-]", "[-]", "[-]", "[-]", 
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]", 
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]",
         "[-]", "[m]",  ""],
        ["-8_1_1", "-7_1_1", "-6_1_1", "-5_1_1", 
         "-7_2_1", "-5_2_1","-4_2_1", "1-3_2_1", "1.8-3_2_1", "3-3_2_1", "4-3_2_1",  
         "-7_3_1", "-5_3_1","-4_3_1", "1-3_3_1", "1.8-3_3_1", "3-3_3_1", "4-3_3_1",
         "-7_4_1", "-5_4_1","-4_4_1", "1.8-3_4_1",
         "-8_1_2", "-7_1_2", "-6_1_2", "-5_1_2", 
         "-7_2_2", "-5_2_2","-4_2_2", "1-3_2_2", "1.8-3_2_2", "3-3_2_2", "4-3_2_2",  
         "-7_3_2", "-5_3_2","-4_3_2", "1-3_3_2", "1.8-3_3_2", "3-3_3_2", "4-3_3_2",
         "-7_4_2", "-5_4_2","-4_4_2", "1.8-3_4_2",    
         "rch", "offset", "complete"],
    ]
    fudge_parameters.to_csv(path, index=False, sep=";")
    return

if __name__ == "__main__":
    main()
