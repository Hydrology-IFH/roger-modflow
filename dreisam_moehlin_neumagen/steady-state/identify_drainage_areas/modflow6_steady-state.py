from time import time
from pathlib import Path
import os
import numpy as np
import pandas as pd
import xarray as xr
from xmipy import XmiWrapper
import flopy
import scipy
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

file_config = base_path.parent / "config.yml"
with open(file_config, "r") as file:
    modflow_config = yaml.safe_load(file)

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
        model_run=0,
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
        path = Path(__file__).parent.parent / "input" / "parameters_modflow.nc"
        ds_params = xr.open_dataset(path, engine="h5netcdf")

        path = Path(__file__).parent.parent / "input" / "boundary_conditions.nc"
        ds_bc = xr.open_dataset(path, engine="h5netcdf")

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
        gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file, save_flows=True, newtonoptions="NEWTON")

        # Create the Flopy iterative model solver (ims) Package object
        ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", print_option="all",
                                                 no_ptcrecord="NO_PTC_ALL",
                                                 outer_maximum=50, inner_maximum=200,
                                                 outer_dvclose=0.1, inner_dvclose=0.1,
                                                 linear_acceleration="BICGSTAB")

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
        initial_conditions_layer1 = (topography - elevation_bottom_layer1) * 0.75 + elevation_bottom_layer1
        initial_conditions_layer2 = (elevation_bottom_layer1 - elevation_bottom_layer2) * 0.75 + elevation_bottom_layer2
        initial_conditions_layer3 = (elevation_bottom_layer2 - elevation_bottom_layer3) * 0.75 + elevation_bottom_layer3
        initial_conditions_layer4 = (elevation_bottom_layer3 - elevation_bottom_layer4) * 0.75 + elevation_bottom_layer4
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
        mask1 = (hydraulic_conductivities_layer1_ <= 10**-10)
        mask2 = (hydraulic_conductivities_layer2_ <= 10**-10)
        mask3 = (hydraulic_conductivities_layer3_ <= 10**-10)
        mask4 = (hydraulic_conductivities_layer4_ <= 10**-10)  
        hydraulic_conductivities_layer1[mask1] = hydraulic_conductivities_layer1[mask1] * 1000
        hydraulic_conductivities_layer2[mask2] = hydraulic_conductivities_layer2[mask2] * 1000
        hydraulic_conductivities_layer3[mask3] = hydraulic_conductivities_layer3[mask3] * 1000
        hydraulic_conductivities_layer4[mask4] = hydraulic_conductivities_layer4[mask4] * 1000

        mask_ = (hydraulic_conductivities_layer2_ == 1.9999999e-07)
        hydraulic_conductivities_layer2_ = np.where(mask_, 1.9722222e-07, hydraulic_conductivities_layer2_)
        mask_ = (hydraulic_conductivities_layer3_ == 1.9999999e-07)
        hydraulic_conductivities_layer3_ = np.where(mask_, 1.9722222e-07, hydraulic_conductivities_layer3_)
        mask_ = (hydraulic_conductivities_layer4_ == 1.9999999e-07)
        hydraulic_conductivities_layer4_ = np.where(mask_, 1.9722222e-07, hydraulic_conductivities_layer4_)

        mask81 = (hydraulic_conductivities_layer1_ == 1.1574075e-08) | (hydraulic_conductivities_layer1_ == 2.7777778e-08)

        mask71 = (hydraulic_conductivities_layer1_ == 1.9722222e-07) | (hydraulic_conductivities_layer1_ == 2.3055554e-07) | (hydraulic_conductivities_layer1_ == 5.7777777e-07)
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
        hydraulic_conductivities_layer1[mask81] = hydraulic_conductivities_layer1[mask81] * 500

        hydraulic_conductivities_layer1[mask71] = hydraulic_conductivities_layer1[mask71] * 75
        hydraulic_conductivities_layer2[mask72] = hydraulic_conductivities_layer2[mask72] * 60
        hydraulic_conductivities_layer3[mask73] = hydraulic_conductivities_layer3[mask73] * 50
        hydraulic_conductivities_layer4[mask74] = hydraulic_conductivities_layer4[mask74] * 10

        hydraulic_conductivities_layer1[mask61] = hydraulic_conductivities_layer1[mask61]

        hydraulic_conductivities_layer1[mask51] = hydraulic_conductivities_layer1[mask51]
        hydraulic_conductivities_layer2[mask52] = hydraulic_conductivities_layer2[mask52]
        hydraulic_conductivities_layer3[mask53] = hydraulic_conductivities_layer3[mask53]
        hydraulic_conductivities_layer4[mask54] = hydraulic_conductivities_layer4[mask54]

        hydraulic_conductivities_layer2[mask42] = hydraulic_conductivities_layer2[mask42]
        hydraulic_conductivities_layer3[mask43] = hydraulic_conductivities_layer3[mask43]
        hydraulic_conductivities_layer4[mask44] = hydraulic_conductivities_layer4[mask44]

        hydraulic_conductivities_layer2[mask132] = hydraulic_conductivities_layer2[mask132]
        hydraulic_conductivities_layer3[mask133] = hydraulic_conductivities_layer3[mask133]

        hydraulic_conductivities_layer2[mask232] = hydraulic_conductivities_layer2[mask232]
        hydraulic_conductivities_layer3[mask233] = hydraulic_conductivities_layer3[mask233]
        hydraulic_conductivities_layer4[mask234] = hydraulic_conductivities_layer4[mask234]

        hydraulic_conductivities_layer2[mask332] = hydraulic_conductivities_layer2[mask332]
        hydraulic_conductivities_layer3[mask333] = hydraulic_conductivities_layer3[mask333]

        hydraulic_conductivities_layer2[mask432] = hydraulic_conductivities_layer2[mask432]
        hydraulic_conductivities_layer3[mask433] = hydraulic_conductivities_layer3[mask433]

        # smooth transition between fissured and porous aquifers
        hydraulic_conductivities_layer1[np.isnan(hydraulic_conductivities_layer1)] = 0
        hydraulic_conductivities_layer2[np.isnan(hydraulic_conductivities_layer2)] = 0
        hydraulic_conductivities_layer3[np.isnan(hydraulic_conductivities_layer3)] = 0
        hydraulic_conductivities_layer4[np.isnan(hydraulic_conductivities_layer4)] = 0
        hydraulic_conductivities_layer1 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer1, [1.0, 1.0], mode='constant')
        hydraulic_conductivities_layer2 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer2, [1.0, 1.0], mode='constant')
        hydraulic_conductivities_layer3 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer3, [1.0, 1.0], mode='constant')
        hydraulic_conductivities_layer4 = scipy.ndimage.gaussian_filter(hydraulic_conductivities_layer4, [1.0, 1.0], mode='constant')
        hydraulic_conductivities_layer1[~mask] = np.nan
        hydraulic_conductivities_layer2[~mask] = np.nan
        hydraulic_conductivities_layer3[~mask] = np.nan
        hydraulic_conductivities_layer4[~mask] = np.nan

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
            constant_head = ds_bc['constant_head_porous_aquifer'].values[rows_bc[ii], cols_bc[ii]]
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
        recharge = ds_bc['recharge'].values / 1000  # convert mm/day to m/day
        rcha = flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge * 0.8, fixed_cell=True)

        # load the groundwater extraction data
        groundwater_extraction = pd.read_csv(base_path.parent / 'input' / 'groundwater_extraction.csv', sep=';')
        # Create the well package (Neumann boundary condition i.e. second type)
        # pumping rate in m3/day
        wells_q = groundwater_extraction["annual_average"].values.tolist()
        # location of the wells
        groundwater_extraction["cell_y"] = groundwater_extraction["cell_y"].values - 1
        groundwater_extraction["cell_x"] = groundwater_extraction["cell_x"].values - 1
        groundwater_extraction["layer"] = groundwater_extraction["layer"].values - 1

        wells_y = groundwater_extraction["cell_y"].values.tolist()
        wells_x = groundwater_extraction["cell_x"].values.tolist()
        wells_layer = groundwater_extraction["layer"].values.tolist()
        wel_rec = []
        for i in range(len(wells_x)):
            wel_rec.append((wells_layer[i], wells_y[i], wells_x[i], -wells_q[i]))

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

        # limit the execution time of the numerical solver
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(120)  # Set the timeout duration to 60 seconds

        converged = 0
        self.mf6.prepare_solve(1)
        t0 = time()
        try:
            # convergence loop
            for i in range(self.max_iter):
                has_converged = self.mf6.solve(1)
                print(f"MODFLOW iteration {i+1} of {self.max_iter}. Convergence of numerical solution: {has_converged}")

                if has_converged:
                    converged = 1
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

        return converged

    def finalize(self):
        self.mf6.finalize()

@click.option("-mr", "--model-run", type=int, default=0)
@click.command("main", short_help="Run MODFLOW in steady-state mode")
def main(model_run):    
    # initialize the MODFLOW model using XMI
    modflow_interface = ModFlowSimulation(
        f"dmn_run_{model_run}",
        base_path,
        nlay=4,
        nrow=modflow_config['nx'],
        ncol=modflow_config['ny'],
        rowsize=modflow_config['dx'],
        colsize=modflow_config['dy'],
        model_run=0,
        verbose=True
    )
    # run MODFLOW for one timestep
    converged = modflow_interface.step()
    
    modflow_interface.finalize()
    print("MODFLOW (steady-state) finalized")

    print(f"converged: {converged}")
    return

if __name__ == "__main__":
    main()
