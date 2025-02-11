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

def recalc_specific_yield(hydraulic_conductivity, specific_yield_min=0.05):
    """Recalculate specific yield based on hydraulic conductivity using the formula of Marotz (1968)

    Args:
        hydraulic_conductivity (numpy.ndarray): hydraulic conductivity in m/day
        specific_yield_min (float, optional): Constraint of specific yield. Default is 0.05.

    Returns:
        numpy.ndarray: specific yield
    """
    specific_yield = 0.462 + 0.045 * np.log(hydraulic_conductivity/86400)
    specific_yield[specific_yield < specific_yield_min] = specific_yield_min
    return specific_yield

base_path = Path(__file__).parent

xlim1 = 480
xlim2 = 520
ylim1 = 190
ylim2 = 228

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
        
        # Create the discretization package
        # load elevation data of the layers
        topography = ds_params['elevations'].isel(z=0).values
        # elevation_bottom_layer = topography - 50
        elevation_bottom_layer = ds_params['elevations'].isel(z=3).values

        mask = np.isfinite(topography)
        mask[:, :] = False
        mask[ylim1:ylim2, xlim1:xlim2] = True
        mask_upper_dreisam = (ds_params['mask_upper_dreisam'].values == 1)
        mask_upper_moehlin = (ds_params['mask_upper_moehlin'].values == 1)
        mask_neumagen = (ds_params['mask_neumagen'].values == 1)
        mask_valleys_black_forest = mask_upper_dreisam | mask_upper_moehlin | mask_neumagen
        domain = np.empty_like(topography)
        domain[mask] = 1
        domain[~mask] = -1
        self.modflow_basin = mask
        self.n_active_cells = np.nansum(self.modflow_basin)
        domain_layers = [domain]
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
            botm=elevation_bottom_layer,
            idomain=domain_layers,
        )

        # mf.dis.sr = SpatialReference(delr=delRArray,delc=delCArray, xul=GloRefBox[0], yul= GloRefBox[3],epsg=32718)

        # Create the initial conditions package
        initial_conditions_layer = (topography - elevation_bottom_layer) * 0.5 + elevation_bottom_layer
        initial_conditions_layers = [initial_conditions_layer]
        ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=initial_conditions_layers)

        # Create the node property flow package with hydraulic conducitivities
        hydraulic_conductivities_layer = ds_params['kf'].isel(layer=3).values * 1
        
        hydraulic_conductivities_layers = [hydraulic_conductivities_layer]
        npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
            gwf, pname="npf", icelltype=0, k=hydraulic_conductivities_layers, save_flows=True
        )

        # create the storage package
        specific_yield_layer = recalc_specific_yield(hydraulic_conductivities_layer)

        thickness_layer = topography - elevation_bottom_layer
        specific_storage_layer = specific_yield_layer * thickness_layer

        sto = flopy.mf6.ModflowGwfsto(gwf, pname="sto",
            iconvert=1, ss=specific_storage_layer, sy=specific_yield_layer, steady_state=True)

        # # Create the constant head package (Dirichlet boundary condition i.e. first type)
        # mask_boundary_condition = ds_bc['mask_rivers'].values
        # index = np.where(mask_boundary_condition == 1)
        # rows_bc = index[0]
        # cols_bc = index[1]

        # chd_rec = []
        # for ii in range(0, len(rows_bc)):
        #     constant_head = topography[rows_bc[ii], cols_bc[ii]] - 0.2
        #     chd_rec.append(((0, rows_bc[ii], cols_bc[ii]), constant_head))

        # chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
        #     gwf,
        #     pname="chd",
        #     maxbound=len(chd_rec),
        #     stress_period_data=chd_rec,
        #     save_flows=True,
        # )

        # # Create the general head package (Cauchy boundary condition i.e. third type)
        # ghb_rec = []
        # mask_boundary_condition = ds_bc['mask_porous_aquifer_bc'].values
        # index = np.where(mask_boundary_condition == 1)
        # rows_bc = index[0]
        # cols_bc = index[1]
        # for ii in range(0, len(rows_bc)):
        #     if mask[rows_bc[ii], cols_bc[ii]]:
        #         if rows_bc[ii] == ylim2-1:
        #             constant_head = ds_bc['constant_head_porous_aquifer'].values[rows_bc[ii], cols_bc[ii]] - 1
        #             width = 50
        #             layer = 0
        #             thickness = constant_head - elevation_bottom_layer[rows_bc[ii], cols_bc[ii]]
        #             conductance = hydraulic_conductivities_layers[layer][rows_bc[ii], cols_bc[ii]] * thickness * width * 0.0005
        #             ghb_rec.append([layer, rows_bc[ii], cols_bc[ii], constant_head, conductance])
                # elif topography[rows_bc[ii], cols_bc[ii]] >= 355:
                #     constant_head = ds_bc['constant_head_porous_aquifer'].values[rows_bc[ii], cols_bc[ii]]
                #     width = 50
                #     layer = 0
                #     thickness = constant_head - elevation_bottom_layer[rows_bc[ii], cols_bc[ii]]
                #     conductance = hydraulic_conductivities_layers[layer][rows_bc[ii], cols_bc[ii]] * thickness * width * 0.002
                #     ghb_rec.append([layer, rows_bc[ii], cols_bc[ii], constant_head, conductance])
                # # elif cols_bc[ii] > xlim1 + 100 and cols_bc[ii] < xlim2:
                # else:
                #     constant_head = ds_bc['constant_head_porous_aquifer'].values[rows_bc[ii], cols_bc[ii]]
                #     width = 50
                #     layer = 0
                #     thickness = constant_head - elevation_bottom_layer[rows_bc[ii], cols_bc[ii]]
                #     conductance = hydraulic_conductivities_layers[layer][rows_bc[ii], cols_bc[ii]] * thickness * width * 0.00005
                #     ghb_rec.append([layer, rows_bc[ii], cols_bc[ii], constant_head, conductance])
                # else:
                #     constant_head = ds_bc['constant_head_porous_aquifer'].values[rows_bc[ii], cols_bc[ii]] - 30
                #     width = 50
                #     layer = 0
                #     thickness = constant_head - elevation_bottom_layer[rows_bc[ii], cols_bc[ii]]
                #     conductance = hydraulic_conductivities_layers[layer][rows_bc[ii], cols_bc[ii]] * thickness * width * 0.001
                #     ghb_rec.append([layer, rows_bc[ii], cols_bc[ii], constant_head, conductance])

        # ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
        #     gwf,
        #     pname="ghb",
        #     maxbound=len(ghb_rec),
        #     stress_period_data=ghb_rec,
        #     save_flows=True,
        # )

        # Create the general head package (Cauchy boundary condition i.e. third type)
        ghb_rec = []
        mask_boundary_condition = ds_bc['mask_rivers'].values
        index = np.where(mask_boundary_condition == 1)
        rows_bc = index[0]
        cols_bc = index[1]
        for ii in range(0, len(rows_bc)):
            constant_head = topography[rows_bc[ii], cols_bc[ii]] - 5
            width = 10
            layer = 0
            thickness = 0.3
            conductance = 100 * thickness * width * 0.01
            ghb_rec.append([layer, rows_bc[ii], cols_bc[ii], constant_head, conductance])

        ghb = flopy.mf6.modflow.mfgwfghb.ModflowGwfghb(
            gwf,
            pname="ghb",
            maxbound=len(ghb_rec),
            stress_period_data=ghb_rec,
            save_flows=True,
        )

        # Recharge package (Neumann boundary condition i.e. second type)
        recharge = ds_bc['recharge'].values / 1000  # convert mm/day to m/day
        rcha = flopy.mf6.ModflowGwfrcha(gwf, recharge=recharge * 1, fixed_cell=True)

        # # Create the well package (Neumann boundary condition i.e. second type)
        # # pumping rate in m3/day
        # wells_q = [5727, 5822, 3494, 4315, 4525, 2899, 6401, 7024, 3160, 1117, 920, 1340, 729]
        # # location of the wells
        # wells_y = [266, 268, 271, 272, 280, 259, 210, 212, 217, 225, 232, 228, 264]
        # wells_x = [66, 64, 63, 59, 56, 88, 464, 464, 465, 465, 477, 459, 496]
        # wel_rec = []
        # for i in range(len(wells_x)):
        #     if i <= 5:
        #         if wells_y[i] >= ylim1 and wells_y[i] <= ylim2 and wells_x[i] >= xlim1 and wells_x[i] <= xlim2:
        #             wel_rec.append((1, wells_y[i], wells_x[i], -wells_q[i]))
        #     else:
        #         if wells_y[i] >= ylim1 and wells_y[i] <= ylim2 and wells_x[i] >= xlim1 and wells_x[i] <= xlim2:
        #             wel_rec.append((2, wells_y[i], wells_x[i], -wells_q[i]))  # extraction from layer 3

        # if len(wel_rec) > 0:
        #     wel = flopy.mf6.ModflowGwfwel(
        #         gwf,
        #         pname="wel",
        #         maxbound=len(wel_rec),
        #         stress_period_data=wel_rec,
        #     )

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
        config_file = self.folder / 'output'/ 'mfsim.nam'
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

@click.option("-mr", "--model-run", type=int, default=5)
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
        nlay=1,
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
        ["[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[m]",  ""],
        ["l_1", "rz10_23", "r110_23", "r0011_23", "r10_4", "r110_4", "r0011_4", "z0011", "m0011", "m110", "rch", "offset", "complete"],
    ]
    fudge_parameters.to_csv(path, index=False, sep=";")
    return

if __name__ == "__main__":
    main()
