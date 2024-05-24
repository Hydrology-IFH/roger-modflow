from time import time
from pathlib import Path
import os
import pandas as pd
import numpy as np
from xmipy import XmiWrapper
import flopy
import platform
import yaml
from timeit import default_timer as timer
from datetime import timedelta
import subprocess
import click

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
        top,
        bottom,
        modflow_basin,
        roger_config,
        verbose=False
    ):
        self.name = name.upper()  # MODFLOW requires the name to be uppercase
        self.folder = folder
        self.nrow = nrow
        self.ncol = ncol
        self.rowsize = rowsize
        self.colsize = colsize
        self.modflow_basin = modflow_basin
        self.n_active_cells = self.modflow_basin.sum()
        self.roger_config = roger_config
        self.working_directory = os.path.join(folder, 'output')
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.verbose = verbose

        # values to generate constant head boundary conditions
        h1 = np.max(top) - 8
        h2 = np.min(top) - 8
        h1_h2 = np.linspace(h1, h2, self.ncol)
        k = 0.1  # hydraulic conductivity

        # Temporal discretization (TDIS)
        # One or more models (GWF is the only model supported at present)
        # Zero or more exchanges (instructions for how models are coupled)
        # Solutions
        #
        # For this simple hillslope example, the simulation consists of the temporal discretization (TDIS) package (TDIS), a groundwater flow (GWF) model, and an iterative model solution (IMS), which controls how the GWF model is solved.

        # Create the Flopy simulation object
        sim = flopy.mf6.MFSimulation(
            sim_name=name, exe_name="mf6", version="mf6", sim_ws=self.working_directory
        )

        # Create the Flopy temporal discretization object
        tdis = flopy.mf6.modflow.mftdis.ModflowTdis(
            sim, pname="tdis", time_units="DAYS", start_date_time=str(self.roger_config['t_origin']), nper=ndays, perioddata=[(1.0, 1, 1)] * ndays
        )

        # Create the Flopy groundwater flow (gwf) model object
        model_nam_file = "{}.nam".format(name)
        gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file)

        # Create the Flopy iterative model solver (ims) Package object
        ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="COMPLEX",
                                                 outer_maximum=300,inner_maximum=750)
        # Now that the overall simulation is set up, we can focus on building the groundwater flow model.  The groundwater flow model will be built by adding packages to it that describe the model characteristics.
        #
        # Define the discretization of the model. All layers are given equal thickness. The `bot` array is build from `H` and the `Nlay` values to indicate top and bottom of each layer, and `delrow` and `delcol` are computed from model size `L` and number of cells `N`. Once these are all computed, the Discretization file is built.

        # Create the discretization package
        top = np.max(top)
        bot = np.min(bottom)
        dis = flopy.mf6.modflow.mfgwfdis.ModflowGwfdis(
            gwf,
            pname="dis",
            nlay=nlay,
            nrow=self.nrow,
            ncol=self.ncol,
            delr=self.rowsize, 
            delc=self.colsize,
            top=top,
            botm=bot,
        )
        # Create the initial conditions package
        start = np.mean(h1_h2) * np.ones((nlay, self.nrow, self.ncol))
        ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=start)

        # Create the node property flow package with random hydraulic conducitivities
        npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
            gwf, pname="npf", icelltype=1, k=k, save_flows=True
        )

        # create the storage package
        sto = flopy.mf6.ModflowGwfsto(gwf, pname="sto",
            iconvert=1, ss=0.00001, sy=0.15, transient={0: True})

        # Create the constant head package.
        # List information is created a bit differently for
        # MODFLOW 6 than for other MODFLOW versions.  The
        # cellid (layer, row, column, for a regular grid)
        # must be entered as a tuple as the first entry.
        # Remember that these must be zero-based indices!
        chd_rec = []
        for layer in range(0, nlay):
            for col in range(0, self.ncol):
                if col != 0 and col != self.ncol - 1:
                    # boundary to the west
                    chd_rec.append(((layer, 0, col), h1_h2[col]))
                    # boundary to the east
                    chd_rec.append(((layer, self.nrow - 1, col), h1_h2[col]))
            for row in range(0, self.nrow):
                # boundary to the north
                chd_rec.append(((layer, row, 0), h1))
                # boundary to the south
                chd_rec.append(((layer, row, self.ncol - 1), h2))
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
        recharge[:, 1] = recharge_locations[1]
        recharge[:, 2] = recharge_locations[2]
        recharge = recharge.tolist()

        recharge = flopy.mf6.ModflowGwfrch(gwf, fixed_cell=False,
                            print_input=False, print_flows=False,
                            save_flows=False, boundnames=None,
                            maxbound=self.modflow_basin.sum(), stress_period_data=recharge)

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
        library_path = self.folder.parent / "bin" / libary_name
        try:
            self.mf6 = XmiWrapper(str(library_path), working_directory=self.working_directory)
        except Exception as e:
            print(f"Failed to load {library_path}")
            print("with message: " + str(e))
            return self.bmi_return(success, self.working_directory)

        # modflow requires the real path (no symlinks etc.)
        config_file = self.folder / 'output' / 'mfsim.nam'
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"roger_config file {config_file} not found on disk. Did you create the model first (load_from_disk = False)?")

        # initialize the model
        try:
            self.mf6.initialize(str(config_file))
        except:
            return self.bmi_return(success, str(self.folder / 'output'))

        if self.verbose:
            print("MODFLOW model initialized")
        
        self.end_time = self.mf6.get_end_time()
        
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
        self.mf6.set_value(recharge_tag, recharge)

    def get_groundwater_head(self, groundwater_head):
        """Set recharge, value in m/day"""
        head_tag = self.mf6.get_var_address("X", self.name)
        groundwater_head[:] = self.mf6.get_value_ptr(head_tag)

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

@click.option("-b", "--backend", type=click.Choice(["numpy", "jax"]), default="numpy")
@click.command("main", short_help="Run RoGeR and MODFLOW transient benchmark.")
def main(backend):
    from roger import runtime_settings
    runtime_settings.update(
    backend=backend,
    )
    from bmiroger import BmiRoger
    from roger.bmimodels.svat import SVATSetup
    base_path = Path(__file__).parent
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        roger_config = yaml.safe_load(file)

    # number of rows and columns to run benchmark
    rowscols = [(10, 10),
                (50, 50),
                (100, 100),
                (200, 200),
                (400, 400),
                (800, 800),
                (1000, 1000)]
    
    # dataframe to store the number of cells and the time it took to run the model
    df_time = pd.DataFrame(index=range(len(rowscols)), columns=['ncells', 'time', 'time_per_step', 'backend'])

    NDAYS = 30
    NAME = roger_config['identifier']
    NLAY = 1
    ACQUIFER_THICKNESS = 30

    for i, nxny in enumerate(rowscols):
        nx = nxny[0]
        ny = nxny[1]
        modflow_config = {
            'dx': roger_config['dx'],
            'dy': roger_config['dy'],
            'nx': nx,
            'ny': ny,
        }
        modflow_basin = np.empty((NLAY, modflow_config['nx'], modflow_config['ny']))
        modflow_basin[:, :, :] = True
        modflow_basin = modflow_basin.astype(bool)


        # initialize the SVAT model of RoGeR using BMI
        # modify the roger_config file to set the number of rows and columns
        roger_config['nx'] = nx
        roger_config['ny'] = ny
        with open(file_config, "w") as file:
            yaml.dump(roger_config, file)
        # write the parameters for the SVAT model
        file = base_path / "write_parameters.py"
        subprocess.run(["python", str(file), f"--nrows={nx}", f"--ncols={ny}"], check=True, timeout=10)
        model = SVATSetup(base_path)
        roger_interface = BmiRoger(model=model)
        roger_interface._model._output_dir = base_path / "output"
        roger_interface.initialize(base_path)
        print("RoGeR model initialized")

        # set soil depth to generate the layer boundaries for MODFLOW
        soildepth = np.zeros(roger_interface.get_grid_node_count())
        roger_interface.get_value("z_soil", soildepth)
        soildepth = soildepth / 1000  # mm to m

        # generate hillslope topography with constant slope
        arrays = [np.linspace(210, 200, modflow_config['ny']) for _ in range(modflow_config['nx'])]
        topography = np.concatenate(arrays).reshape(modflow_config['nx'], modflow_config['ny'])

        layer_boundaries = np.empty((NLAY + 1, modflow_config['nx'], modflow_config['ny']))
        layer_boundaries[0] = topography - soildepth.reshape(modflow_config['nx'], modflow_config['ny']) - 0.05
        layer_boundaries[1] = layer_boundaries[0] - ACQUIFER_THICKNESS

        modflow_interface = ModFlowSimulation(
            NAME,
            base_path,
            ndays=NDAYS,
            nlay=NLAY,
            nrow=modflow_config['nx'],
            ncol=modflow_config['ny'],
            rowsize=modflow_config['dx'],
            colsize=modflow_config['dy'],
            top=layer_boundaries[0],
            bottom=layer_boundaries[1],
            modflow_basin=modflow_basin,
            roger_config=roger_config,
            verbose=True
        )
        for t in range(NDAYS):
            if t == 3:
                start = timer()
            # update groundwater head
            groundwater_head = np.zeros(modflow_config['nx'] * modflow_config['ny'])
            modflow_interface.get_groundwater_head(groundwater_head)
            # RoGeR requires depth of groundwater head (in meters)
            groundwater_depth = topography.flatten() - groundwater_head
            roger_interface.set_value("z_gw", groundwater_depth)

            # run RoGeR for one timestep
            roger_interface.update_until(roger_interface._model._config["OUTPUT_FREQUENCY"])

            # update recharge and pass it to MODFLOW
            recharge = np.zeros(roger_interface.get_grid_node_count())
            roger_interface.get_value("q_ss", recharge)
            recharge = recharge.reshape(roger_config['nx'], roger_config['ny']).astype(np.float64) / 1000  # mm/day to m/day
            modflow_interface.set_recharge(recharge)

            # run MODFLOW for one timestep
            modflow_interface.step()

        end = timer()
        roger_interface.finalize()
        modflow_interface.finalize()
        elapsed_time = (end-start)
        elapsed_time_seconds = timedelta(seconds=elapsed_time)
        df_time.iloc[i, 0] = nx * ny
        df_time.iloc[i, 1] = elapsed_time_seconds
        df_time.iloc[i, 2] = elapsed_time_seconds/(NDAYS-3)
        df_time.iloc[i, 3] = backend
        print("RoGeR and MODFLOW (transient) finalized simulation for {} cells in {} seconds".format(nx * ny, elapsed_time_seconds))
        df_time.to_csv(base_path / f"benchmark_times_roger_{backend}_modflow_transient.csv", index=False, sep=";")

        # remove the output files
        os.remove(base_path / "output" / f"{NAME}.collect.nc")
        os.remove(base_path / "output" / f"{NAME}.rate.nc")
        os.remove(base_path / "output" / f"{NAME}.maximum.nc")
    return

if __name__ == "__main__":
    main()
