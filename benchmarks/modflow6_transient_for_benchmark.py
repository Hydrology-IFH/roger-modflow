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
        config,
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
        self.config = config
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
            sim, pname="tdis", time_units="DAYS", start_date_time=str(self.config['t_origin']), nper=ndays, perioddata=[(1.0, 1, 1)] * ndays
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
            raise FileNotFoundError(f"Config file {config_file} not found on disk. Did you create the model first (load_from_disk = False)?")

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


def main():
    base_path = Path(__file__).parent
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        config = yaml.safe_load(file)

    # number of rows and columns to run benchmark
    rowscols = [(10, 10),
                (50, 50),
                (100, 100),
                (200, 200),
                (400, 400),
                (800, 800),
                (1000, 1000)]
    
    # dataframe to store the number of cells and the time it took to run the model
    df_time = pd.DataFrame(index=range(len(rowscols)), columns=['ncells', 'time', 'time_per_step'])

    NDAYS = 30  # number of days to run the model
    NAME = config['identifier']
    NLAY = 1  # number of layers
    SOILDEPTH = 1  # soil depth in meters
    ACQUIFER_THICKNESS = 30  # thickness of the aquifer in meters

    for i, nxny in enumerate(rowscols):
        nx = nxny[0]
        ny = nxny[1]
        domain = {
            'rowsize': config['dx'],
            'colsize': config['dy'],
            'nrow': nx,
            'ncol': ny,
        }
        modflow_basin = np.empty((NLAY, domain['nrow'], domain['ncol']))
        modflow_basin[:, :, :] = True
        modflow_basin = modflow_basin.astype(bool)

        # generate hillslope topography with constant slope
        arrays = [np.linspace(210, 200, domain['ncol']) for _ in range(domain['nrow'])]
        topography = np.concatenate(arrays).reshape(domain['nrow'], domain['ncol'])

        layer_boundaries = np.empty((NLAY + 1, domain['nrow'], domain['ncol']))
        layer_boundaries[0] = topography - SOILDEPTH - 0.05
        layer_boundaries[1] = layer_boundaries[0] - ACQUIFER_THICKNESS

        mf = ModFlowSimulation(
            NAME,
            base_path,
            ndays=NDAYS,
            nlay=NLAY,
            nrow=domain['nrow'],
            ncol=domain['ncol'],
            rowsize=domain['rowsize'],
            colsize=domain['colsize'],
            top=layer_boundaries[0],
            bottom=layer_boundaries[1],
            modflow_basin=modflow_basin,
            config=config,
            verbose=True
        )
        start = timer()
        for _ in range(NDAYS):
            mf.step()
        
        end = timer()
        mf.finalize()
        elapsed_time = (end-start)
        elapsed_time_seconds = timedelta(seconds=elapsed_time)
        df_time.iloc[i, 0] = nx * ny
        df_time.iloc[i, 1] = elapsed_time_seconds
        df_time.iloc[i, 2] = elapsed_time_seconds/NDAYS
        print("MODFLOW (transient) finalized simulation for {} cells in {} seconds".format(nx * ny, elapsed_time_seconds))
        df_time.to_csv(base_path / "benchmark_times_modflow_transient.csv", index=False, sep=";")


if __name__ == "__main__":
    main()
