from time import time
from pathlib import Path
import matplotlib.pyplot as plt
import os
import rasterio
import numpy as np
from xmipy import XmiWrapper
import flopy
import platform
import yaml
import xarray as xr


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
        self.working_directory = os.path.join(folder, 'output/steady-state')
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.verbose = verbose

        # values to generate constant head boundary conditions
        h1 = np.max(top) - 2
        h2 = np.min(top) - 2
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
            sim, pname="tdis", time_units="DAYS", start_date_time=str(self.config['t_origin']), nper=1, perioddata=[(1.0, 1, 1)]
        )

        # Create the Flopy groundwater flow (gwf) model object
        model_nam_file = "{}.nam".format(name)
        gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file)

        # Create the Flopy iterative model solver (ims) Package object
        ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", complexity="SIMPLE")

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
        start = np.mean(h1_h2) * np.ones((nlay, self.nrow, self.ncol)) - 2
        ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=start)

        # Create the node property flow package with hydraulic conducitivities
        npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
            gwf, pname="npf", icelltype=1, k=k, save_flows=True
        )

        # create the storage package
        sto = flopy.mf6.ModflowGwfsto(gwf, pname="sto",
            iconvert=1, ss=0.00001, sy=0.15,  steady_state={0: True})

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
        

        # Evapotranspiration package for capillary rise
        cpr = np.zeros((self.modflow_basin.sum(), 8), dtype=np.int32)
        cpr_locations = np.where(self.modflow_basin == True)  # only set capillary rise where modflow_basin is True
        # 0: layer, 1: y-idx, 2: x-idx, 3: rate
        cpr[:, 1] = cpr_locations[1]
        cpr[:, 2] = cpr_locations[2]
        cpr = cpr.tolist()

        cpr = flopy.mf6.ModflowGwfevt(gwf, fixed_cell=False,
                            print_input=False, print_flows=False,
                            save_flows=False, boundnames=None,
                            maxbound=self.modflow_basin.sum(), stress_period_data=cpr)


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
        config_file = Path(self.working_directory) / 'mfsim.nam'
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

        recharge_tag = self.mf6.get_var_address("RECHARGE", self.name, "RCH_0")
        # there seems to be a bug in xmipy where the size of the pointer to RCHA is
        # is the size of the entire modflow area, including modflow_basined cells. Only the first
        # part of the array is actually used, when a part of the area is modflow_basined. Since
        # numpy returns a view of the array when the array[]-syntax is used, we can simply
        # use the view of the first part of the array up to the number of active
        # (non-modflow_basined) cells
        self.recharge = self.mf6.get_value_ptr(recharge_tag)

        cpr_tag = self.mf6.get_var_address("RATE", self.name, "EVT_0")
        self.cpr = self.mf6.get_value_ptr(cpr_tag)
        
        head_tag = self.mf6.get_var_address("X", self.name)
        self.head = self.mf6.get_value_ptr(head_tag)

        mxit_tag = self.mf6.get_var_address("MXITER", "SLN_1")
        self.max_iter = self.mf6.get_value_ptr(mxit_tag)[0]

        self.prepare_time_step()

    def compress(self, a):
        return np.compress(self.modflow_basin, a)

    def decompress(self, a):
        o = np.empty(self.modflow_basin.shape, dtype=a.dtype)
        o[self.modflow_basin] = a
        return o

    def prepare_time_step(self):
        dt = self.mf6.get_time_step()
        self.mf6.prepare_time_step(dt)

    def set_recharge(self, recharge):
        """Set recharge, value in m/day"""
        recharge_tag = self.mf6.get_var_address("RECHARGE", self.name, "RCH_0")
        self.mf6.set_value(recharge_tag, recharge)

    def set_capillary_rise(self, cpr):
        """Set capillary rise, value in m/day"""
        cpr_tag = self.mf6.get_var_address("RATE", self.name, "EVT_0")
        self.mf6.set_value(cpr_tag, cpr)

    def set_evt_surface(self, val):
        """Set elevation of the ET surface, value in m"""
        tag = self.mf6.get_var_address("SURFACE", self.name, "EVT_0")
        self.mf6.set_value(tag, val)

    def set_evt_depth(self, val):
        """Set ET extinction depth, value in m"""
        tag = self.mf6.get_var_address("DEPTH", self.name, "EVT_0")
        self.mf6.set_value(tag, val)

    def set_evt_pxdp(self, val):
        """Set proportion of the ET extinction depth at the bottom of a segment, value in -"""
        tag = self.mf6.get_var_address("PXDP", self.name, "EVT_0")
        self.mf6.set_value(tag, val)

    def set_evt_petm(self, val):
        """Set proportion of the maximum ET flux rate at the bottom of a segment, value in -"""
        tag = self.mf6.get_var_address("PETM", self.name, "EVT_0")
        self.mf6.set_value(tag, val)

    def set_evt_petm0(self, val):
        """Set proportion of the maximum ET flux rate, value in -"""
        tag = self.mf6.get_var_address("PETM0", self.name, "EVT_0")
        self.mf6.set_value(tag, val)

    def get_groundwater_head(self, groundwater_head):
        """Set groundwater head, value in m"""
        head_tag = self.mf6.get_var_address("X", self.name)
        groundwater_head[:] = self.mf6.get_value_ptr(head_tag)

    def step(self, plot=False):
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

        if plot:
            _, (ax0, ax1, ax2) = plt.subplots(1, 3)
            
            # Recharge
            recharge = self.recharge / (self.rowsize * self.colsize)
            self.plot_compressed(recharge, ax=ax0)
            ax0.set_title('Recharge')

            # Head
            self.plot_compressed(self.head, ax=ax1)
            ax1.set_title('Head')

            # Pumping
            self.plot_compressed(self.well_rate / (self.rowsize * self.colsize), ax=ax2)
            ax2.set_title('Well rate')

            plt.show()
        
        # If next step exists, prepare timestep. Otherwise the data set through the bmi
        # will be overwritten when preparing the next timestep.
        if self.mf6.get_current_time() < self.end_time:
            self.prepare_time_step()

    def plot(self, a, ax=None):
        show = True if ax is None else False
        a = a.reshape(self.nrow, self.ncol)
        if a.dtype in ('float64', 'float32'):
            a[~self.modflow_basin] = np.nan
        
        if ax:
            ax.imshow(a)
        else:
            plt.imshow(a)
        if show:
            plt.show()

    def plot_compressed(self, a, ax=None):
        a = self.decompress(a)
        self.plot(a, ax=ax)

    def finalize(self):
        self.mf6.finalize()


def main():
    base_path = Path(__file__).parent
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        config = yaml.safe_load(file)

    # load the parameter file of RoGeR to get the soil depth
    params_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_file, engine="h5netcdf")
    ds_params["z_soil"].values.flatten()
    soildepth = ds_params["z_soil"].values.flatten() / 1000  # mm to m
    
    # initialize the MODFLOW model using XMI
    NDAYS = 1
    NAME = config['identifier']
    NLAY = config['nz']
    ACQUIFER_THICKNESS = config['aquifer_thickness']

    domain = {
        'rowsize': config['dx'],
        'colsize': config['dy'],
        'nrow': config['nx'],
        'ncol': config['ny'],
    }

    modflow_basin = np.empty((NLAY, domain['nrow'], domain['ncol']))
    modflow_basin[:, :, :] = True
    modflow_basin = modflow_basin.astype(bool)

    # generate hillslope topography with constant slope
    arrays = [np.linspace(210, 200, domain['ncol']) for _ in range(domain['nrow'])]
    topography = np.concatenate(arrays).reshape(domain['nrow'], domain['ncol'])

    grid_extent = (0, config['nx'] * config['dx'], 0, config['ny'] * config['dy'])
    fig, axes = plt.subplots(figsize=(4, 4))
    plt.imshow(topography.T, extent=grid_extent, cmap='terrain', vmin=200, vmax=210)
    plt.colorbar(label='[m a.s.l.]', shrink=0.5)
    plt.grid(zorder=0)
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.tight_layout()
    file = base_path / f"elevation.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    layer_boundaries = np.empty((NLAY + 1, domain['nrow'], domain['ncol']))
    layer_boundaries[0] = topography - soildepth.reshape(domain['nrow'], domain['ncol']) - 0.05
    layer_boundaries[1] = layer_boundaries[0] - ACQUIFER_THICKNESS

    modflow_interface = ModFlowSimulation(
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
    mask = modflow_basin[0, :, :]
    # run MODFLOW for one timestep
    # update recharge and pass it to MODFLOW
    recharge = np.zeros(domain['nrow'] * domain['ncol'])
    recharge = recharge.reshape(domain['nrow'], domain['ncol']).astype(np.float64)
    recharge[:, :] = 0
    recharge[~mask] = np.nan
    modflow_interface.set_recharge(recharge)
    modflow_interface.step(plot=False)
    
    modflow_interface.finalize()
    print("MODFLOW (steady-state) finalized")

if __name__ == "__main__":
    main()
