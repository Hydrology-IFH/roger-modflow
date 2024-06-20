from time import time
from pathlib import Path
import os
import numpy as np
from xmipy import XmiWrapper
import flopy
from flopy.utils import Raster
import platform
import yaml
import click

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
        verbose=False
    ):
        self.name = name.upper()  # MODFLOW requires the name to be uppercase
        self.folder = folder
        self.nrow = nrow
        self.ncol = ncol
        self.rowsize = rowsize
        self.colsize = colsize
        self.working_directory = os.path.join(folder, 'output/steady-state')
        if not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
        self.verbose = verbose

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
            sim, pname="tdis", time_units="DAYS", nper=1, perioddata=[(1.0, 1, 1)]
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
        file = base_path / "input" / "elevation.grd"
        layer_elevations = Raster.load(file)
        file = base_path / "input" / "domain.grd"
        domain = Raster.load(file)
        self.modflow_basin = (domain.get_array(1)[:, :-1] == 1)  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        self.n_active_cells = np.sum(self.modflow_basin)
        topography = layer_elevations.get_array(1)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        elevation_bottom_layer1 = layer_elevations.get_array(2)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        topography[topography <= elevation_bottom_layer1] = elevation_bottom_layer1[topography <= elevation_bottom_layer1]
        elevation_bottom_layer2 = layer_elevations.get_array(3)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        elevation_bottom_layer3 = layer_elevations.get_array(4)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        elevation_bottom_layer4 = layer_elevations.get_array(5)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        elevation_bottom_layer4[elevation_bottom_layer4 <= 100] = 100
        elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]
        domain_layer1 = domain.get_array(1)[:, :-1]
        domain_layers = [domain_layer1, domain_layer1, domain_layer1, domain_layer1]
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
            idomain=domain_layers
        )
        # Create the initial conditions package
        initial_conditions_layer1 = elevation_bottom_layer1
        initial_conditions_layer2 = elevation_bottom_layer2
        initial_conditions_layer3 = elevation_bottom_layer3
        initial_conditions_layer4 = ((elevation_bottom_layer3 - elevation_bottom_layer4)) / 2 + elevation_bottom_layer4
        # constrain the initial conditions to the average groundwater head of the closest groundwater gauge
        initial_conditions_layer1 = np.where(initial_conditions_layer1 < 224.5, 224.5, initial_conditions_layer1)
        initial_conditions_layer2 = np.where(initial_conditions_layer1 < 224.5, 224.5, initial_conditions_layer2)
        initial_conditions_layer3 = np.where(initial_conditions_layer1 < 224.5, 224.5, initial_conditions_layer3)
        initial_conditions_layer4 = np.where(initial_conditions_layer1 < 224.5, 224.5, initial_conditions_layer4)
        initial_conditions_layers = [initial_conditions_layer1, initial_conditions_layer2, initial_conditions_layer3, initial_conditions_layer4]
        ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=initial_conditions_layers)

        # Create the node property flow package with hydraulic conducitivities
        file = base_path / "input" / "hydraulic_conductivity_layer1.grd"
        hydraulic_conductivities_layer1 = Raster.load(file).get_array(1)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        file = base_path / "input" / "hydraulic_conductivity_layer2.grd"
        hydraulic_conductivities_layer2 = Raster.load(file).get_array(1)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        file = base_path / "input" / "hydraulic_conductivity_layer3.grd"
        hydraulic_conductivities_layer3 = Raster.load(file).get_array(1)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        file = base_path / "input" / "hydraulic_conductivity_layer1.grd"
        hydraulic_conductivities_layer4 = Raster.load(file).get_array(1)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        hydraulic_conductivities_layers = [hydraulic_conductivities_layer1, hydraulic_conductivities_layer2, hydraulic_conductivities_layer3, hydraulic_conductivities_layer4]
        npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
            gwf, pname="npf", icelltype=0, k=hydraulic_conductivities_layers, save_flows=True, wetdry=0.5
        )

        # create the storage package
        specific_yield = flopy.mf6.ModflowGwfsto.sy.empty(gwf, layered=True)
        file = base_path / "input" / "specific_yield_layer1.grd"
        specific_yield[0]["data"] = Raster.load(file).get_array(1)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        file = base_path / "input" / "specific_yield_layer2.grd"
        specific_yield[1]["data"] = Raster.load(file).get_array(1)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        file = base_path / "input" / "specific_yield_layer3.grd"
        specific_yield[2]["data"] = Raster.load(file).get_array(1)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        file = base_path / "input" / "specific_yield_layer4.grd"
        specific_yield[3]["data"] = Raster.load(file).get_array(1)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain

        # specific_storage = flopy.mf6.ModflowGwfsto.ss.empty(
        #     gwf, layered=True, default_value=0.000001
        # )

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

        # Create the constant head package
        file = base_path / "input" / "boundary_condition.grd"
        boundary_condition = Raster.load(file).get_array(1)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        index = np.where(boundary_condition == -1)
        rows_bc = index[0]
        cols_bc = index[1]

        chd_rec = []
        for layer in range(0, nlay):
            for ii in range(0, len(rows_bc)):
                # set constant head using initial conditions
                constant_head = initial_conditions_layers[layer][rows_bc[ii], cols_bc[ii]]
                chd_rec.append(((layer, rows_bc[ii], cols_bc[ii]), constant_head))

        chd = flopy.mf6.modflow.mfgwfchd.ModflowGwfchd(
            gwf,
            pname="chd",
            maxbound=len(chd_rec),
            stress_period_data=chd_rec,
            save_flows=True,
        )
            
        # Recharge package
        recharge = flopy.mf6.ModflowGwfrcha(gwf, recharge=1/1000000, fixed_cell=True)
        
        # create streamflow routing package
        # Prepare the reach data
        file = base_path / "input" / "river_reach.csv"
        reach_data = np.genfromtxt(file, delimiter=',', names=True)
        nstrm = len(reach_data)
        reach_data_merged = np.column_stack((reach_data['k'],reach_data['i'],reach_data['j'])).astype(int)
        reach_data_merged  = tuple(map(tuple,reach_data_merged))

        reaches = []
        for index in range(0, nstrm):
            reaches.append((reach_data['rno'][index].astype(int)-1, reach_data_merged[index],reach_data['rlen'][index],
                        reach_data['rwid'][index],reach_data['rgrd'][index],reach_data['rtp'][index],
                        reach_data['rbth'][index],reach_data['rhk'][index],
                        reach_data['man'][index],
                        reach_data['ncon'][index],reach_data['ustrf'][index],reach_data['ndv'][index]))

        # Prepare connection data
        file = base_path / "input" / "river_reach_hydraulic_conductivity.csv"
        reach_connection_data = np.genfromtxt(file, delimiter=',', names=True, missing_values='nan')
        direction = reach_connection_data['ic1']/np.absolute(reach_connection_data['ic1'])
        zw1 = direction*(np.absolute(reach_connection_data['ic1'])-1)
        direction = reach_connection_data['ic2']/np.absolute(reach_connection_data['ic2'])
        zw2 = direction*(np.absolute(reach_connection_data['ic2'])-1)
        direction = reach_connection_data['ic3']/np.absolute(reach_connection_data['ic3'])
        zw3 = direction*(np.absolute(reach_connection_data['ic3'])-1)
        zw_merged = np.column_stack((zw1,zw2,zw3)).astype(int)
        connection = []
        for index in range(0, nstrm):
            length = sum(zw_merged[index] > -1000000000)
            selection = []
            selection.append(reach_connection_data['rno'][index]-1)
            for i2 in range(0,length):
                selection.append(zw_merged[index][i2])
            selection = tuple(selection)
            connection.append((selection))

        sfr = flopy.mf6.modflow.mfgwfsfr.ModflowGwfsfr(gwf, pname="sfr",
            time_conversion=86400, nreaches=nstrm, packagedata=reaches, 
            connectiondata=connection,
            save_flows=True)


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
        config_file = self.folder / 'output' / 'steady-state' / 'mfsim.nam'
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

@click.command("main", short_help="Run MODFLOW in steady-state mode")
def main():
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        roger_config = yaml.safe_load(file)

    res_modflow = 50  # spatial resolution of MODFLOW in meters

    modflow_config = {
        'dx': int(roger_config['dx']*(res_modflow/roger_config['dx'])),
        'dy': int(roger_config['dy']*(res_modflow/roger_config['dx'])),
        'nx': int(roger_config['nx']/(res_modflow/roger_config['dx'])),
        'ny': int(roger_config['ny']/(res_modflow/roger_config['dx'])),
        'nz': 4,
    }
    
    # initialize the MODFLOW model using XMI
    modflow_interface = ModFlowSimulation(
        roger_config['identifier'],
        base_path,
        nlay=4,
        nrow=modflow_config['nx'],
        ncol=modflow_config['ny'],
        rowsize=modflow_config['dx'],
        colsize=modflow_config['dy'],
        verbose=True
    )
    # run MODFLOW for one timestep
    modflow_interface.step()
    
    modflow_interface.finalize()
    print("MODFLOW (steady-state) finalized")

if __name__ == "__main__":
    main()
