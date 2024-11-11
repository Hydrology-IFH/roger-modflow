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
import click
import shutil
import subprocess


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
        roger_config,
        verbose=False
    ):
        self.name = name.upper()  # MODFLOW requires the name to be uppercase
        self.folder = folder
        self.nrow = nrow
        self.ncol = ncol
        self.rowsize = rowsize
        self.colsize = colsize
        self.roger_config = roger_config
        self.working_directory = os.path.join(folder, 'output/transient')
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
        file = base_path / "input" / "elevation.grd"
        layer_elevations = Raster.load(file)
        file = base_path / "input" / "domain.grd"
        domain = Raster.load(file)
        self.modflow_basin = (domain.get_array(1)[:, :-1] == 1)  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        self.n_active_cells_per_layer = np.sum(self.modflow_basin)
        topography = layer_elevations.get_array(1)[:, :-1]
        elevation_bottom_layer1 = layer_elevations.get_array(2)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        topography[topography <= elevation_bottom_layer1] = elevation_bottom_layer1[topography <= elevation_bottom_layer1]
        elevation_bottom_layer2 = layer_elevations.get_array(3)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        elevation_bottom_layer3 = layer_elevations.get_array(4)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        elevation_bottom_layer4 = layer_elevations.get_array(5)[:, :-1]  # [:, :-1]=leave last column out to match the shape of the RoGeR domain
        elevation_bottom_layers = [elevation_bottom_layer1, elevation_bottom_layer2, elevation_bottom_layer3, elevation_bottom_layer4]
        elevation_bottom_layer4[elevation_bottom_layer4 <= 100] = 100
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
        file = base_path / "output" / "steady-state" / "moehlin.hds"
        hds = flopy.utils.HeadFile(file).get_data()
        initial_conditions_layer1 = hds[0, ...]
        initial_conditions_layer2 = hds[1, ...]
        initial_conditions_layer3 = hds[2, ...]
        initial_conditions_layer4 = hds[3, ...]
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

        specific_storage = flopy.mf6.ModflowGwfsto.ss.empty(
            gwf, layered=True, default_value=0.000001
        )

        sto = flopy.mf6.ModflowGwfsto(gwf, pname="sto",
            iconvert=1, ss=specific_storage, sy=specific_yield, steady_state=False, transient=True)

        # Create the constant head package
        file = base_path / "input" / "boundary_condition.grd"
        boundary_condition = Raster.load(file).get_array(1)[:, :-1]
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
        config_file = self.folder / 'output' / 'transient' / 'mfsim.nam'
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

@click.option("-b", "--backend", type=click.Choice(["numpy", "jax"]), default="jax", help="Computational backend of RoGeR")
@click.option("-ft", "--float_type", type=click.Choice(["float32", "float64"]), default="float32", help="Float type of RoGeR")
@click.command("main", short_help="Run MODFLOW in transient mode coupled with RoGeR.")
def main(backend, float_type):
    from roger import runtime_settings
    runtime_settings.update(
    backend=backend,
    float_type=float_type,
    )
    from bmiroger import BmiRoger
    from roger.bmimodels.oneD import ONEDSetup
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        roger_config = yaml.safe_load(file)

    # define the output variables of RoGeR
    roger_config['OUTPUT_COLLECT'] = ["theta", "z_gw"]
    roger_config['OUTPUT_RATE'] = ["q_hof", "q_ss"]

    # choose parameters depending on the resolution of RoGeR
    file1 = base_path / "input" / f"parameters_{int(roger_config['dx'])}.nc"
    file2 = base_path / "parameters.nc"
    shutil.copy(file1, file2)
    file = base_path / "write_parameters_to_csv_for_ONED.py"
    subprocess.run(["python", str(file)], check=True, timeout=20)

    # set the number of grid cells in x and y direction from the parameters file
    file = base_path / "parameters.nc"
    with xr.open_dataset(file, engine="h5netcdf") as ds:
        roger_config['nx'] = ds.sizes['y']
        roger_config['ny'] = ds.sizes['x']

    # save the updated config file
    with open(file_config, "w") as file:
        yaml.dump(roger_config, file)

    res_modflow = 50  # spatial resolution of MODFLOW in meters

    modflow_config = {
        'dx': int(roger_config['dx']*(res_modflow/roger_config['dx'])),
        'dy': int(roger_config['dy']*(res_modflow/roger_config['dx'])),
        'nx': int(roger_config['nx']/(res_modflow/roger_config['dx'])),
        'ny': int(roger_config['ny']/(res_modflow/roger_config['dx'])),
        'nz': 4,
    }

    # load the spatial domain and elevation data
    file = base_path / "input" / "domain.grd"
    domain = Raster.load(file)
    modflow_mask = (domain.get_array(1)[:, :-1] == 1)
    file = base_path / "input" / "elevation.grd"
    layer_elevations = Raster.load(file)
    topography = layer_elevations.get_array(1)[:, :-1]
    topography = aggregate_to_finer_resolution(topography, modflow_config['dx'], roger_config['dx'], method="keep")

    # initialize the SVAT model of RoGeR using BMI
    model = ONEDSetup(base_path)
    roger_interface = BmiRoger(model=model)
    roger_interface._model._output_dir = base_path / "output" / "transient"
    roger_interface.initialize(base_path)
    print("RoGeR model initialized")
    NDAYS = int(roger_interface.get_end_time() / (60 * 60 * 24))  # seconds to days

    soildepth = np.zeros(roger_interface.get_grid_node_count())
    roger_interface.get_value("z_soil", soildepth)
    soildepth = soildepth / 1000  # mm to m
    roger_mask = np.empty(roger_interface.get_grid_node_count(), dtype=bool)
    roger_interface.get_value("maskCatch", roger_mask)
    roger_mask = roger_mask.reshape(roger_config['nx'], roger_config['ny'])
    
    # initialize the MODFLOW model using XMI
    modflow_interface = ModFlowSimulation(
        roger_config['identifier'],
        base_path,
        ndays=NDAYS,
        nlay=modflow_config['nz'],
        nrow=modflow_config['nx'],
        ncol=modflow_config['ny'],
        rowsize=modflow_config['dx'],
        colsize=modflow_config['dy'],
        roger_config=roger_config,
        verbose=True
    )
    for _ in range(NDAYS):
        # update groundwater head
        groundwater_head = np.zeros(modflow_config['nx'] * modflow_config['ny'])
        modflow_interface.get_groundwater_head(groundwater_head)
        groundwater_head = groundwater_head.reshape(modflow_config['nx'], modflow_config['ny'])
        # aggregate groundwater head to the resolution of RoGeR
        groundwater_head = aggregate_to_finer_resolution(groundwater_head, modflow_config['dx'], roger_config['dx'], method="keep")
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
        recharge = recharge.reshape(roger_config['nx'], roger_config['ny']).astype(np.float64) / 1000  # mm/day to m/day
        recharge = aggregate_to_coarser_resolution(recharge, roger_config['dx'], modflow_config['dx'], method="average")
        recharge[~modflow_mask] = np.nan
        recharge = recharge.flatten()
        modflow_interface.set_recharge(recharge)
    
        # run MODFLOW for one timestep
        modflow_interface.step()

    roger_interface.finalize()
    modflow_interface.finalize()
    print("RoGeR and MODFLOW (transient) finalized")

if __name__ == "__main__":
    main()
