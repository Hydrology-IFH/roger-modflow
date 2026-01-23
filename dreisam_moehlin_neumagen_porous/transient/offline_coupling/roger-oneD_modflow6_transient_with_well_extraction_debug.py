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
import click

base_path = Path(__file__).parent

file_config = base_path.parent / "config_modflow.yml"
with open(file_config, "r") as file:
    config_modflow = yaml.safe_load(file)

config_modflow["nz"] = 1
config_modflow["ny"] = 10
config_modflow["nx"] = 20

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
            sim, pname="tdis", time_units="DAYS", nper=1, perioddata=[(1.0, ndays, 1.0)]
        )
        # Create the Flopy groundwater flow (gwf) model object
        model_nam_file = "{}.nam".format(name)
        gwf = flopy.mf6.ModflowGwf(sim, modelname=name, model_nam_file=model_nam_file, save_flows=True, newtonoptions="NEWTON")

        # Create the Flopy iterative model solver (ims) Package object
        ims = flopy.mf6.modflow.mfims.ModflowIms(sim, pname="ims", print_option="summary", linear_acceleration="bicgstab")
        sim.register_solution_package(ims, [gwf.name])

        # Now that the overall simulation is set up, we can focus on building the groundwater flow model.  The groundwater flow model will be built by adding packages to it that describe the model characteristics.
        #
        # Define the discretization of the model. All layers are given equal thickness. The `bot` array is build from `H` and the `Nlay` values to indicate top and bottom of each layer, and `delrow` and `delcol` are computed from model size `L` and number of cells `N`. Once these are all computed, the Discretization file is built.

        # Create the discretization package
        # load elevation data of the layers
        topography = np.zeros((nrow, ncol))
        topography[:, :] = 100.0  # flat topography for debugging
        elevation_bottom_layer1 = np.zeros((nrow, ncol))
        elevation_bottom_layer1[:, :] = 90.0  # flat bottom for debugging
        elevation_bottom_layers = [elevation_bottom_layer1]

        mask = np.isfinite(topography)
        domain = np.empty_like(topography, dtype=np.int32)
        domain[mask] = 1
        domain[~mask] = -1
        self.modflow_basin = mask
        self.n_active_cells_per_layer = np.nansum(self.modflow_basin)
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
            botm=elevation_bottom_layer1,
            idomain=domain,
        )

        # Create the initial conditions package
        # use interpolated groundwater heads from well observations as initial conditions
        initial_conditions_layers = [topography - 5.0]
        ic = flopy.mf6.modflow.mfgwfic.ModflowGwfic(gwf, pname="ic", strt=initial_conditions_layers)

        npf = flopy.mf6.modflow.mfgwfnpf.ModflowGwfnpf(
            gwf, pname="npf", icelltype=0, k=0.1, wetdry=0.5, save_flows=True, save_specific_discharge="budget save file"
        )
        
        sto = flopy.mf6.ModflowGwfsto(gwf, pname="sto",
            iconvert=1, ss=0.0001, sy=0.1, transient={0: True})

        # Create the constant head package (Dirichlet boundary condition i.e. first type)
        mask_boundary_condition_porous_aquifer = np.zeros((nrow, ncol))
        mask_boundary_condition_porous_aquifer[0, :] = 1  # set first row as constant head boundary for debugging
        index = np.where(mask_boundary_condition_porous_aquifer == 1)
        rows_bc = index[0]
        cols_bc = index[1]

        chd_rec = []
        for ii in range(0, len(rows_bc)):
            constant_head = topography[rows_bc[ii], cols_bc[ii]] - 2.0  # constant head 2 m below topography for debugging
            chd_rec.append(((0, rows_bc[ii], cols_bc[ii]), constant_head))

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
        recharge[:, 3] = 0.001
        recharge = recharge.tolist()

        recharge = flopy.mf6.ModflowGwfrch(gwf, fixed_cell=False,
                            print_input=False, print_flows=False,
                            save_flows=False, boundnames=None,
                            maxbound=self.modflow_basin.sum(), stress_period_data=recharge)
        
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
            print(f"Failed to load {library_path}")
            print("with message: " + str(e))
            return self.bmi_return(success, self.working_directory)

        # modflow requires the real path (no symlinks etc.)
        config_file = self.folder / "output" / "mfsim.nam"
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found on disk. Did you create the model first (load_from_disk = False)?")

        try:
            # initialize the model
            self.mf6.initialize(str(config_file))
        except:
            return self.bmi_return(success, str(self.folder / "output"))

        if self.verbose:
            print("MODFLOW model initialized")

        
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

    def set_well_rate(self, well_rate):
        """Set pumping rate, value in m/day"""
        well_tag = self.mf6.get_var_address("Q", self.name, "WEL_0")
        self.mf6.set_value(well_tag, well_rate)

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

@click.command("main", short_help="Run MODFLOW in transient mode coupled with RoGeR.")
def main():

    # NDAYS = 365 * 11 + 2
    NDAYS = 365
    NDAYS = int(NDAYS)
    # generate random arrays for testing
    RNG = np.random.default_rng(42)
    recharge_weights = RNG.uniform(0, 1, size=NDAYS)

    # initialize the MODFLOW model using XMI
    # f"{stress_test_meteo}-m{stress_test_meteo_magnitude}-d{stress_test_meteo_duration}_{_irrig}_{_yellow_mustard}_{_soil_compaction}",
    modflow_interface = ModFlowSimulation(
        "test",
        base_path,
        ndays=float(NDAYS),
        nlay=1,
        nrow=config_modflow["ny"],
        ncol=config_modflow["nx"],
        rowsize=config_modflow["dx"],
        colsize=config_modflow["dy"],
        verbose=True
    )

    # doys = ds_roger_simulation["time"].dt.dayofyear.values
    # years = ds_roger_simulation["time"].dt.year.values
    doys = np.array([(i % 365) + 1 for i in range(NDAYS)])
    years = np.array([2013 + (i // 365) for i in range(NDAYS)])

    for i in range(NDAYS):
        # year = years[i]
        # doy = doys[i]
        # daily_weights_drinking_water_supply_year_doy = daily_weights_drinking_water_supply.loc[int(year), f"{int(doy)}"]
        # if i > 1:
        #     if years[i] > years[i-1]:
        #         for index, row in groundwater_extraction_drinking_water_supply.iterrows():
        #             well_extraction_rate_drinking_water_supply[row["layer"], row["cell_y"], row["cell_x"]] = row[f"{year}"]
                    
        # update groundwater head
        groundwater_head = np.zeros(config_modflow['ny'] * config_modflow['nx'])
        modflow_interface.get_groundwater_head(groundwater_head)
        groundwater_head = groundwater_head.reshape(config_modflow['ny'], config_modflow['nx'])
        print(groundwater_head[5, 10])
        # # aggregate groundwater head to the resolution of RoGeR
        # groundwater_head = aggregate_to_finer_resolution(groundwater_head, config_modflow['dx'], 25, method="keep")
        # # RoGeR requires depth of groundwater head (in meters)
        # groundwater_depth = topography.flatten() - groundwater_head.flatten()
        # groundwater_depth[(groundwater_depth <= soildepth)] = soildepth[(groundwater_depth <= soildepth)] + 0.05

        recharge = np.zeros(config_modflow['ny'] * config_modflow['nx'])
        recharge = recharge.flatten()
        recharge[:] = 0.002 * recharge_weights[i]
        modflow_interface.set_recharge(recharge)
        recharge = np.zeros(config_modflow['ny'] * config_modflow['nx'])
        modflow_interface.get_recharge(recharge)
        recharge = recharge.reshape(config_modflow['ny'], config_modflow['nx'])
        print(recharge[5, 10])

        # # update recharge and pass it to MODFLOW
        # # recharge = ds_roger_simulation["q_ss"].isel(time=i).values.flatten()
        # recharge_ = aggregate_to_finer_resolution(ds_bc["recharge"].values, config_modflow['dx'], 25, method="keep")
        # recharge = recharge_.flatten() * recharge_weights[i]  # apply recharge weight
        # recharge[(groundwater_depth <= soildepth)] = 0 # constrain recharge to zero where groundwater depth is equal to soil depth
        # recharge = recharge.reshape(config_modflow['ny'] * 2, config_modflow['nx'] * 2).astype(np.float64) / 1000  # mm/day to m/day
        # recharge = aggregate_to_coarser_resolution(recharge, 25, config_modflow['dx'], method="average")
        # recharge[:, :] = 0
        # recharge = recharge.flatten()
        # modflow_interface.set_recharge(recharge)

        # # update well rate and pass it to MODFLOW
        # well_extraction_rate = np.zeros((config_modflow['nz'], config_modflow['ny'], config_modflow['nx']))
        # well_extraction_rate[modflow_interface.well_mask] = modflow_interface.well_extraction_rate[modflow_interface.well_mask]
        # well_extraction_rate[mask_drinking_water_supply] = well_extraction_rate_drinking_water_supply[mask_drinking_water_supply] * daily_weights_drinking_water_supply_year_doy
        # well_extraction_rate = well_extraction_rate.flatten()
        # modflow_interface.set_well_rate(well_extraction_rate)
    
        # run MODFLOW for one timestep
        modflow_interface.step()

    modflow_interface.finalize()
    print("MODFLOW (transient) finalized")

if __name__ == "__main__":
    main()