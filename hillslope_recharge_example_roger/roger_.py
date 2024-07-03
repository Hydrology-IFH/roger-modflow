from pathlib import Path
import numpy as np
import yaml
from bmiroger import BmiRoger
from roger.bmimodels.svat import SVATSetup


def main():
    base_path = Path(__file__).parent
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        config = yaml.safe_load(file)

    # initialize the SVAT model of RoGeR using BMI
    model = SVATSetup(base_path)
    roger_interface = BmiRoger(model=model)
    roger_interface._model._output_dir = base_path / "output" / "transient"
    roger_interface.initialize(base_path)
    print("RoGeR model initialized")

    soildepth = np.zeros(roger_interface.get_grid_node_count())
    roger_interface.get_value("z_soil", soildepth)
    soildepth = soildepth / 1000  # mm to m
    
    # initialize the MODFLOW model using XMI
    NDAYS = config['ndays']
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

    layer_boundaries = np.empty((NLAY + 1, domain['nrow'], domain['ncol']))
    layer_boundaries[0] = topography - soildepth.reshape(domain['nrow'], domain['ncol']) - 0.05
    layer_boundaries[1] = layer_boundaries[0] - ACQUIFER_THICKNESS

    mask = modflow_basin[0, :, :]
    for _ in range(NDAYS):
        # update groundwater head
        groundwater_head = np.zeros(roger_interface.get_grid_node_count())
        groundwater_head[:] = topography.flatten() - 20
        # RoGeR requires depth of groundwater head (in meters)
        groundwater_depth = topography.flatten() - groundwater_head
        with roger_interface._model.state.variables.unlock():
            roger_interface.set_value("z_gw", groundwater_depth)

        # run RoGeR for one timestep
        roger_interface.update_until(roger_interface._model._config["OUTPUT_FREQUENCY"])

        # update recharge and pass it to MODFLOW
        recharge = np.zeros(roger_interface.get_grid_node_count())
        roger_interface.get_value("q_ss", recharge)
        recharge = recharge.reshape(domain['nrow'], domain['ncol']).astype(np.float64) / 1000  # mm/day to m/day
        recharge[~mask] = 0
        recharge = recharge.flatten()
    
    roger_interface.finalize()
    print("RoGeR finalized")

if __name__ == "__main__":
    main()
