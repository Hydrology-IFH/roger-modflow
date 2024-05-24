from pathlib import Path
import os
import pandas as pd
import numpy as np
import yaml
from timeit import default_timer as timer
from datetime import timedelta
import subprocess
import click

@click.option("-b", "--backend", type=click.Choice(["numpy", "jax"]), default="numpy")
@click.command("main", short_help="Run RoGeR benchmark.")
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
    df_time = pd.DataFrame(index=range(len(rowscols)), columns=['ncells', 'time', 'time_per_step', 'backend'])

    NDAYS = 30
    NAME = config['identifier']

    for i, nxny in enumerate(rowscols):
        nx = nxny[0]
        ny = nxny[1]
        domain = {
            'rowsize': config['dx'],
            'colsize': config['dy'],
            'nrow': nx,
            'ncol': ny,
        }
        # initialize the SVAT model of RoGeR using BMI
        # modify the config file to set the number of rows and columns
        config['nx'] = nx
        config['ny'] = ny
        with open(file_config, "w") as file:
            yaml.dump(config, file)
        # write the parameters for the SVAT model
        file = base_path / "write_parameters.py"
        subprocess.run(["python", str(file), f"--nrows={nx}", f"--ncols={ny}"], check=True, timeout=10)
        model = SVATSetup(base_path)
        roger_interface = BmiRoger(model=model)
        roger_interface._model._output_dir = base_path / "output"
        roger_interface.initialize(base_path)
        print("RoGeR model initialized")

        for t in range(NDAYS):
            if t == 3:
                start = timer()
            # update groundwater head
            groundwater_depth = np.zeros(roger_interface.get_grid_node_count())
            # RoGeR requires depth of groundwater head (in meters)
            groundwater_depth[:] = 50
            roger_interface.set_value("z_gw", groundwater_depth)

            # run RoGeR for one timestep
            roger_interface.update_until(roger_interface._model._config["OUTPUT_FREQUENCY"])

            # update recharge and pass it to MODFLOW
            recharge = np.zeros(roger_interface.get_grid_node_count())
            roger_interface.get_value("q_ss", recharge)
            recharge = recharge.reshape(domain['nrow'], domain['ncol']).astype(np.float64) / 1000  # mm/day to m/day

        end = timer()
        roger_interface.finalize()
        elapsed_time = (end-start)
        elapsed_time_seconds = timedelta(seconds=elapsed_time)
        df_time.iloc[i, 0] = nx * ny
        df_time.iloc[i, 1] = elapsed_time_seconds
        df_time.iloc[i, 2] = elapsed_time_seconds/(NDAYS-3)
        df_time.iloc[i, 3] = backend
        print("RoGeR finalized simulation for {} cells in {} seconds".format(nx * ny, elapsed_time_seconds))
        df_time.to_csv(base_path / f"benchmark_times_roger_{backend}.csv", index=False, sep=";")
        # remove the output files
        os.remove(base_path / "output" / f"{NAME}.collect.nc")
        os.remove(base_path / "output" / f"{NAME}.rate.nc")
        os.remove(base_path / "output" / f"{NAME}.maximum.nc")
    return

if __name__ == "__main__":
    main()
