import sys
from pathlib import Path
import flopy

import click

@click.option("-mr", "--model-run", type=int, default=0)
@click.command("main")
def main(model_run):
    try:
        print(sys.version)
        print(f"flopy version: {flopy.__version__}")

        base_path = Path(__file__).parent

        sim = flopy.mf6.MFSimulation.load(
            sim_ws=base_path / "output" / "steady-state",
            exe_name="mf6",
            version="mf6",
            verbosity_level=0,
        )

        ml = sim.get_model(f"dmn_run_{model_run}")

        # export groundwater head to netcdf
        fhead = base_path / "output" / "steady-state" / f"dmn_run_{model_run}.hds"
        hds = flopy.utils.HeadFile(fhead)

        export_dict = {"hds": hds}

        file = base_path / "output" / "steady-state" / f"modflow_output_run_{model_run}.nc"
        fnc = flopy.export.utils.output_helper(
            file, ml, export_dict,
        )
        # # write to csv
        # if "steady-state" == "steady-state":
        #     hds_data = hds.get_data()
        #     for i in range(4):
        #         file = base_path / "output" / "steady-state" / f"groundwater_heads_layer{i+1}.csv"
        #         hds_data_layer = hds_data[i, ...]
        #         mask = (hds_data_layer > 1200) | (hds_data_layer < -100)
        #         hds_data_layer[mask] = np.nan
        #         np.savetxt(file, hds_data_layer, delimiter=";")
    except:
        pass
    return


if __name__ == "__main__":
    main()
