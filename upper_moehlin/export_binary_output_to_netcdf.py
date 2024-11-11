import sys
from pathlib import Path
import numpy as np

import flopy

print(sys.version)
print(f"flopy version: {flopy.__version__}")

base_path = Path(__file__).parent

for model_type in ["steady-state", "transient"]:
    sim = flopy.mf6.MFSimulation.load(
        sim_ws=base_path / "output" / model_type,
        exe_name="mf6",
        version="mf6",
        verbosity_level=0,
    )

    ml = sim.get_model("moehlin")

    # export groundwater head to netcdf
    fhead = base_path / "output" / model_type / "moehlin.hds"
    hds = flopy.utils.HeadFile(fhead)

    export_dict = {"hds": hds}

    file = base_path / "output" / model_type / "modflow_output.nc"
    fnc = flopy.export.utils.output_helper(
        file, ml, export_dict,
    )

    # write to csv
    if model_type == "steady-state":
        hds_data = hds.get_data()
        for i in range(4):
            file = base_path / "output" / model_type / f"groundwater_heads_layer{i+1}.csv"
            hds_data_layer = hds_data[i, ...]
            mask = (hds_data_layer > 1200) | (hds_data_layer < -100)
            hds_data_layer[mask] = np.nan
            np.savetxt(file, hds_data_layer, delimiter=";")