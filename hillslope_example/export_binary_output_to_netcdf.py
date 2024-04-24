import sys
from pathlib import Path

import flopy

print(sys.version)
print(f"flopy version: {flopy.__version__}")

base_path = Path(__file__).parent

sim = flopy.mf6.MFSimulation.load(
    sim_ws=base_path / "output",
    exe_name="mf6",
    version="mf6",
    verbosity_level=0,
)

ml = sim.get_model("hillslope")

# export groundwater head to netcdf
fhead = base_path / "output" / "hillslope.hds"
hds = flopy.utils.HeadFile(fhead)

export_dict = {"hds": hds}

file = base_path / "output" / "output.nc"
fnc = flopy.export.utils.output_helper(
    file, ml, export_dict
)