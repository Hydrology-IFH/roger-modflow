import h5netcdf
import xarray as xr
from pathlib import Path
import numpy as np

# replace kf of layer1 with the values from the soil map data (BK50)
base_path = Path(__file__)
path = base_path.parent / "input" / "parameters_roger_50m.nc"
with xr.open_dataset(path) as df:
    kf_layer1 = (df.variables["TP"].values * (24/1000))  # convert from mm/h to m/day

kf_layer1[kf_layer1 <= 0] = 0.001
kf_layer1[np.isnan(kf_layer1)] = 0.001
# recalculate specific yield for layer1 following the formula of Marotz (1968)
sy_layer1 = 0.462 + 0.045 * np.log(kf_layer1/86400)
sy_layer1[sy_layer1 < 0.05] = 0.05

path = str(base_path.parent / "parameters_modflow.nc")
with h5netcdf.File(path, "a", decode_vlen_strings=False) as f:
    var_obj = f.variables.get("kf")
    var_obj[0, :, :] = kf_layer1

    var_obj = f.variables.get("sy")
    var_obj[0, :, :] = sy_layer1

