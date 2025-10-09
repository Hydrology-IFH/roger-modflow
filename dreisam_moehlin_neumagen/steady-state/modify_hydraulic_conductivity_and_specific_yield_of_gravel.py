import h5netcdf
from pathlib import Path
import numpy as np

base_path = Path(__file__).parent

path = str(base_path / "input" / "parameters_modflow.nc")
with h5netcdf.File(path, "a", decode_vlen_strings=False) as f:
    mask_zarten_brugga = (f.variables.get('mask_zarten_brugga')[:, :] == 1)
    mask_zarten_gravel_north = (f.variables.get('mask_zarten_gravel_north')[:, :] == 1)
    mask_staufen_gravel = (f.variables.get('mask_staufen_gravel')[:, :] == 1)
    mask_gravel = mask_zarten_brugga | mask_zarten_gravel_north | mask_staufen_gravel

    kf_layer2 = f.variables.get("kf")[1, :, :]
    kf_layer3 = f.variables.get("kf")[2, :, :]
    kf_layer2[mask_zarten_brugga] = 86.4
    kf_layer2[mask_zarten_gravel_north] = 259.2
    kf_layer2[mask_staufen_gravel] = 345.6
    kf_layer3[mask_zarten_brugga] = 86.4
    kf_layer3[mask_zarten_gravel_north] = 259.2
    kf_layer3[mask_staufen_gravel] = 345.6

    sy_layer2 = 0.462 + 0.045 * np.log(kf_layer2/86400)
    sy_layer2[sy_layer2 < 0.05] = 0.05
    sy_layer3 = 0.462 + 0.045 * np.log(kf_layer3/86400)
    sy_layer3[sy_layer3 < 0.05] = 0.05 
    
    var_obj = f.variables.get("kf")
    var_obj[1, :, :] = kf_layer2
    var_obj[2, :, :] = kf_layer3

    var_obj = f.variables.get("sy")
    var_obj[1, :, :] = sy_layer2
    var_obj[2, :, :] = sy_layer3

