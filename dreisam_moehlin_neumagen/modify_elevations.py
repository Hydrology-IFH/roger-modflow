import h5netcdf
from pathlib import Path

# shift elevations to avoid zero values
base_path = Path(__file__).parent
path = str(base_path / "parameters_modflow.nc")
with h5netcdf.File(path, "a", decode_vlen_strings=False) as f:
    var_obj = f.variables.get("elevations")

    topography = var_obj[0, :, :]
    elevation_bottom_layer1 = var_obj[1, :, :]
    elevation_bottom_layer2 = var_obj[2, :, :]
    elevation_bottom_layer3 = var_obj[3, :, :]
    elevation_bottom_layer4 = var_obj[4, :, :]

    elevation_bottom_layer3[elevation_bottom_layer3 <= 100] = 100
    elevation_bottom_layer4[elevation_bottom_layer4 <= 100] = 100
    topography[topography <= elevation_bottom_layer1] = elevation_bottom_layer1[topography <= elevation_bottom_layer1]
    elevation_bottom_layer1[topography <= elevation_bottom_layer1] = topography[topography <= elevation_bottom_layer1]
    elevation_bottom_layer1[topography <= elevation_bottom_layer1] = elevation_bottom_layer1[topography <= elevation_bottom_layer1] - 1
    elevation_bottom_layer2[elevation_bottom_layer1 <= elevation_bottom_layer2] = elevation_bottom_layer1[elevation_bottom_layer1 <= elevation_bottom_layer2]
    elevation_bottom_layer2[elevation_bottom_layer1 <= elevation_bottom_layer2 + 5] = elevation_bottom_layer2[elevation_bottom_layer1 <= elevation_bottom_layer2 + 5] - 5
    elevation_bottom_layer3[elevation_bottom_layer2 <= elevation_bottom_layer3] = elevation_bottom_layer2[elevation_bottom_layer2 <= elevation_bottom_layer3]
    elevation_bottom_layer3[elevation_bottom_layer2 <= elevation_bottom_layer3 + 5] = elevation_bottom_layer3[elevation_bottom_layer2 <= elevation_bottom_layer3 + 5] - 5
    elevation_bottom_layer4[elevation_bottom_layer3 <= elevation_bottom_layer4] = elevation_bottom_layer3[elevation_bottom_layer3 <= elevation_bottom_layer4]
    elevation_bottom_layer4[elevation_bottom_layer3 <= elevation_bottom_layer4 + 5] = elevation_bottom_layer4[elevation_bottom_layer3 <= elevation_bottom_layer4 + 5] - 5 
    elevation_bottom_layer4[elevation_bottom_layer4 <= 100.1] = 50

    var_obj[0, :, :] = topography
    var_obj[1, :, :] = elevation_bottom_layer1
    var_obj[2, :, :] = elevation_bottom_layer2
    var_obj[3, :, :] = elevation_bottom_layer3
    var_obj[4, :, :] = elevation_bottom_layer4