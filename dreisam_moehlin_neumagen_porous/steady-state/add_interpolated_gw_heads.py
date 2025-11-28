from pathlib import Path
import h5netcdf
import shutil
import rasterio
from pathlib import Path
import numpy as np


base_path = Path(__file__).parent

# load interpolated groundwater heads
src = rasterio.open(str(base_path / "input" / "groundwater_heads_interpolated_50m.tif"))
gw_heads_interpolated = src.read(1)

path = str(base_path / "input" / "parameters_modflow.nc")
with h5netcdf.File(path, "a", decode_vlen_strings=False) as f:
    topography = f.variables.get("elevations")[0, :, :]
    gw_depths_interpolated = topography - gw_heads_interpolated
    mask = np.isfinite(topography)
    gw_depths_interpolated[~mask] = np.nan
    gw_depths_interpolated[gw_depths_interpolated < 0] = 0
    gw_heads_interpolated[~mask] = np.nan

    try:
        v = f.create_variable("gw_heads_interpolated", ("y", "x"), float, compression="gzip", compression_opts=1)
        v[:, :] = gw_heads_interpolated
        v.attrs.update(long_name="gw_heads_interpolated", units="m a.s.l.")
    except ValueError:
        var_obj = f.variables.get("Interpolated groundwater heads")
        var_obj[:, :] = gw_heads_interpolated
    try:
        v = f.create_variable("gw_depths_interpolated", ("y", "x"), float, compression="gzip", compression_opts=1)
        v[:, :] = gw_depths_interpolated
        v.attrs.update(long_name="gw_depths_interpolated", units="m")
    except ValueError:
        var_obj = f.variables.get("Interpolated groundwater heads")
        var_obj[:, :] = gw_depths_interpolated