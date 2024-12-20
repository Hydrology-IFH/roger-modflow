import h5netcdf
import rasterio
from pathlib import Path

# add mask of Schoenberg to parameters_modflow.nc
base_path = Path(__file__).parent
src = rasterio.open(str(base_path.parent / "input" / "schoenberg.tif"))
schoenberg_mask = src.read(1)

# src = rasterio.open(str(base_path / "input" / "schoenberg_hohfirst.tif"))
# schoenberg_mask = src.read(1)

path = str(base_path / "parameters_modflow.nc")
with h5netcdf.File(path, "a", decode_vlen_strings=False) as f:
    try:
        v = f.create_variable("mask_schoenberg", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = schoenberg_mask 
        v.attrs.update(long_name="Mask of Schoenberg", units="")
    except ValueError:
        var_obj = f.variables.get("mask_schoenberg")
        var_obj[:, :] = schoenberg_mask