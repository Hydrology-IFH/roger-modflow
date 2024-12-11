import h5netcdf
import xarray as xr
from pathlib import Path


base_path = Path(__file__).parent
with xr.open_dataset(base_path / "input" / "mask_drainage.nc") as ds:
    mask_drainage = ds.mask_drainage.values


path = str(base_path / "parameters_modflow.nc")
with h5netcdf.File(path, "a", decode_vlen_strings=False) as f:
    try:
        v = f.create_variable("mask_drainage", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_drainage
        v.attrs.update(long_name="Mask of drainage areas", units="")
    except ValueError:
        var_obj = f.variables.get("mask_drainage")
        var_obj[:, :] = mask_drainage