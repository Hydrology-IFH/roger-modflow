import h5netcdf
import rasterio
from pathlib import Path

# add mask of subatchments to parameters_modflow.nc
base_path = Path(__file__).parent
src = rasterio.open(str(base_path / "input" / "mask_upper_dreisam.tif"))
mask_upper_dreisam = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_lower_dreisam.tif"))
mask_lower_dreisam = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_upper_moehlin.tif"))
mask_upper_moehlin = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_lower_moehlin.tif"))
mask_lower_moehlin = src.read(1)
src = rasterio.open(str(base_path / "input" / "mask_neumagen.tif"))
mask_neumagen = src.read(1)


path = str(base_path / "parameters_modflow.nc")
with h5netcdf.File(path, "a", decode_vlen_strings=False) as f:
    try:
        v = f.create_variable("mask_upper_dreisam", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_upper_dreisam
        v.attrs.update(long_name="Mask of upper Dreisam catchment", units="")
    except ValueError:
        var_obj = f.variables.get("mask_upper_dreisam")
        var_obj[:, :] = mask_upper_dreisam
    try:
        v = f.create_variable("mask_lower_dreisam", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_lower_dreisam
        v.attrs.update(long_name="Mask of lower Dreisam catchment", units="")
    except ValueError:
        var_obj = f.variables.get("mask_lower_dreisam")
        var_obj[:, :] = mask_lower_dreisam
    try:
        v = f.create_variable("mask_upper_moehlin", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_upper_moehlin
        v.attrs.update(long_name="Mask of upper Moehlin catchment", units="")
    except ValueError:
        var_obj = f.variables.get("mask_upper_moehlin")
        var_obj[:, :] = mask_upper_moehlin
    try:
        v = f.create_variable("mask_lower_moehlin", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_lower_moehlin
        v.attrs.update(long_name="Mask of lower Moehlin catchment", units="")
    except ValueError:
        var_obj = f.variables.get("mask_lower_moehlin")
        var_obj[:, :] = mask_lower_moehlin
    try:
        v = f.create_variable("mask_neumagen", ("y", "x"), int, compression="gzip", compression_opts=1)
        v[:, :] = mask_neumagen
        v.attrs.update(long_name="Mask of Neumagen catchment", units="")
    except ValueError:
        var_obj = f.variables.get("mask_neumagen")
        var_obj[:, :] = mask_neumagen