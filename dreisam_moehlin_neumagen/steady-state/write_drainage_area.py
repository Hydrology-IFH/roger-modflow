import sys
from pathlib import Path
import flopy
import xarray as xr
import geoxarray
import datetime

import click

@click.command("main")
def main():

    base_path = Path(__file__).parent

    # load spatial reference and coordinates
    with xr.open_dataset(base_path / "parameters_modflow.nc") as ds:
        spatial_ref = ds.spatial_ref
        xcoords = ds.x.values
        ycoords = ds.y.values
        mask_drainage = ds.mask_drainage.values

    # create xarray dataset
    attrs = dict(
            date_created=datetime.datetime.today().isoformat(),
            title="Drainage area of the Dreisam-Moehlin-Neumagen catchment",
            institution="University of Freiburg, Chair of Hydrology",
        )
    coords = {
            "lon": ("lon", xcoords),  # x
            "lat": ("lat", ycoords),  # y
        }
    data_vars=dict(
            mask_drainage=(["lat", "lon"], mask_drainage),
        )

    ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
    ds["mask_drainage"].attrs["units"] = ""
    ds["mask_drainage"].attrs["long_name"] = "Mask of drainage areas"
    # create spatial reference
    ds = ds.geo.write_crs("EPSG:25832")
    ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
    file = base_path / "input" / "mask_drainage.nc"
    comp = dict(zlib=True, complevel=1)  # compress data to save storage
    encoding = {var: comp for var in ds.data_vars}
    ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
    return


if __name__ == "__main__":
    main()