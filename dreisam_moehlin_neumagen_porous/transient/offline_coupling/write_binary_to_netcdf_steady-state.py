import sys
from pathlib import Path
import flopy
import xarray as xr
import geoxarray
import numpy as np
import pandas as pd
import datetime

import click

@click.command("main")
def main():
    try:
        print(sys.version)
        print(f"flopy version: {flopy.__version__}")

        base_path = Path(__file__).parent

        sim = flopy.mf6.MFSimulation.load(
            sim_ws=base_path / "output",
            exe_name="mf6",
            version="mf6",
            verbosity_level=0,
        )

        ml = sim.get_model("dmn_run_1806")
        nlayers = np.arange(ml.modelgrid.nlay)

        # load spatial reference and coordinates
        with xr.open_dataset(base_path.parent / "input" / "parameters_modflow.nc") as ds:
            topography = ds['elevations'].isel(z=0).values
            spatial_ref = ds.spatial_ref
            xcoords = ds.x.values
            ycoords = ds.y.values

        # export groundwater head to netcdf
        fhead = base_path / "output" / "dmn_run_1806.hds"
        hds = flopy.utils.HeadFile(fhead)
        ntimesteps = hds.get_alldata().shape[0]
        timesteps = np.arange(ntimesteps)

        fbudget = base_path / "output" / "dmn_run_1806.cbc"
        cbb = flopy.utils.CellBudgetFile(fbudget)

        cbb_headers = cbb.headers
        file = base_path / "output" / "cbb_headers.csv"
        cbb_headers_df = pd.DataFrame(cbb_headers)
        cbb_headers_df.to_csv(file, index=False, sep=";")

        # recharge = cbb.get_data(text="RCH", kstpkper=(2, 0), full3D=True)[0].filled(fill_value=np.nan)
        # np.nanmax(recharge)

        # wel = cbb.get_data(text="WEL", kstpkper=(3, 0), full3D=True)[0].filled(fill_value=np.nan)
        # np.nanmax(wel)

        # sfr = cbb.get_data(text="SFR", kstpkper=(1, 0), full3D=True)[0].filled(fill_value=np.nan)

        # recharge = ml.oc.output.budget().get_data(text="RCH", kstpkper=(0, 0))[0]


        # create xarray dataset
        attrs = dict(
                date_created=datetime.datetime.today().isoformat(),
                title="MODFLOW6 transient simulations of the Dreisam-Moehlin-Neumagen catchment (offline coupling with RoGeR)",
                institution="University of Freiburg, Chair of Hydrology",
                flopy_version=f"{flopy.__version__}",
                modflow_version=f"{ml.version}",
            )
        coords = {
                "lon": ("lon", xcoords),  # x
                "lat": ("lat", ycoords),  # y
                "layer": ("layer", nlayers),
                "Time": ("Time", timesteps),
            }
        data_vars=dict(
                head=(["Time", "layer", "lat", "lon"], np.where(hds.get_alldata() > 10000, np.nan, hds.get_alldata())),
                depth=(["Time", "layer", "lat", "lon"], np.where(hds.get_alldata() > 10000, np.nan, np.where(topography[np.newaxis, np.newaxis, :, :] - hds.get_alldata() > 0, topography[np.newaxis, np.newaxis, :, :] - hds.get_alldata(), 0))),
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        ds["head"].attrs["units"] = "m a.s.l."
        ds["head"].attrs["long_name"] = "Groundwater head"
        ds["depth"].attrs["units"] = "m"
        ds["depth"].attrs["long_name"] = "Groundwater depth"
        # create spatial reference
        ds = ds.geo.write_crs("EPSG:25832")
        ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
        file = base_path / "output" / "dmn_run_1806.nc"
        comp = dict(zlib=True, complevel=1)  # compress data to save storage
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)
    except:
        pass
    return


if __name__ == "__main__":
    main()