import sys
from pathlib import Path
import flopy
import xarray as xr
import geoxarray
import numpy as np
import datetime
import flopy.utils.binaryfile as bf

import click

@click.option("-mr", "--model-run", type=int, default=184)
@click.command("main")
def main(model_run):
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

        ml = sim.get_model(f"dmn_run_{model_run}")
        nlayers = np.arange(ml.modelgrid.nlay)


        # load spatial reference and coordinates
        with xr.open_dataset(base_path.parent / "input" / "parameters_modflow.nc") as ds:
            topography = ds['elevations'].isel(z=0).values
            spatial_ref = ds.spatial_ref
            xcoords = ds.x.values
            ycoords = ds.y.values[::-1]

        # export groundwater head to netcdf
        fhead = base_path / "output" / f"dmn_run_{model_run}.hds"
        hds = flopy.utils.HeadFile(fhead)

        fbudget = base_path / "output" / f"dmn_run_{model_run}.cbc"
        cbb = flopy.utils.CellBudgetFile(fbudget)

        flowja = ml.oc.output.budget().get_data(text="FLOW-JA-FACE", kstpkper=(0, 0))[0]
        grb_file = base_path / "output" / f"dmn_run_{model_run}.dis.grb"
        residual = flopy.mf6.utils.get_residuals(flowja, grb_file=grb_file)

        # create xarray dataset
        attrs = dict(
                date_created=datetime.datetime.today().isoformat(),
                title="MODFLOW steady-state simulations of the Dreisam-Moehlin-Neumagen catchment",
                institution="University of Freiburg, Chair of Hydrology",
                flopy_version=f"{flopy.__version__}",
                modflow_version=f"{ml.version}",
            )
        coords = {
                "lon": ("lon", xcoords),  # x
                "lat": ("lat", ycoords),  # y
                "layer": ("layer", nlayers),
                "Time": ("Time", [1]),
            }
        data_vars=dict(
                head=(["Time", "layer", "lat", "lon"], np.where(hds.get_data()[np.newaxis, :, :, :] > 10000, np.nan, hds.get_data()[np.newaxis, :, :, :])),
                depth=(["Time", "layer", "lat", "lon"], np.where(hds.get_data()[np.newaxis, :, :, :] > 10000, np.nan, np.where(topography[np.newaxis, np.newaxis, :, :] - hds.get_data()[np.newaxis, :, :, :] > 0, topography[np.newaxis, np.newaxis, :, :] - hds.get_data()[np.newaxis, :, :, :], 0))),
                flow_residual=(["Time", "layer", "lat", "lon"], residual[np.newaxis, :, :, :]),
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        ds["head"].attrs["units"] = "m a.s.l."
        ds["head"].attrs["long_name"] = "Groundwater head"
        ds["depth"].attrs["units"] = "m"
        ds["depth"].attrs["long_name"] = "Groundwater depth"
        ds["flow_residual"].attrs["units"] = "m/day"
        ds["flow_residual"].attrs["long_name"] = "Flow residuals"
        # create spatial reference
        ds = ds.geo.write_crs("EPSG:25832")
        ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
        file = base_path / "output" / f"modflow_output_run_{model_run}.nc"
        comp = dict(zlib=True, complevel=1)  # compress data to save storage
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(file, engine="h5netcdf", encoding=encoding)

        # # write to csv
        # if "steady-state" == "steady-state":
        #     hds_data = hds.get_data()
        #     for i in range(4):
        #         file = base_path / "output" / "steady-state" / f"groundwater_heads_layer{i+1}.csv"
        #         hds_data_layer = hds_data[i, ...]
        #         mask = (hds_data_layer > 1200) | (hds_data_layer < -100)
        #         hds_data_layer[mask] = np.nan
        #         np.savetxt(file, hds_data_layer, delimiter=";")

    except:
        pass
    return


if __name__ == "__main__":
    main()