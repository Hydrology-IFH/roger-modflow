import sys
from pathlib import Path
import flopy
import xarray as xr
import geoxarray
import numpy as np
import datetime
import flopy.utils.binaryfile as bf
import pandas as pd
import yaml

import click

@click.option("-mr", "--model-run", type=int, default=9491)
@click.option("-c", "--converged", type=int, default=1)
@click.command("main")
def main(model_run, converged):
    if converged == 1:
        # try:
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
            kf_layers = ds['kf'].values
            spatial_ref = ds.spatial_ref
            xcoords = ds.x.values
            ycoords = ds.y.values[::-1]

        # load the modflow config file
        file_config = base_path.parent / "config.yml"
        with open(file_config, "r") as file:
            modflow_config = yaml.safe_load(file)

        dict_obs_stage_id = modflow_config["dict_obs_stage_rnos"]
        dict_obs_flow_id = modflow_config["dict_obs_flow_rnos"]

        dict_obs_stage_id_inv = {v: k for k, v in dict_obs_stage_id.items()}
        dict_obs_flow_id_inv = {v: k for k, v in dict_obs_flow_id.items()}

        # load the fugde parameters
        path = base_path / "fudge_parameters_modflow.csv"
        fudge_parameters = pd.read_csv(path, sep=";", skiprows=1)

        # load the SFR reaches
        reaches = pd.read_csv(base_path.parent / 'input' / 'sfr_packagedata_modified.csv', sep=';')
        reaches["rno"] = reaches["rno"] - 1
        reaches["k"] = reaches["k"] - 1
        reaches["i"] = reaches["i"] - 1
        reaches["j"] = reaches["j"] - 1

        # fudge streambed conductivity
        cond = (reaches["kf"] >= 10e-6)
        reaches.loc[cond, "rhk"] = reaches.loc[cond, "rhk"] * fudge_parameters["rhkp"].values[model_run]
        cond = (reaches["kf"] < 10e-6)
        reaches.loc[cond, "rhk"] = reaches.loc[cond, "rhk"] * fudge_parameters["rhkf"].values[model_run]
        reaches["man"] = reaches["man"] * fudge_parameters["man"].values[model_run]

        # write groundwater head to netcdf
        fhead = base_path / "output" / f"dmn_run_{model_run}.hds"
        hds = flopy.utils.HeadFile(fhead)

        fbudget = base_path / "output" / f"dmn_run_{model_run}.cbc"
        cbb = flopy.utils.CellBudgetFile(fbudget)

        gw_sw = cbb.get_data(text="SFR", kstpkper=(0, 0), full3D=True)[0].filled(fill_value=np.nan)[np.newaxis, :, :, :]
        gw_sw = np.nansum(gw_sw[0, ...], axis=0)


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
                gw_sw_=(["Time", "lat", "lon"], np.nansum(cbb.get_data(text="SFR", kstpkper=(0, 0), full3D=True)[0].filled(fill_value=np.nan), axis=0)[np.newaxis, :, :]),
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        ds["head"].attrs["units"] = "m a.s.l."
        ds["head"].attrs["long_name"] = "Groundwater head"
        ds["gw_sw"].attrs["units"] = "m3/day"
        ds["gw_sw"].attrs["long_name"] = "Groundwater-surface water flux"
        # create spatial reference
        ds = ds.geo.write_crs("EPSG:25832")
        ds.coords["spatial_ref"] = spatial_ref  # update spatial reference from parameters_modflow.nc
        file = base_path / "output" / f"modflow_output_run_{model_run}_pre1.nc"
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

        # except:
        #     pass
    return


if __name__ == "__main__":
    main()