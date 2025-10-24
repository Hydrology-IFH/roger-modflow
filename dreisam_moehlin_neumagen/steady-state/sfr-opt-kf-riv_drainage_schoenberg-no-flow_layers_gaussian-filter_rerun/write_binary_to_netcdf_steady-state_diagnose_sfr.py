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

@click.option("-mr", "--model-run", type=int, default=5)
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

        flowja = ml.oc.output.budget().get_data(text="FLOW-JA-FACE", kstpkper=(0, 0))[0]
        grb_file = base_path / "output" / f"dmn_run_{model_run}.dis.grb"
        residual = flopy.mf6.utils.get_residuals(flowja, grb_file=grb_file)

        # load the SFR output file
        output_file = base_path / "output" / f"dmn_run_{model_run}_sfr.obs.csv"
        df_sfr_ = pd.read_csv(output_file, sep=",")

        gw_head = np.where(hds.get_data()[np.newaxis, :, :, :] > 10000, np.nan, hds.get_data()[np.newaxis, :, :, :])
        gw_depth =  np.where(hds.get_data()[np.newaxis, :, :, :] > 10000, np.nan, np.where(topography[np.newaxis, np.newaxis, :, :] - hds.get_data()[np.newaxis, :, :, :] > 0, topography[np.newaxis, np.newaxis, :, :] - hds.get_data()[np.newaxis, :, :, :], 0))
        gw_sw = cbb.get_data(text="SFR", kstpkper=(0, 0), full3D=True)[0].filled(fill_value=np.nan)[np.newaxis, :, :, :]
        gw_sw = np.nanmean(gw_sw[0, ...], axis=0) / 86400

        rwid = np.nan * np.ones(topography.shape)
        man = np.nan * np.ones(topography.shape)
        rhk = np.nan * np.ones(topography.shape)
        rgrd = np.nan * np.ones(topography.shape)
        sfr_head = np.nan * np.ones(topography.shape)
        sfr_depth = np.nan * np.ones(topography.shape)
        sfr_flow = np.nan * np.ones(topography.shape)

        df_sfr = pd.DataFrame(columns=["rno", "layer", "x", "y", "rlen", "rwid", "rtp", "rgrd", "man", "rhk", "sfr_head", "sfr_depth", "flow", "kf", "topo", "gw_head", "gw-sw", "sw-gw_flux"])
        df_sfr["rno"] = reaches["rno"].values
        for rno in df_sfr["rno"].values:
            z = reaches.loc[reaches["rno"] == rno, "k"].values[0]
            y = reaches.loc[reaches["rno"] == rno, "i"].values[0]
            x = reaches.loc[reaches["rno"] == rno, "j"].values[0]
            kf = kf_layers[z, y, x]
            df_sfr.loc[df_sfr["rno"] == rno, "layer"] = reaches.loc[reaches["rno"] == rno, "k"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "y"] = reaches.loc[reaches["rno"] == rno, "i"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "x"] = reaches.loc[reaches["rno"] == rno, "j"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rlen"] = reaches.loc[reaches["rno"] == rno, "rlen"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rwid"] = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rtp"] = reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rgrd"] = reaches.loc[reaches["rno"] == rno, "rgrd"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "rhk"] = reaches.loc[reaches["rno"] == rno, "rhk"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "man"] = reaches.loc[reaches["rno"] == rno, "man"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "kf"] = kf
            rwidth = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
            stage_depth = df_sfr_.loc[0, dict_obs_stage_id_inv[rno]] - reaches.loc[reaches["rno"] == rno, "rtp"].values[0]
            df_sfr.loc[df_sfr["rno"] == rno, "sfr_head"] = df_sfr_.loc[0, dict_obs_stage_id_inv[rno]]
            df_sfr.loc[df_sfr["rno"] == rno, "sfr_depth"] = stage_depth
            flow = (df_sfr_.loc[0, dict_obs_flow_id_inv[rno]] * (-1)) / 86400
            df_sfr.loc[df_sfr["rno"] == rno, "sfr_flow"] = flow * stage_depth * rwidth
            df_sfr.loc[df_sfr["rno"] == rno, "gw_head"] = gw_head[0, z, y, x]
            df_sfr.loc[df_sfr["rno"] == rno, "gw-sw"] = gw_head[0, z, y, x] - df_sfr_.loc[0, dict_obs_stage_id_inv[rno]]
            df_sfr.loc[df_sfr["rno"] == rno, "sw-gw_flux"] = gw_sw[y, x] 
            df_sfr.loc[df_sfr["rno"] == rno, "topo"] = topography[y, x]
            rwid[y, x] = reaches.loc[reaches["rno"] == rno, "rwid"].values[0]
            rhk[y, x] = reaches.loc[reaches["rno"] == rno, "rhk"].values[0]
            rgrd[y, x] = reaches.loc[reaches["rno"] == rno, "rgrd"].values[0]
            sfr_head[y, x] = df_sfr.loc[df_sfr["rno"] == rno, "sfr_head"].values[0]
            sfr_depth[y, x] = df_sfr.loc[df_sfr["rno"] == rno, "sfr_depth"].values[0]
            sfr_flow[y, x] = df_sfr.loc[df_sfr["rno"] == rno, "sfr_flow"].values[0]
            man[y, x] = reaches.loc[reaches["rno"] == rno, "man"].values[0]

        file = base_path / "output" / f"dmn_run_{model_run}_sfr_.csv"
        df_sfr.to_csv(file, sep=";")

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
                specific_discharge=(["Time", "layer", "lat", "lon"], cbb.get_data(text="DATA-SPDIS", kstpkper=(0, 0), full3D=True)[0].filled(fill_value=np.nan)[np.newaxis, :, :, :]),
                gw_sw=(["Time", "layer", "lat", "lon"], cbb.get_data(text="SFR", kstpkper=(0, 0), full3D=True)[0].filled(fill_value=np.nan)[np.newaxis, :, :, :]),
                sfr_depth=(["lat", "lon"], sfr_depth),
                sfr_head=(["lat", "lon"], sfr_head),
                sfr_flow=(["lat", "lon"], sfr_flow),
                sfr_width=(["lat", "lon"], rwid),
                sfr_manning_coefficient=(["lat", "lon"], man),
                sfr_hydraulic_conductivity=(["lat", "lon"], rhk),
                sfr_gradient=(["lat", "lon"], rgrd),
            )

        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)
        ds["head"].attrs["units"] = "m a.s.l."
        ds["head"].attrs["long_name"] = "Groundwater head"
        ds["depth"].attrs["units"] = "m"
        ds["depth"].attrs["long_name"] = "Groundwater depth"
        ds["flow_residual"].attrs["units"] = "m/day"
        ds["flow_residual"].attrs["long_name"] = "Flow residuals"
        ds["specific_discharge"].attrs["units"] = "m3/day"
        ds["specific_discharge"].attrs["long_name"] = "Groundwater flux"
        ds["gw_sw"].attrs["units"] = "m3/day"
        ds["gw_sw"].attrs["long_name"] = "Groundwater-surface water flux"
        ds["sfr_depth"].attrs["units"] = "m"
        ds["sfr_depth"].attrs["long_name"] = "Streamflow depth"
        ds["sfr_head"].attrs["units"] = "m a.s.l."
        ds["sfr_head"].attrs["long_name"] = "Streamflow head"
        ds["sfr_flow"].attrs["units"] = "m3/s"
        ds["sfr_flow"].attrs["long_name"] = "Streamflow"
        ds["sfr_width"].attrs["units"] = "m"
        ds["sfr_width"].attrs["long_name"] = "Reach width"
        ds["sfr_manning_coefficient"].attrs["units"] = ""
        ds["sfr_manning_coefficient"].attrs["long_name"] = "Reach Manning's roughness coefficient"
        ds["sfr_hydraulic_conductivity"].attrs["units"] = "m/s"
        ds["sfr_hydraulic_conductivity"].attrs["long_name"] = "Reach hydraulic conductivity"
        ds["sfr_gradient"].attrs["units"] = ""
        ds["sfr_gradient"].attrs["long_name"] = "Reach gradient"
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

        # except:
        #     pass
    return


if __name__ == "__main__":
    main()