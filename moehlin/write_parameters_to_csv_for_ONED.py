from pathlib import Path
import xarray as xr
import numpy as onp
import pandas as pd
import click


_UNITS = {
    "z_soil": "mm",
    "slope": "-",
    "dmpv": "1/m2",
    "dmph": "1/m2",
    "lmpv": "mm",
    "theta_ac": "-",
    "theta_ufc": "-",
    "theta_pwp": "-",
    "ks": "mm/hour",
    "kf": "mm/hour",
    "ta_offset": "degC",
    "pet_weight": "-",
    "prec_weight": "-",
}


@click.command("main")
def main():
    base_path = Path(__file__).parent

    # load the netcdf file
    params_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_file, engine="h5netcdf")
    nrows = ds_params.sizes["y"]
    ncols = ds_params.sizes["x"]

    # write parameters to csv
    df_params = pd.DataFrame(index=range(nrows * ncols))
    df_params.loc[:, "lu_id"] = ds_params.lanu.values.T.flatten()
    df_params.loc[:, "slope"] = ds_params.slope.values.T.flatten() / 100
    df_params.loc[:, "sealing"] = ds_params.vers.values.T.flatten() / 100
    df_params.loc[:, "z_soil"] = ds_params.GRUND.values.T.flatten() * 10
    df_params.loc[:, "dmpv"] = ds_params.MPD_V.values.T.flatten()
    df_params.loc[:, "lmpv"] = ds_params.MPL_V.values.T.flatten()
    df_params.loc[:, "dmph"] = ds_params.MPD_H.values.T.flatten()
    df_params.loc[:, "theta_ac"] = ds_params.LK.values.T.flatten() / 100
    df_params.loc[:, "theta_ufc"] = ds_params.NFK.values.T.flatten() / 100
    df_params.loc[:, "theta_pwp"] = ds_params.PWP.values.T.flatten() / 100
    df_params.loc[:, "ks"] = ds_params.KS.values.T.flatten()
    df_params.loc[:, "kf"] = ds_params.TP.values.T.flatten()
    df_params.loc[:, "ta_offset"] = ds_params.F_t.T.values.flatten()
    df_params.loc[:, "pet_weight"] = ds_params.F_et.T.values.flatten() / 100
    df_params.loc[:, "prec_weight"] = ds_params.F_n_h_y.T.values.flatten() / 100

    df_params = df_params.loc[
        :,
        [
            "lu_id",
            "slope",
            "sealing",
            "z_soil",
            "dmpv",
            "lmpv",
            "dmph",
            "theta_ac",
            "theta_ufc",
            "theta_pwp",
            "ks",
            "kf",
            "ta_offset",
            "pet_weight",
            "prec_weight"
        ],
    ]
    df_params.fillna(-9999, inplace=True)
    df_params["lu_id"] = df_params["lu_id"].astype(onp.int16)

    # write parameters to csv
    df_params.columns = [
        ["", "[-]", "[-]", "[mm]", "[1/m2]", "[mm]", "[1/m2]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]", "[degC]", "[-]", "[-]"],
        ["lu_id", "slope", "sealing", "z_soil", "dmpv", "lmpv", "dmph", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "ta_offset", "pet_weight", "prec_weight"],
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
