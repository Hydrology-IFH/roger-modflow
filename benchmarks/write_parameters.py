from pathlib import Path
import h5netcdf
import datetime
import yaml
import numpy as onp
import pandas as pd
import click
import roger


_UNITS = {
    "lu_id": "-",
    "sealing": "-",
    "z_soil": "mm",
    "dmpv": "1/m2",
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


@click.option("-nx", "--nrows", type=int, default=12)
@click.option("-ny", "--ncols", type=int, default=24)
@click.command("main")
def main(nrows, ncols):
    base_path = Path(__file__).parent
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        config = yaml.safe_load(file)

    # write parameters to csv
    df_params = pd.DataFrame(index=range(nrows * ncols))
    df_params.loc[:, "lu_id"] = 8
    df_params.loc[:, "sealing"] = 0
    df_params.loc[:, "z_soil"] = 1000
    df_params.loc[:, "dmpv"] = 50
    df_params.loc[:, "lmpv"] = 300
    df_params.loc[:, "theta_ac"] = 0.1
    df_params.loc[:, "theta_ufc"] = 0.1
    df_params.loc[:, "theta_pwp"] = 0.2
    df_params.loc[:, "ks"] = 5
    df_params.loc[:, "kf"] = 100
    df_params.loc[:, "ta_offset"] = 0
    df_params.loc[:, "pet_weight"] = 1
    df_params.loc[:, "prec_weight"] = 1

    # write parameters to netcdf
    RNG = onp.random.default_rng(42)
    file_params = base_path / "parameters.nc"
    with h5netcdf.File(file_params, "w", decode_vlen_strings=False) as f:
        f.attrs.update(
            date_created=datetime.datetime.today().isoformat(),
            title="RoGeR model parameters",
            institution="University of Freiburg, Chair of Hydrology",
            references="",
            comment="",
            model_structure="SVAT model with free drainage",
            roger_version=f"{roger.__version__}",
        )
        dict_dim = {"x": nrows, "y": ncols}
        f.dimensions = dict_dim
        v = f.create_variable("x", ("x",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "x"
        v.attrs["units"] = "m"
        v[:] = onp.arange(dict_dim["x"]) * config["dx"]
        v = f.create_variable("y", ("y",), float, compression="gzip", compression_opts=1)
        v.attrs["long_name"] = "y"
        v.attrs["units"] = "m"
        v[:] = onp.arange(dict_dim["y"]) * config["dy"]
        for i, param in enumerate(df_params.columns):
            if param in ["z_soil", "lmpv", "dmpv"]:
                values = df_params.loc[:, param].values.reshape((dict_dim["x"], dict_dim["y"])).astype(onp.int32)
            else:
                values = df_params.loc[:, param].values.reshape((dict_dim["x"], dict_dim["y"])).astype(onp.float32)
            v = f.create_variable(param, ("x", "y"), float, compression="gzip", compression_opts=1)
            v[:, :] = values
            v.attrs.update(units=_UNITS[param])
            df_params.loc[:, param] = values.flatten()

    df_params = df_params.loc[
        :,
        [
            "lu_id",
            "sealing",
            "z_soil",
            "dmpv",
            "lmpv",
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

    # write parameters to csv
    df_params.columns = [
        ["", "[-]", "[mm]", "[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]", "[degC]", "[-]", "[-]"],
        ["lu_id", "sealing", "z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks", "kf", "ta_offset", "pet_weight", "prec_weight"],
    ]
    df_params.to_csv(base_path / "parameters.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
