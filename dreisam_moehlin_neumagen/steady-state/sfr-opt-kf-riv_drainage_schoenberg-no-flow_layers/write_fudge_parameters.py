from pathlib import Path
import numpy as onp
import pandas as pd
import click


@click.option("-ns", "--nsamples", type=int, default=10000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

    bounds = {"-8_1": [400, 600], 
              "-7_1": [70, 80], 
              "-6_1": [0.9, 10], 
              "-5_1": [0.9, 10],
              "-7_2": [50, 60], 
              "-5_2": [0.9, 10], 
              "-4_2": [0.5, 2.5],
              "1-3_2": [0.5, 2.5], 
              "1.8-3_2": [0.5, 2.5], 
              "3-3_2": [0.5, 2.5],  
              "4-3_2": [0.5, 2.5],
              "-7_3": [50, 60], 
              "-5_3": [0.5, 2.5], 
              "-4_3": [0.5, 2.5],
              "1-3_3": [0.5, 2.5], 
              "1.8-3_3": [0.5, 2.5], 
              "3-3_3": [0.5, 2.5],  
              "4-3_3": [0.5, 2.5],
              "-7_4": [5, 20], 
              "-5_4": [0.5, 2.5], 
              "-4_4": [0.5, 2.5],
              "1.8-3_4": [0.5, 2.5],
              "kf_riv": [1.5, 5],
              "rhk": [0.75, 1.25],
              "rch": [0.95, 1.05],
    }

    nrows = nsamples
    ncols = 1

    # write parameters to csv
    df_params = pd.DataFrame(index=range(nrows * ncols))
    RNG = onp.random.default_rng(42)
    for i, param in enumerate(bounds.keys()):
        # generate random values
        values = (
            RNG.uniform(bounds[param][0], bounds[param][1], size=nrows).reshape((nrows, ncols)).astype(onp.float32)
        )
        # write parameters to dataframe
        df_params.loc[:, param] = values.flatten()

    df_params["offset"] = 0.
    df_params.iloc[:34, :] = 1.
    df_params.loc[:33, "kf_riv"] = 5
    df_params.loc[:33, "rhk"] = 1

    df_params.loc[:33, "-8_1"] = 500.
    df_params.loc[:33, "-7_1"] = 75.
    df_params.loc[:33, "-7_2"] = 60.
    df_params.loc[:33, "-7_3"] = 50.
    df_params.loc[:33, "-7_4"] = 10.
    df_params.loc[0, "rch"] = 1.25
    df_params.loc[1, "rch"] = 1.2
    df_params.loc[2, "rch"] = 1.15
    df_params.loc[3, "rch"] = 1.1
    df_params.loc[4, "rch"] = 1.05
    df_params.loc[5, "rch"] = 1.0
    df_params.loc[6, "rch"] = 0.95
    df_params.loc[7, "rch"] = 0.9
    df_params.loc[8, "rch"] = 0.85
    df_params.loc[9, "rch"] = 0.8
    df_params.loc[10, "rch"] = 0.75
    df_params.loc[11, "offset"] = 0.1
    df_params.loc[12, "offset"] = 0.2
    df_params.loc[13, "offset"] = 0.3
    df_params.loc[14, "offset"] = 0.4
    df_params.loc[15, "offset"] = 0.5
    df_params.loc[16, "offset"] = 0.6
    df_params.loc[17, "offset"] = 0.7
    df_params.loc[18, "offset"] = 0.8
    df_params.loc[19, "offset"] = 0.9
    df_params.loc[20, "offset"] = 1.0
    df_params.loc[21, "offset"] = 1.5
    df_params.loc[22, "offset"] = 2.0
    df_params.loc[23, "offset"] = 2.5
    df_params.loc[24, "offset"] = 5.0
    df_params.loc[25, "offset"] = 10.0
    df_params.loc[26, "kf_riv"] = 2
    df_params.loc[27, "kf_riv"] = 3
    df_params.loc[28, "kf_riv"] = 4
    df_params.loc[29, "kf_riv"] = 5
    df_params.loc[30, "rhk"] = 1.2
    df_params.loc[31, "rhk"] = 1.1
    df_params.loc[32, "rhk"] = 0.9
    df_params.loc[33, "rhk"] = 0.8

    # constrain parameters by upper layers (i.e. kf of lower layer is equal or less than kf of upper layer)
    cond = (df_params["-7_3"] >= df_params["-7_2"])
    df_params.loc[cond, "-7_3"] = df_params.loc[cond, "-7_2"]

    cond = (df_params["-7_4"] >= df_params["-7_3"])
    df_params.loc[cond, "-7_4"] = df_params.loc[cond, "-7_3"] * 0.5

    cond = (df_params["4-3_3"] >= df_params["4-3_2"])
    df_params.loc[cond, "4-3_3"] = df_params.loc[cond, "4-3_2"]

    cond = (df_params["3-3_3"] >= df_params["3-3_2"])
    df_params.loc[cond, "3-3_3"] = df_params.loc[cond, "3-3_2"]

    cond = (df_params["1.8-3_3"] >= df_params["1.8-3_2"])
    df_params.loc[cond, "1.8-3_3"] = df_params.loc[cond, "1.8-3_2"]
    cond = (df_params["1.8-3_4"] >= df_params["1.8-3_3"])
    df_params.loc[cond, "1.8-3_4"] = df_params.loc[cond, "1.8-3_3"]

    cond = (df_params["1-3_3"] >= df_params["1-3_2"])
    df_params.loc[cond, "1-3_3"] = df_params.loc[cond, "1-3_2"]

    cond = (df_params["4-3_3"] >= df_params["4-3_2"])
    df_params.loc[cond, "4-3_3"] = df_params.loc[cond, "4-3_2"]

    cond = (df_params["-5_3"] >= df_params["-5_2"])
    df_params.loc[cond, "-5_3"] = df_params.loc[cond, "-5_2"]
    cond = (df_params["-5_4"] >= df_params["-5_3"])
    df_params.loc[cond, "-5_4"] = df_params.loc[cond, "-5_3"]

    df_params["complete"] = 0
    df_params = df_params.loc[:, ["-8_1", "-7_1", "-6_1", "-5_1", 
                                  "-7_2", "-5_2","-4_2", "1-3_2", "1.8-3_2", "3-3_2", "4-3_2",  
                                  "-7_3", "-5_3","-4_3", "1-3_3", "1.8-3_3", "3-3_3", "4-3_3",
                                  "-7_4", "-5_4","-4_4", "1.8-3_4",
                                  "kf_riv", "rhk",
                                  "rch", "offset", "complete"]]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", 
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]",
         "[-]", "[m]",  ""],
        ["-8_1", "-7_1", "-6_1", "-5_1", 
         "-7_2", "-5_2", "-4_2", "1-3_2", "1.8-3_2", "3-3_2", "4-3_2",  
         "-7_3", "-5_3", "-4_3", "1-3_3", "1.8-3_3", "3-3_3", "4-3_3",
         "-7_4", "-5_4", "-4_4", "1.8-3_4",
         "kf_riv", "rhk",
         "rch", "offset", "complete"],
    ]
    df_params.to_csv(base_path / "fudge_parameters_modflow.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
