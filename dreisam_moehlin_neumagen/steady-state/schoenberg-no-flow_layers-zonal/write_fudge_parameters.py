from pathlib import Path
import numpy as onp
import pandas as pd
import click


@click.option("-ns", "--nsamples", type=int, default=5000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

# Layer 1
# [1.1574075e-08 2.7777778e-08 1.9444444e-07 2.3055554e-07 5.7777777e-07
#  1.1583334e-06 1.7361111e-06 1.8055556e-06 1.9916667e-06 2.8944444e-06
#  4.6305554e-06 6.9444445e-06 8.1027783e-06 1.1575000e-05 1.8194445e-05]
# Layer 2
# [2.7777778e-11 1.9722222e-07 1.9999999e-07 1.8180555e-05 1.8181944e-04
#  1.0000000e-03 1.8181807e-03 3.0000000e-03 4.0000002e-03]


    bounds = {"-8_1_1": [10, 1000],  # zone 1
              "-7_1_1": [10, 1000], 
              "-6_1_1": [10, 1000], 
              "-5_1_1": [0.1, 100],
              "-7_2_1": [10, 1000], 
              "-5_2_1": [0.1, 100], 
              "-4_2_1": [0.01, 10],
              "1-3_2_1": [0.01, 2], 
              "1.8-3_2_1": [0.01, 2], 
              "3-3_2_1": [0.01, 2],  
              "4-3_2_1": [0.01, 2],
              "-7_3_1": [10, 1000], 
              "-5_3_1": [0.1, 100], 
              "-4_3_1": [0.01, 10],
              "1-3_3_1": [0.01, 2], 
              "1.8-3_3_1": [0.01, 2], 
              "3-3_3_1": [0.01, 2],  
              "4-3_3_1": [0.01, 2],
              "-7_4_1": [10, 100], 
              "-5_4_1": [0.1, 100], 
              "-4_4_1": [0.01, 10],
              "1.8-3_4_1": [0.01, 2],
              "-8_1_2": [10, 1000],  # zone 2
              "-7_1_2": [10, 1000], 
              "-6_1_2": [10, 1000], 
              "-5_1_2": [0.1, 100],
              "-7_2_2": [10, 1000], 
              "-5_2_2": [0.1, 100], 
              "-4_2_2": [0.01, 10],
              "1-3_2_2": [0.01, 2], 
              "1.8-3_2_2": [0.01, 2], 
              "3-3_2_2": [0.01, 2],  
              "4-3_2_2": [0.01, 2],
              "-7_3_2": [10, 1000], 
              "-5_3_2": [0.1, 100], 
              "-4_3_2": [0.01, 10],
              "1-3_3_2": [0.01, 2], 
              "1.8-3_3_2": [0.01, 2], 
              "3-3_3_2": [0.01, 2],  
              "4-3_3_2": [0.01, 2],
              "-7_4_2": [10, 100], 
              "-5_4_2": [0.1, 100], 
              "-4_4_2": [0.01, 10],
              "1.8-3_4_2": [0.01, 2],  
              "rch": [0.75, 1.25],
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

    df_params.iloc[:27, :] = 1.
    df_params["offset"] = 0.
    df_params.loc[0, "rch"] = 1.5
    df_params.loc[1, "rch"] = 1.4
    df_params.loc[2, "rch"] = 1.3
    df_params.loc[3, "rch"] = 1.2
    df_params.loc[4, "rch"] = 1.1
    df_params.loc[5, "rch"] = 1.0
    df_params.loc[6, "rch"] = 0.9
    df_params.loc[7, "rch"] = 0.8
    df_params.loc[8, "rch"] = 0.7
    df_params.loc[9, "rch"] = 0.6
    df_params.loc[10, "rch"] = 0.5
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
    df_params.loc[26, "offset"] = 20.0

    # constrain parameters by upper layers (i.e. kf of lower layer is equal or less than kf of upper layer)
    # zone 1
    cond = (df_params["-7_3_1"] >= df_params["-7_2_1"])
    df_params.loc[cond, "-7_3_1"] = df_params.loc[cond, "-7_2_1"] * 0.666

    cond = (df_params["-7_4_1"] >= df_params["-7_3_1"])
    df_params.loc[cond, "-7_4_1"] = df_params.loc[cond, "-7_3_1"] * 0.333

    cond = (df_params["4-3_3_1"] >= df_params["4-3_2_1"])
    df_params.loc[cond, "4-3_3_1"] = df_params.loc[cond, "4-3_2_1"]

    cond = (df_params["3-3_3_1"] >= df_params["3-3_2_1"])
    df_params.loc[cond, "3-3_3_1"] = df_params.loc[cond, "3-3_2_1"]

    cond = (df_params["1.8-3_3_1"] >= df_params["1.8-3_2_1"])
    df_params.loc[cond, "1.8-3_3_1"] = df_params.loc[cond, "1.8-3_2_1"]
    cond = (df_params["1.8-3_4_1"] >= df_params["1.8-3_3_1"])
    df_params.loc[cond, "1.8-3_4_1"] = df_params.loc[cond, "1.8-3_3_1"]

    cond = (df_params["1-3_3_1"] >= df_params["1-3_2_1"])
    df_params.loc[cond, "1-3_3_1"] = df_params.loc[cond, "1-3_2_1"]

    cond = (df_params["4-3_3_1"] >= df_params["4-3_2_1"])
    df_params.loc[cond, "4-3_3_1"] = df_params.loc[cond, "4-3_2_1"]

    cond = (df_params["-5_3_1"] >= df_params["-5_2_1"])
    df_params.loc[cond, "-5_3_1"] = df_params.loc[cond, "-5_2_1"]
    cond = (df_params["-5_4_1"] >= df_params["-5_3_1"])
    df_params.loc[cond, "-5_4_1"] = df_params.loc[cond, "-5_3_1"]

    # zone 2
    cond = (df_params["-7_3_2"] >= df_params["-7_2_2"])
    df_params.loc[cond, "-7_3_2"] = df_params.loc[cond, "-7_2_2"]

    cond = (df_params["-7_4_2"] >= df_params["-7_3_2"])
    df_params.loc[cond, "-7_4_2"] = df_params.loc[cond, "-7_3_2"] * 0.333

    cond = (df_params["4-3_3_2"] >= df_params["4-3_2_2"])
    df_params.loc[cond, "4-3_3_2"] = df_params.loc[cond, "4-3_2_2"]

    cond = (df_params["3-3_3_2"] >= df_params["3-3_2_2"])
    df_params.loc[cond, "3-3_3_2"] = df_params.loc[cond, "3-3_2_2"]

    cond = (df_params["1.8-3_3_2"] >= df_params["1.8-3_2_2"])
    df_params.loc[cond, "1.8-3_3_2"] = df_params.loc[cond, "1.8-3_2_2"]
    cond = (df_params["1.8-3_4_2"] >= df_params["1.8-3_3_2"])
    df_params.loc[cond, "1.8-3_4_2"] = df_params.loc[cond, "1.8-3_3_2"]

    cond = (df_params["1-3_3_2"] >= df_params["1-3_2_2"])
    df_params.loc[cond, "1-3_3_2"] = df_params.loc[cond, "1-3_2_2"]

    cond = (df_params["4-3_3_2"] >= df_params["4-3_2_2"])
    df_params.loc[cond, "4-3_3_2"] = df_params.loc[cond, "4-3_2_2"]

    cond = (df_params["-5_3_2"] >= df_params["-5_2_2"])
    df_params.loc[cond, "-5_3_2"] = df_params.loc[cond, "-5_2_2"]
    cond = (df_params["-5_4_2"] >= df_params["-5_3_2"])
    df_params.loc[cond, "-5_4_2"] = df_params.loc[cond, "-5_3_2"]

    df_params["complete"] = 0
    df_params = df_params.loc[:, ["-8_1_1", "-7_1_1", "-6_1_1", "-5_1_1", 
                                  "-7_2_1", "-5_2_1","-4_2_1", "1-3_2_1", "1.8-3_2_1", "3-3_2_1", "4-3_2_1",  
                                  "-7_3_1", "-5_3_1","-4_3_1", "1-3_3_1", "1.8-3_3_1", "3-3_3_1", "4-3_3_1",
                                  "-7_4_1", "-5_4_1","-4_4_1", "1.8-3_4_1",
                                  "-8_1_2", "-7_1_2", "-6_1_2", "-5_1_2", 
                                  "-7_2_2", "-5_2_2","-4_2_2", "1-3_2_2", "1.8-3_2_2", "3-3_2_2", "4-3_2_2",  
                                  "-7_3_2", "-5_3_2","-4_3_2", "1-3_3_2", "1.8-3_3_2", "3-3_3_2", "4-3_3_2",
                                  "-7_4_2", "-5_4_2","-4_4_2", "1.8-3_4_2",   
                                  "rch", "offset", "complete"]]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", 
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]", 
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]",
         "[-]", "[m]",  ""],
        ["-8_1_1", "-7_1_1", "-6_1_1", "-5_1_1", 
         "-7_2_1", "-5_2_1","-4_2_1", "1-3_2_1", "1.8-3_2_1", "3-3_2_1", "4-3_2_1",  
         "-7_3_1", "-5_3_1","-4_3_1", "1-3_3_1", "1.8-3_3_1", "3-3_3_1", "4-3_3_1",
         "-7_4_1", "-5_4_1","-4_4_1", "1.8-3_4_1",
         "-8_1_2", "-7_1_2", "-6_1_2", "-5_1_2", 
         "-7_2_2", "-5_2_2","-4_2_2", "1-3_2_2", "1.8-3_2_2", "3-3_2_2", "4-3_2_2",  
         "-7_3_2", "-5_3_2","-4_3_2", "1-3_3_2", "1.8-3_3_2", "3-3_3_2", "4-3_3_2",
         "-7_4_2", "-5_4_2","-4_4_2", "1.8-3_4_2",    
         "rch", "offset", "complete"],
    ]
    df_params.to_csv(base_path / "fudge_parameters_modflow.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
