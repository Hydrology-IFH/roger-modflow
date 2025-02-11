from pathlib import Path
import numpy as onp
import pandas as pd
import click


@click.option("-ns", "--nsamples", type=int, default=5000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

    bounds = {"l_1": [0.5, 20], 
              "rz10_23": [0.1, 5], 
              "r110_23": [0.5, 50], 
              "r0011_23": [0.5, 50],
              "r10_4": [0.1, 5], 
              "r110_4": [0.5, 50], 
              "r0011_4": [0.5, 50],
              "z0011": [0.5, 50],  
              "m0011": [0.5, 50], 
              "m110": [0.5, 10],
              "rch": [0.5, 1.1],
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

    df_params["complete"] = 0
    df_params = df_params.loc[:, ["l_1", "rz10_23", "r110_23", "r0011_23",  "r10_4", "r110_4", "r0011_4", "z0011", "m0011", "m110", "rch", "offset", "complete"]]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[m]",  ""],
        ["l_1", "rz10_23", "r110_23", "r0011_23", "r10_4", "r110_4", "r0011_4", "z0011", "m0011", "m110", "rch", "offset", "complete"],
    ]
    df_params.to_csv(base_path / "fudge_parameters_modflow.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
