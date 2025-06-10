from pathlib import Path
import numpy as onp
import pandas as pd
import click


@click.option("-ns", "--nsamples", type=int, default=5000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

    bounds = {"-8": [10, 1000], 
              "-7": [10, 1000], 
              "-6": [10, 1000], 
              "-5": [0.1, 100], 
              "-4": [0.01, 10],
              "1-3": [0.01, 2], 
              "1.8-3": [0.01, 2], 
              "3-3": [0.01, 2],  
              "4-3": [0.01, 2], 
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
    df_params.loc[27, "-8"] = 1000
    df_params.loc[27, "-7"] = 2500
    df_params.loc[27, "-6"] = 500
    df_params.loc[27, "-5"] = 10
    df_params.loc[27, "-4"] = 0.75
    df_params.loc[27, "1-3"] = 0.5
    df_params.loc[27, "1.8-3"] = 0.1
    df_params.loc[27, "3-3"] = 1.2
    df_params.loc[27, "4-3"] = 0.5
    df_params.loc[27, "rch"] = 1.0
    df_params.loc[27, "offset"] = 0.0

    df_params["complete"] = 0
    df_params = df_params.loc[:, ["-8", "-7", "-6", "-5", "-4", "1-3", "1.8-3", "3-3", "4-3", "rch", "offset", "complete"]]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[m]",  ""],
        ["-8", "-7", "-6", "-5", "-4", "1-3", "1.8-3", "3-3", "4-3", "rch", "offset", "complete"],
    ]
    df_params.to_csv(base_path / "fudge_parameters_modflow.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
