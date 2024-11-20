from pathlib import Path
import numpy as onp
import pandas as pd
import click


@click.option("-ns", "--nsamples", type=int, default=5000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

    bounds = {"r10_2": [0.01, 10], 
              "r110_2": [0.1, 10], 
              "r00101_2": [0.1, 100],
              "r10_3": [0.01, 10], 
              "r110_3": [0.1, 10], 
              "r00101_3": [0.1, 100],
              "r10_4": [0.01, 10], 
              "r110_4": [0.1, 10], 
              "r00101_4": [0.1, 100],
              "z10": [0.001, 2], 
              "z00101": [0.1, 100],  
              "m00101": [0.1, 100], 
              "m110": [0.1, 10],
              "offset_constant_head": [0, 50]
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

    df_params.iloc[0, 0] = 0.9
    df_params.iloc[0, 1] = 1.0
    df_params.iloc[0, 2] = 1.0
    df_params.iloc[0, 3] = 1.0
    df_params.iloc[0, 4] = 50.0
    df_params.iloc[0, 5] = 100.0
    df_params.iloc[0, 6] = 0.9
    df_params.iloc[0, 7] = 50.0
    df_params.iloc[0, 8] = 100.0
    df_params.iloc[0, 9] = 0.2
    df_params.iloc[0, 10] = 30.0
    df_params.iloc[0, 11] = 50.0
    df_params.iloc[0, 12] = 1.0
    df_params.iloc[0, 13] = 19.0
    df_params["complete"] = 0
    df_params = df_params.loc[:, ["r10_2", "r110_2", "r00101_2", "r10_3", "r110_3", "r00101_3", "r10_4", "r110_4", "r00101_4", "z10", "z00101", "m00101", "m110", "offset_constant_head", "complete"]]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[m]", ""],
        ["r10_2", "r110_2", "r00101_2", "r10_3", "r110_3", "r00101_3", "r10_4", "r110_4", "r00101_4", "z10", "z00101", "m00101", "m110", "offset_constant_head", "complete"],
    ]
    df_params.to_csv(base_path / "fudge_parameters_modflow.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
