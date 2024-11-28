from pathlib import Path
import numpy as onp
import pandas as pd
import click


@click.option("-ns", "--nsamples", type=int, default=5000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

    bounds = {"r10_2": [0.1, 5], 
              "r110_2": [0.1, 5], 
              "r0011_2": [1, 50],
              "r10_3": [0.1, 5], 
              "r110_3": [1, 10], 
              "r0011_3": [1, 50],
              "r10_4": [0.1, 5], 
              "r110_4": [1, 10], 
              "r0011_4": [1, 50],
              "z10": [0.1, 5], 
              "z0011": [20, 40],  
              "m0011": [30, 70], 
              "m110": [20, 40],
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

    df_params.iloc[0, :] = 1.
    df_params.loc[0, "r0011_2"] = 30.
    df_params.loc[0, "r0011_3"] = 30.
    df_params.loc[0, "r0011_4"] = 20.
    df_params.loc[0, "r110_2"] = 5.
    df_params.loc[0, "r110_3"] = 5.
    df_params.loc[0, "r110_4"] = 5.
    df_params.loc[0, "r10_2"] = 2.5
    df_params.loc[0, "r10_3"] = 2.5
    df_params.loc[0, "r10_4"] = 4
    df_params.loc[0, "z10"] = 0.78
    df_params.loc[0, "z0011"] = 30.
    df_params.loc[0, "m0011"] = 50.
    df_params["complete"] = 0
    df_params = df_params.loc[:, ["r10_2", "r110_2", "r0011_2", "r10_3", "r110_3", "r0011_3", "r10_4", "r110_4", "r0011_4", "z10", "z0011", "m0011", "m110", "complete"]]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", ""],
        ["r10_2", "r110_2", "r0011_2", "r10_3", "r110_3", "r0011_3", "r10_4", "r110_4", "r0011_4", "z10", "z0011", "m0011", "m110", "complete"],
    ]
    df_params.to_csv(base_path / "fudge_parameters_modflow.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
