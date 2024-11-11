from pathlib import Path
import numpy as onp
import pandas as pd
import click


@click.option("-ns", "--nsamples", type=int, default=5000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

    bounds = {"v10": [0.1, 10], 
              "v110": [0.1, 10], 
              "v00101": [0.1, 10], 
              "m00101": [0.1, 10], 
              "m110": [0.1, 10]
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

    df_params.iloc[0, :] = 1.0
    df_params.iloc[0, 3] = 2.0
    df_params.iloc[0, 4] = 2.0
    df_params["complete"] = 0
    df_params = df_params.loc[:, ["v10", "v110", "v00101", "m00101", "m110", "complete"]]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", "[-]", ""],
        ["v10", "v110", "v00101", "m00101", "m110", "complete"],
    ]
    df_params.to_csv(base_path / "fudge_parameters_modflow.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
