from pathlib import Path
import numpy as onp
import pandas as pd
import click


@click.option("-ns", "--nsamples", type=int, default=10000)
@click.command("main")
def main(nsamples):
    base_path = Path(__file__).parent

    bounds = {"-7_2_re": [1.1, 5.0],
              "-7_3_re": [1.1, 3.5],
              "-7_4_re": [1.1, 2.5],
              "-7_2_re1": [1.1, 5.0],
              "-7_3_re1": [1.1, 3.5],
              "-7_4_re1": [1.1, 2.5], 
              "hausen1_re": [1.1, 2.5], 
              "hausen2_re": [1.1, 2.5], 
    }

    nrows = nsamples
    ncols = 1

    # write parameters to csv
    df_params_ = pd.DataFrame(index=range(nrows * ncols))
    RNG = onp.random.default_rng(42)
    for i, param in enumerate(bounds.keys()):
        # generate random values
        values = (
            RNG.uniform(bounds[param][0], bounds[param][1], size=nrows).reshape((nrows, ncols)).astype(onp.float32)
        )
        # write parameters to dataframe
        df_params_.loc[:, param] = values.flatten()

    file = base_path.parent / "sfr-opt-kf-riv_drainage_schoenberg-no-flow_layers_gaussian-filter" / "fudge_parameters_metrics_porous.csv"
    fudge_parameters_metrics = pd.read_csv(file, sep=";", skiprows=0)
    df_params = fudge_parameters_metrics.loc[:, :"offset"].join(df_params_)
    df_params.loc[:, "-7_2_re"] = 20.0
    df_params.loc[:, "-7_3_re"] = 10.0
    df_params.loc[:, "-7_4_re"] = 5.0
    df_params.loc[:, "-7_2_re1"] = 0.1
    df_params.loc[:, "-7_3_re1"] = 0.1
    df_params.loc[:, "-7_4_re1"] = 0.1
    df_params.loc[:, "hausen1_re"] = 1.5
    df_params.loc[:, "hausen2_re"] = 15.

    df_params = df_params.loc[:, ["-8_1", "-7_1", "-6_1", "-5_1", 
                                  "-7_2", "-5_2","-4_2", "1-3_2", "1.8-3_2", "3-3_2", "4-3_2",  
                                  "-7_3", "-5_3","-4_3", "1-3_3", "1.8-3_3", "3-3_3", "4-3_3",
                                  "-7_4", "-5_4","-4_4", "1.8-3_4",
                                  "kf_riv", "rhkp", "rhkf", "man",
                                  "rch", "offset", 
                                  "-7_2_re", "-7_3_re", "-7_4_re", "hausen1_re", "hausen2_re",
                                  "-7_2_re1", "-7_3_re1", "-7_4_re1"]]

    # write parameters to csv
    df_params.columns = [
        ["[-]", "[-]", "[-]", "[-]", 
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]", "[-]",
         "[-]", "[m]",
         "[-]", "[-]", "[-]", "[-]", "[-]",
         "[-]", "[-]", "[-]"
         ],
        ["-8_1", "-7_1", "-6_1", "-5_1", 
         "-7_2", "-5_2", "-4_2", "1-3_2", "1.8-3_2", "3-3_2", "4-3_2",  
         "-7_3", "-5_3", "-4_3", "1-3_3", "1.8-3_3", "3-3_3", "4-3_3",
         "-7_4", "-5_4", "-4_4", "1.8-3_4",
         "kf_riv", "rhkp", "rhkf", "man",
         "rch", "offset",
         "-7_2_re", "-7_3_re", "-7_4_re", "hausen1_re", "hausen2_re",
         "-7_2_re1", "-7_3_re1", "-7_4_re1"],
    ]
    df_params.to_csv(base_path / "fudge_parameters_modflow.csv", index=False, sep=";")
    return


if __name__ == "__main__":
    main()
