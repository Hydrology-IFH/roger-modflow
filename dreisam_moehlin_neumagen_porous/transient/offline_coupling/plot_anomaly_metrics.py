from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import click

def aggregate_to_coarser_resolution(vals, res_fine, res_coarse, method="sum", x_origin=0, y_origin=0):
    """Aggregate raster data to a coarser resolution.
    
    Args
    ----
    vals : numpy.ndarray
        2D array with the raster data.

    res_fine : int
        spatial resolution of the fine grid in meters.

    res_coarse : int
        spatial resolution of the coarse grid in meters.

    method : str
        Method to aggregate the data. Options are "sum" and "average".

    x_origin : int
        x-coordinate of the grid origin (lower left corner).

    y_origin : int
        y-coordinate of the grid origin (lower left corner).  
    """
    ny_fine, nx_fine = vals.shape[0], vals.shape[1]
    nlat_coarse, nlon_coarse = int(res_coarse / res_fine), int(res_coarse / res_fine)
    meters_to_latlon = 111195
    lat_fine = np.linspace(y_origin, y_origin + ny_fine*(res_fine/meters_to_latlon), ny_fine)/meters_to_latlon  # boundaries
    lon_fine = np.linspace(x_origin, x_origin + nx_fine*(res_fine/meters_to_latlon), nx_fine)/meters_to_latlon  # boundaries

    arr_fine = xr.DataArray(vals, coords={"lat": lat_fine, "lon": lon_fine}, dims=["lat", "lon"])

    if method == "sum":
        arr_coarse = xr.DataArray(
            arr_fine.coarsen(lat=nlat_coarse, lon=nlon_coarse).sum().values,
            dims=("lat", "lon"),
        )
        
    elif method == "average":
        arr_coarse = xr.DataArray(
            arr_fine.coarsen(lat=nlat_coarse, lon=nlon_coarse).mean().values,
            dims=("lat", "lon"),
        )
    return arr_coarse.values


@click.option("-mr", "--model-run", type=int, default=1806)
@click.command("main", short_help="Evaluate the transient simulation")
def main(model_run):
    base_path = Path(__file__).parent
    # base_path_output = base_path / "output"
    base_path_output = Path("/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline-coupling/output")

    areas = ["dmn"]

    base = "base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction"

    stress_test_scenarios = ["summer-drought-magnitude2-duration3_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress"]

    
    date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
    years = np.unique(date_time.year.values)

    # load metrics
    output_file = base_path_output / f"metrics_run{model_run}.csv"
    df_metrics = pd.read_csv(output_file, sep=";")
    output_file = base_path_output / f"anomaly_metrics_abs_run{model_run}.csv"
    df_anomaly_metrics_abs = pd.read_csv(output_file, sep=";")
    output_file = base_path_output / f"anomaly_metrics_rel_run{model_run}.csv"
    df_anomaly_metrics_rel = pd.read_csv(output_file, sep=";")

    _labels = {"air_temperature": "TA", 
               "direct_recharge": "Dir. GWR", 
               "indirect_recharge": "Indir. GWR", 
               "gw_depth": "GW Depth"}
    
    _units_abs = {"air_temperature": " [°C]", 
                  "direct_recharge": " [$m^3$/d]", 
                  "indirect_recharge": " [$m^3$/d]", 
                  "gw_depth": " [m]"}

    for area in areas:
        # # select metrics for the base scenario and for the current area 
        # df_metrics_base_area = df_metrics[(df_metrics["scenario"] == base) & (df_metrics["area"] == area)]


        for stress_test_scenario in stress_test_scenarios:
            # select metrics for the current scenario and area
            df_anomaly_metrics_abs_scenario_area = df_anomaly_metrics_abs[(df_anomaly_metrics_abs["scenario"] == stress_test_scenario) & (df_anomaly_metrics_abs["area"] == area)]
            df_anomaly_metrics_rel_scenario_area = df_anomaly_metrics_rel[(df_anomaly_metrics_rel["scenario"] == stress_test_scenario) & (df_anomaly_metrics_rel["area"] == area)]
            variables = ["air_temperature", "direct_recharge", "indirect_recharge", "gw_depth"]

            # make bar plot for absolute anomaly metrics
            fig, ax = plt.subplots(nrows=1, ncols=len(variables), figsize=(len(variables)*2, 3))
            for i, variable in enumerate(variables):
                df = df_anomaly_metrics_abs_scenario_area[(df_anomaly_metrics_abs_scenario_area["variable"] == variable) & (df_anomaly_metrics_abs_scenario_area["metric"] == "average")]
                val = df["value"].values[0]
                # plot orange if negative, blue if positive
                colors = ["orange" if val < 0 else "blue"]
                ax[i].bar(0, val, color=colors)
                ax[i].axhline(0, color="black", linewidth=1, linestyle="-")
                # remove x-ticks
                ax[i].set_xticks([])
                ax[i].set_xlabel("")
                ax[i].set_ylabel(f"{_labels[variable]}{_units_abs[variable]}")
                if val < 0:
                    ax[i].set_ylim(val*1.1, abs(val)*1.1)
                else:
                    ax[i].set_ylim(-val*1.1, val*1.1)
            fig.tight_layout()
            fig.savefig(base_path_output / f"anomaly_metrics_abs_{stress_test_scenario}_{area}.png")
            plt.close(fig)

            # make bar plot for relative anomaly metrics
            fig, ax = plt.subplots(nrows=1, ncols=len(variables), figsize=(len(variables)*2, 3))
            for i, variable in enumerate(variables):
                df = df_anomaly_metrics_rel_scenario_area[(df_anomaly_metrics_rel_scenario_area["variable"] == variable) & (df_anomaly_metrics_rel_scenario_area["metric"] == "average")]
                val = df["value"].values[0]
                # plot orange if negative, blue if positive
                colors = ["orange" if val < 0 else "blue"]
                ax[i].bar(0, val, color=colors)
                ax[i].axhline(0, color="black", linewidth=1, linestyle="-")
                # remove x-ticks
                ax[i].set_xticks([])
                ax[i].set_xlabel("")
                ax[i].set_ylabel(f"{_labels[variable]} [%]")
                if variable == "air_temperature":
                    ax[i].set_ylim(-100, 100)
                else:
                    ax[i].set_ylim(-10, 10)
            fig.tight_layout()
            fig.savefig(base_path_output / f"anomaly_metrics_rel_{stress_test_scenario}_{area}.png")
            plt.close(fig)

    return

if __name__ == "__main__":
    main()
