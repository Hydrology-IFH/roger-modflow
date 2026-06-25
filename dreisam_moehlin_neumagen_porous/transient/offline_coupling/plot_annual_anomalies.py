from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import click


@click.option("-mr", "--model-run", type=int, default=1806)
@click.command("main", short_help="Evaluate the transient simulation")
def main(model_run):
    base_path_output = Path("/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline-coupling") / "output"
    base_path_figures = Path("/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline-coupling") / "figures"

    areas = ["dmn", "wsg_hausen", "wsg_zartener_becken", "wsg_boetzingen", "wsg_breisach", "wsg_ebringen", "wsg_eichstetten", "wsg_gottenheim", "wsg_krozinger_berg", "wsg_march", "wsg_schlatt", "wsg_tuniberg", "wsg_umkirch"]

    stress_test_scenarios = ["base-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude0-duration3_no-irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude0-duration3_irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude0-duration3_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "summer-drought-magnitude0-duration3_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude2-duration3_irrigation_no-yellow-mustard_soil-compaction",
                             "summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "summer-drought-magnitude2-duration3_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "long-term-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction",
                             "long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction",
                             "long-term-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress",
                             "long-term-magnitude2-duration0_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress"]

    stress_test_scenarios = ["summer-drought-magnitude2-duration3_irrigation_no-yellow-mustard_soil-compaction_well-extraction-stress"]

    periods = ["overall", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]
    variables = ["air_temperature", "potential_evapotranspiration", "actual_evapotranspiration", "precipitation", "direct_recharge", "indirect_recharge", "gw_depth"]
    metrics = ["average", "5th_percentile", "25th_percentile", "median", "75th_percentile", "95th_percentile"]

    # load metrics
    output_file = base_path_output / f"annual_values_run{model_run}.csv"
    df_values = pd.read_csv(output_file, sep=";")
    output_file = base_path_output / f"annual_anomalies_abs_run{model_run}.csv"
    df_anomalies_abs = pd.read_csv(output_file, sep=";")
    output_file = base_path_output / f"annual_anomalies_rel_run{model_run}.csv"
    df_anomalies_rel = pd.read_csv(output_file, sep=";")

    _labels = {"air_temperature": "Lufttemp.",
               "potential_evapotranspiration": "PET",
               "actual_evapotranspiration": "AET",
               "precipitation": "Niederschlag",
               "direct_recharge": "Dir. GWN", 
               "indirect_recharge": "Indir. GWN", 
               "gw_depth": "GWFA",
               "well_extraction": "GW-Entnahme",
               "irrigation": "Bewaesserung"}
    
    _units_abs = {"air_temperature": " [°C]",
                  "potential_evapotranspiration": " [mm/Jahr]",
                  "actual_evapotranspiration": " [mm/Jahr]",
                  "precipitation": " [mm/Jahr]",
                  "direct_recharge": " [$m^3$/Jahr]", 
                  "indirect_recharge": " [$m^3$/Jahr]", 
                  "gw_depth": " [m]",
                  "well_extraction": " [$m^3$/Jahr]",
                  "irrigation": " [mm/Jahr]"}

    for area in areas:
        for stress_test_scenario in stress_test_scenarios:
            for period in periods:
                # select metrics for the current scenario and area
                df_values_scenario_area = df_values[(df_values["scenario"] == stress_test_scenario) & (df_values["area"] == area) & (df_values["time"] == period)]
                df_anomalies_abs_scenario_area = df_anomalies_abs[(df_anomalies_abs["scenario"] == stress_test_scenario) & (df_anomalies_abs["area"] == area) & (df_values["time"] == period)]
                df_anomalies_rel_scenario_area = df_anomalies_rel[(df_anomalies_rel["scenario"] == stress_test_scenario) & (df_anomalies_rel["area"] == area) & (df_values["time"] == period)]
                for metric in metrics:
                    # make bar plot for absolute values
                    fig, ax = plt.subplots(nrows=1, ncols=len(variables), figsize=(len(variables), 2))
                    for i, variable in enumerate(variables):
                        df = df_values_scenario_area[(df_values_scenario_area["variable"] == variable) & (df_values_scenario_area["metric"] == metric)]
                        val = df["value"].values[0]
                        ax[i].bar(0, val, color="black")
                        # remove x-ticks
                        ax[i].set_xticks([])
                        ax[i].set_xlabel("")
                        ax[i].set_ylabel(f"{_labels[variable]}{_units_abs[variable]}")
                        if val < 0:
                            ax[i].set_ylim(val*1.1, abs(val)*1.1)
                        else:
                            ax[i].set_ylim(-val*1.1, val*1.1)
                        if variable == "gw_depth":
                            ax[i].invert_yaxis()
                    fig.tight_layout()
                    fig.savefig(base_path_figures / f"barplot_annual_{metric}_{stress_test_scenario}_{area}_{period}.pdf", dpi=300)
                    plt.close(fig)

                    # make bar plot for absolute anomaly metrics
                    fig, ax = plt.subplots(nrows=1, ncols=len(variables), figsize=(len(variables), 2))
                    for i, variable in enumerate(variables):
                        df = df_anomalies_abs_scenario_area[(df_anomalies_abs_scenario_area["variable"] == variable) & (df_anomalies_abs_scenario_area["metric"] == metric)]
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
                    fig.savefig(base_path_figures / f"barplot_annual_anomalies_abs_{metric}_{stress_test_scenario}_{area}_{period}.pdf", dpi=300)
                    plt.close(fig)

                    # make bar plot for relative anomaly metrics
                    fig, ax = plt.subplots(nrows=1, ncols=len(variables), figsize=(len(variables), 2))
                    for i, variable in enumerate(variables):
                        df = df_anomalies_rel_scenario_area[(df_anomalies_rel_scenario_area["variable"] == variable) & (df_anomalies_rel_scenario_area["metric"] == metric)]
                        val = df["value"].values[0]
                        # plot orange if negative, blue if positive
                        colors = ["orange" if val < 0 else "blue"]
                        ax[i].bar(0, val, color=colors)
                        ax[i].axhline(0, color="black", linewidth=1, linestyle="-")
                        # remove x-ticks
                        ax[i].set_xticks([])
                        ax[i].set_xlabel("")
                        ax[i].set_ylabel(f"{_labels[variable]} [%]")
                        ax[i].set_ylim(-100, 100)
                    fig.tight_layout()
                    fig.savefig(base_path_figures / f"barplot_annual_anomalies_rel_{metric}_{stress_test_scenario}_{area}_{period}.pdf", dpi=300)
                    plt.close(fig)

                # make box plot using the 5th, 25th, 50th, 75th, and 95th percentiles
                fig, ax = plt.subplots(nrows=1, ncols=len(variables), figsize=(len(variables), 2))
                for i, variable in enumerate(variables):
                    df = df_values_scenario_area[df_values_scenario_area["variable"] == variable]
                    vals = [df[df["metric"] == m]["value"].values[0] for m in ["5th_percentile", "25th_percentile", "median", "75th_percentile", "95th_percentile"]]
                    ax[i].boxplot(vals, positions=[0], widths=0.6, patch_artist=True, boxprops=dict(facecolor="grey"))
                    # remove x-ticks
                    ax[i].set_xticks([])
                    ax[i].set_xlabel("")
                    ax[i].set_ylabel(f"{_labels[variable]}{_units_abs[variable]}")
                    if variable == "gw_depth":
                        ax[i].invert_yaxis()
                fig.tight_layout()
                fig.savefig(base_path_figures / f"boxplot_annual_{stress_test_scenario}_{area}_{period}.pdf", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots(nrows=1, ncols=len(variables), figsize=(len(variables), 2))
                for i, variable in enumerate(variables):
                    df = df_anomalies_abs_scenario_area[df_anomalies_abs_scenario_area["variable"] == variable]
                    vals = [df[df["metric"] == m]["value"].values[0] for m in ["5th_percentile", "25th_percentile", "median", "75th_percentile", "95th_percentile"]]
                    ax[i].boxplot(vals, positions=[0], widths=0.6, patch_artist=True, boxprops=dict(facecolor="grey"))
                    # remove x-ticks
                    ax[i].set_xticks([])
                    ax[i].set_xlabel("")
                    ax[i].set_ylabel(f"{_labels[variable]}{_units_abs[variable]}")
                fig.tight_layout()
                fig.savefig(base_path_figures / f"boxplot_annual_anomalies_abs_{stress_test_scenario}_{area}_{period}.pdf", dpi=300)
                plt.close(fig)

                fig, ax = plt.subplots(nrows=1, ncols=len(variables), figsize=(len(variables), 2))
                for i, variable in enumerate(variables):
                    df = df_anomalies_rel_scenario_area[df_anomalies_rel_scenario_area["variable"] == variable]
                    vals = [df[df["metric"] == m]["value"].values[0] for m in ["5th_percentile", "25th_percentile", "median", "75th_percentile", "95th_percentile"]]
                    ax[i].boxplot(vals, positions=[0], widths=0.6, patch_artist=True, boxprops=dict(facecolor="grey"))
                    # remove x-ticks
                    ax[i].set_xticks([])
                    ax[i].set_xlabel("")
                    ax[i].set_ylabel(f"{_labels[variable]} [%]")
                    ax[i].set_ylim(-100, 100)
                    ax[i].axhline(0, color="black", linewidth=1, linestyle="-")
                fig.tight_layout()
                fig.savefig(base_path_figures / f"boxplot_annual_anomalies_rel_{stress_test_scenario}_{area}_{period}.pdf", dpi=300)
                plt.close(fig)


    return

if __name__ == "__main__":
    main()
