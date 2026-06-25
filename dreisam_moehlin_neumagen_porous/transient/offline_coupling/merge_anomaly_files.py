from pathlib import Path
import pandas as pd
import click

@click.command("main", short_help="Calculate annual anomalies for MODFLOW output")
def main():
    # base_path = Path(__file__).parent
    base_path = Path("/Volumes/LaCie/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline-coupling")
    areas = ["dmn", "wsg_hausen", "wsg_zartener_becken", "wsg_boetzingen", 
             "wsg_breisach", "wsg_ebringen", "wsg_eichstetten", "wsg_gottenheim", 
             "wsg_krozinger_berg", "wsg_march", "wsg_schlatt", "wsg_tuniberg", "wsg_umkirch"]
    
    # merge annual data
    model_run = 1806
    ll_dfs_annual_values = []
    ll_dfs_annual_anomalies_abs = []
    ll_dfs_annual_anomalies_rel = []
    for area in areas:
        output_file = base_path / "output" / f"annual_values_run{model_run}_{area}_roger.csv"
        df_annual_values = pd.read_csv(output_file, sep=";")
        ll_dfs_annual_values.append(df_annual_values)
        output_file = base_path / "output" / f"annual_values_run{model_run}_{area}_modflow.csv"
        df_annual_values = pd.read_csv(output_file, sep=";")
        ll_dfs_annual_values.append(df_annual_values)
        output_file = base_path / "output" / f"annual_anomalies_abs_run{model_run}_{area}_roger.csv"
        df_anomaly_metrics_abs = pd.read_csv(output_file, sep=";")
        ll_dfs_annual_anomalies_abs.append(df_anomaly_metrics_abs)
        output_file = base_path / "output" / f"annual_anomalies_abs_run{model_run}_{area}_modflow.csv"
        df_anomaly_metrics_abs = pd.read_csv(output_file, sep=";")
        ll_dfs_annual_anomalies_abs.append(df_anomaly_metrics_abs)
        output_file = base_path / "output" / f"annual_anomalies_rel_run{model_run}_{area}_roger.csv"
        df_anomaly_metrics_rel = pd.read_csv(output_file, sep=";")
        ll_dfs_annual_anomalies_rel.append(df_anomaly_metrics_rel)
        output_file = base_path / "output" / f"annual_anomalies_rel_run{model_run}_{area}_modflow.csv"
        df_anomaly_metrics_rel = pd.read_csv(output_file, sep=";")
        ll_dfs_annual_anomalies_rel.append(df_anomaly_metrics_rel)

    df_annual_values = pd.concat(ll_dfs_annual_values, ignore_index=True)
    df_annual_anomalies_abs = pd.concat(ll_dfs_annual_anomalies_abs, ignore_index=True)
    df_annual_anomalies_rel = pd.concat(ll_dfs_annual_anomalies_rel, ignore_index=True)

    output_file = base_path / "output" / f"annual_values_run{model_run}.csv"
    df_annual_values.to_csv(output_file, sep=";", index=False)
    output_file = base_path / "output" / f"annual_anomalies_abs_run{model_run}.csv"
    df_annual_anomalies_abs.to_csv(output_file, sep=";", index=False)
    output_file = base_path / "output" / f"annual_anomalies_rel_run{model_run}.csv"
    df_annual_anomalies_rel.to_csv(output_file, sep=";", index=False)

    # # merge daily data
    # ll_dfs_daily_values = []
    # ll_dfs_daily_anomalies_abs = []
    # ll_dfs_daily_anomalies_rel = []
    # for area in areas:
    #     output_file = base_path / "output" / f"daily_values_run{model_run}_{area}_roger.csv"
    #     df_daily_values = pd.read_csv(output_file, sep=";")
    #     ll_dfs_daily_values.append(df_daily_values)
    #     output_file = base_path / "output" / f"daily_anomalies_abs_run{model_run}_{area}_roger.csv"
    #     df_daily_anomalies_abs = pd.read_csv(output_file, sep=";")
    #     ll_dfs_daily_anomalies_abs.append(df_daily_anomalies_abs)
    #     output_file = base_path / "output" / f"daily_anomalies_rel_run{model_run}_{area}_roger.csv"
    #     df_daily_anomalies_rel = pd.read_csv(output_file, sep=";")
    #     ll_dfs_daily_anomalies_rel.append(df_daily_anomalies_rel)
    #     output_file = base_path / "output" / f"daily_anomalies_rel_run{model_run}_{area}_modflow.csv"
    #     df_daily_anomalies_rel = pd.read_csv(output_file, sep=";")
    #     ll_dfs_daily_anomalies_rel.append(df_daily_anomalies_rel)
    #     output_file = base_path / "output" / f"daily_anomalies_abs_run{model_run}_{area}_modflow.csv"
    #     df_daily_anomalies_abs = pd.read_csv(output_file, sep=";")
    #     ll_dfs_daily_anomalies_abs.append(df_daily_anomalies_abs)
    #     output_file = base_path / "output" / f"daily_anomalies_abs_run{model_run}_{area}_roger.csv"
    #     df_daily_anomalies_abs = pd.read_csv(output_file, sep=";")
    #     ll_dfs_daily_anomalies_abs.append(df_daily_anomalies_abs)
    #     output_file = base_path / "output" / f"daily_anomalies_rel_run{model_run}_{area}_modflow.csv"
    #     df_daily_anomalies_rel = pd.read_csv(output_file, sep=";")
    #     ll_dfs_daily_anomalies_rel.append(df_daily_anomalies_rel)

    # df_daily_values = pd.concat(ll_dfs_daily_values, ignore_index=True)
    # df_daily_anomalies_abs = pd.concat(ll_dfs_daily_anomalies_abs, ignore_index=True)
    # df_daily_anomalies_rel = pd.concat(ll_dfs_daily_anomalies_rel, ignore_index=True)

    # output_file = base_path / "output" / f"daily_values_run{model_run}.csv"
    # df_daily_values.to_csv(output_file, sep=";", index=False)
    # output_file = base_path / "output" / f"daily_anomalies_abs_run{model_run}.csv"
    # df_daily_anomalies_abs.to_csv(output_file, sep=";", index=False)
    # output_file = base_path / "output" / f"daily_anomalies_rel_run{model_run}.csv"
    # df_daily_anomalies_rel.to_csv(output_file, sep=";", index=False)

if __name__ == "__main__":
    main()