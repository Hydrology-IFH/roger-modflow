#!/usr/bin/env python3
"""
aggregate_plot.py

Loads PET.txt, TA.txt and (optionally) PREC.txt (whitespace-delimited), aggregates to daily values
(using sum for PET and PREC, mean for TA), aggregates to seasonal values (DJF, MAM, JJA, SON)
and plots seasonal time series (one plot per season).

Usage:
    python aggregate_plot.py

Outputs:
    - figures/seasonal_plot_{SEASON}.png  (one file per season)
    - figures/seasonal_aggregates.csv      (seasonal numeric table)
    - figures/daily_merged.csv             (daily merged table)

Notes:
    - PREC is optional: if PREC.txt is present it will be included and treated with daily sums.
    - The script is robust to files that already contain daily records; it will
      resample/group to daily first (so duplicate timestamps are handled).
"""
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("ticks")

base_path = Path(__file__).parent  # current directory; change if files are elsewhere

PET_FILE = base_path / "input" / "meteo" / "2000-2024" / "1443" / "PET.txt"
TA_FILE = base_path / "input" / "meteo" / "2000-2024" / "1443" / "TA.txt"
PREC_FILE = base_path / "input" / "meteo" / "2000-2024" / "1443" / "PREC.txt"  # new optional precipitation file

def read_whitespace_table(path, parse_datetime_cols=True):
    """Read a whitespace-delimited file with YYYY MM DD hh mm columns and return DataFrame with a datetime index.
    Column names are normalized to uppercase for convenience.
    """
    df = pd.read_csv(path, sep=r"\s+", skiprows=0, header=0, na_values=-9999)
    # Expect columns named YYYY, MM, DD, hh, mm (case-insensitive)
    cols_upper = {c.upper(): c for c in df.columns}
    required = {"YYYY", "MM", "DD"}
    if required.issubset({c.upper() for c in df.columns}) and parse_datetime_cols:
        year_col = "YYYY"
        month_col = "MM"
        day_col = "DD"
        # find hour/min if present, else set to 0
        hh_col = "hh"
        mm_col = "mm"

        df["datetime"] = pd.to_datetime(df[[year_col, month_col, day_col, hh_col, mm_col]].rename(columns={
            year_col: "year", month_col: "month", day_col: "day", hh_col: "hour", mm_col: "minute"
        }), errors="coerce")
        # drop the original date component columns to leave only measurement columns
        drop_cols = [year_col, month_col, day_col]
        if year_col in df.columns: df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
        if hh_col == "_HH_TMP":
            df = df.drop(columns=["_HH_TMP"], errors="ignore")
        if mm_col == "_MM_TMP":
            df = df.drop(columns=["_MM_TMP"], errors="ignore")
        df = df.drop(columns=[c for c in [cols_upper.get("HH"), cols_upper.get("MI"), cols_upper.get("mm")] if c in df.columns], errors="ignore")
        df = df.set_index("datetime").sort_index()
    else:
        # Try to parse a "date" or "datetime" column if present
        datecol = None
        for candidate in ("date", "datetime", "time", "timestamp"):
            if candidate in df.columns:
                datecol = candidate
                break
            if candidate.upper() in {c.upper() for c in df.columns}:
                # find actual column name
                for c in df.columns:
                    if c.upper() == candidate.upper():
                        datecol = c
                        break
                if datecol:
                    break
        if datecol:
            df["datetime"] = pd.to_datetime(df[datecol], errors="coerce")
            df = df.set_index("datetime").sort_index()
        else:
            raise ValueError(f"Could not find date/time columns in {path}")
    # normalize remaining column names to uppercase (makes later detection easier)
    df.columns = [c.upper() for c in df.columns]
    # drop any rows without a valid datetime index
    df = df[~df.index.isna()]
    return df

def detect_precip_column(df):
    """Return the precipitation column name in df (already uppercased) or None."""
    if df is None:
        return None
    candidates = ["PREC", "PRCP", "PRECIP", "PRECIPITATION", "PRC"]
    for c in df.columns:
        if c.upper() in candidates:
            return c
    return None

def daily_aggregate(pet_df=None, ta_df=None, prec_df=None):
    """
    Aggregate source dataframes to daily frequency.
    - PET and PREC: sum per day
    - TA (and TA_MIN/TA_MAX if present): mean per day
    Returns merged daily DataFrame.
    """
    daily_frames = []
    if pet_df is not None:
        # ensure PET col present (case-insensitive handled by uppercase normalization)
        if "PET" not in pet_df.columns:
            # try to find PET-like column
            pet_candidates = [c for c in pet_df.columns if c.upper() == "PET"]
            if pet_candidates:
                pet_df = pet_df.rename(columns={pet_candidates[0]: "PET"})
            else:
                raise ValueError("No PET column found in PET file")
        pet_daily = pet_df[["PET"]].resample("D").agg({"PET": "sum"})
        daily_frames.append(pet_daily)
    if ta_df is not None:
        ta_cols = [c for c in ta_df.columns if c.upper().startswith("TA")]
        if not ta_cols:
            raise ValueError("No TA* columns found in TA file")
        ta_daily = ta_df[ta_cols].resample("D").mean()
        daily_frames.append(ta_daily)
    if prec_df is not None:
        prec_col = detect_precip_column(prec_df)
        if prec_col is None:
            raise ValueError("PREC file provided but no precipitation column found (expected PREC/PRCP/etc.)")
        prec_daily = prec_df[[prec_col]].resample("D").sum().rename(columns={prec_col: "PREC"})
        daily_frames.append(prec_daily)

    if not daily_frames:
        raise ValueError("No input frames provided for daily aggregation")
    # Merge all daily frames on index (outer join)
    daily = pd.concat(daily_frames, axis=1, join="outer")
    daily = daily[~daily.isna().all(axis=1)].copy()
    # order columns: PET, PREC, TA*
    cols_order = []
    if "PET" in daily.columns: cols_order.append("PET")
    if "PREC" in daily.columns: cols_order.append("PREC")
    ta_cols = [c for c in daily.columns if c.startswith("TA")]
    cols_order.extend(ta_cols)
    # add remaining columns
    for c in daily.columns:
        if c not in cols_order:
            cols_order.append(c)
    daily = daily[cols_order]
    return daily

def season_label_and_year(idx):
    """
    Given a DatetimeIndex, produce (season_label, season_year) arrays.
    Seasons used (meteorological):
      DJF: Dec-Jan-Feb
      MAM: Mar-Apr-May
      JJA: Jun-Jul-Aug
      SON: Sep-Oct-Nov
    season_year: for DJF, December belongs to the next year (e.g., Dec 2013 -> DJF 2014)
    """
    months = idx.month
    years = idx.year
    season = []
    season_year = []
    for m, y in zip(months, years):
        if m in (12, 1, 2):
            s = "DJF"
            sy = y + 1 if m == 12 else y
        elif m in (3, 4, 5):
            s = "MAM"
            sy = y
        elif m in (6, 7, 8):
            s = "JJA"
            sy = y
        else:
            s = "SON"
            sy = y
        season.append(s)
        season_year.append(sy)
    return np.array(season), np.array(season_year)

def seasonal_aggregate(daily_df):
    """
    Aggregate daily_df into seasonal values.
    - For PET/PREC columns: sum over season
    - For TA* columns: mean over season
    Returns a MultiIndex DataFrame indexed by (season, season_year).
    """
    df = daily_df.copy()
    season, season_year = season_label_and_year(df.index)
    df = df.assign(SEASON=season, SEASON_YEAR=season_year)
    # Define aggregation mapping
    agg_map = {}
    for col in df.columns:
        if col in ("SEASON", "SEASON_YEAR"):
            continue
        if col in ("PET", "PREC"):
            agg_map[col] = "sum"
        elif col.startswith("TA"):
            agg_map[col] = "mean"
        else:
            agg_map[col] = "mean"

    grouped = df.groupby(["SEASON", "SEASON_YEAR"]).agg(agg_map)
    # Sort index by season order and year
    season_order = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}
    # add temporary column for ordering
    grp_index = grouped.index.to_frame(index=False)
    grouped = grouped.assign(_season_order=[season_order[s] for s in grp_index["SEASON"]])
    grouped = grouped.sort_values(["_season_order", "SEASON_YEAR"])
    grouped = grouped.drop(columns=["_season_order"])
    # set nicer index names
    grouped.index.set_names(["season", "season_year"], inplace=True)
    return grouped

def plot_seasonal_timeseries(seasonal_df, out_dir=Path(".")):
    """
    For each season (DJF, MAM, JJA, SON) create a plot of the seasonal time series.
    Plots PET (line) and TA (line on secondary axis). PREC (if present) is plotted as bars.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seasons = ["DJF", "MAM", "JJA", "SON"]
    for s in seasons:
        if s not in seasonal_df.index.get_level_values(0):
            print(f"Skipping season {s} (no data).")
            continue
        df_s = seasonal_df.loc[s].sort_index()  # index is season_year
        years = df_s.index.values
        fig, ax1 = plt.subplots(figsize=(6, 3))
        color_pet = "tab:green"
        color_ta = "tab:orange"
        color_prec = "tab:blue"

        plotted = False
        if "PREC" in df_s.columns:
            # precipitation often on different scale; plot as bars on the same axis with alpha
            ax1.bar(years, df_s["PREC"], alpha=0.35, label="PREC", color=color_prec)
            # plot average as horizontal line
            ax1.axhline(df_s["PREC"].mean(), color=color_prec, linestyle="--", alpha=0.7)
            plotted = True

        if "PET" in df_s.columns:
            ax1.plot(years, df_s["PET"], marker="o", color=color_pet, label="PET")
            # plot average as horizontal line
            ax1.axhline(df_s["PET"].mean(), color=color_pet, linestyle="--", alpha=0.7)
            ax1.set_ylabel("PET [mm/3 month]\n PREC [mm/3 month]")
            ax1.tick_params(axis="y")
            plotted = True

        # TA on secondary axis (if present)
        ta_columns = [c for c in df_s.columns if c.startswith("TA")]
        ax2 = None
        if ta_columns:
            ax2 = ax1.twinx()
            ax2.plot(years, df_s[ta_columns[0]], marker="s", color=color_ta, label=f"{ta_columns[0]}")
            # plot average as horizontal line
            ax2.axhline(df_s[ta_columns[0]].mean(), color=color_ta, linestyle="--", alpha=0.7)
            ax2.set_ylabel(f"{ta_columns[0]} [degC]")
            ax2.tick_params(axis="y")
            plotted = True
            # if TA_min / TA_max exist, show shaded range
            if "TA_MIN" in df_s.columns and "TA_MAX" in df_s.columns:
                ax2.fill_between(years, df_s["TA_MIN"], df_s["TA_MAX"], color=color_ta, alpha=0.12, label="TA_min/max range")

        if not plotted:
            print(f"No PET/TA/PREC data to plot for season {s}, skipping plot.")
            plt.close(fig)
            continue

        # Title and legend
        lines, labels = ax1.get_legend_handles_labels()
        if ax2:
            l2, lab2 = ax2.get_legend_handles_labels()
            lines += l2
            labels += lab2
        ax1.legend(lines, labels, loc="upper center", ncol=3, frameon=False)
        ax1.set_xlabel("Time [year]")
        fig.tight_layout()
        out_path = out_dir / f"seasonal_plot_{s}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

def main():
    # 1) Load files
    if not PET_FILE.exists():
        raise FileNotFoundError(f"{PET_FILE} not found. Put PET.txt at that path or update PET_FILE variable.")
    if not TA_FILE.exists():
        raise FileNotFoundError(f"{TA_FILE} not found. Put TA.txt at that path or update TA_FILE variable.")

    print("Reading PET.txt ...")
    pet_df = read_whitespace_table(PET_FILE)
    # ensure PET column exists (uppercased)
    if "PET" not in pet_df.columns:
        candidates = [c for c in pet_df.columns if c.upper() == "PET"]
        if candidates:
            pet_df = pet_df.rename(columns={candidates[0]: "PET"})
        else:
            raise ValueError("PET column not found in PET.txt")

    print("Reading TA.txt ...")
    ta_df = read_whitespace_table(TA_FILE)
    ta_cols = [c for c in ta_df.columns if c.startswith("TA")]
    if not ta_cols:
        candidates = [c for c in ta_df.columns if c.upper().startswith("TA")]
        if candidates:
            # normalize names
            ta_df = ta_df.rename(columns={c: c.upper() for c in candidates})
        else:
            raise ValueError("TA columns not found in TA.txt")

    # Try to read PREC.txt if present (it's optional)
    prec_df = None
    if PREC_FILE.exists():
        print("Reading PREC.txt ...")
        prec_df = read_whitespace_table(PREC_FILE)
        prec_col = detect_precip_column(prec_df)
        if prec_col is None:
            raise ValueError("PREC file read but no precipitation column detected (expected PREC/PRCP/etc.)")
        # Keep precipitation column name as-is in the raw df; daily aggregation will rename to PREC
        print(f"Detected precipitation column: {prec_col}")
    else:
        print(f"No PREC.txt found at {PREC_FILE}. Continuing without precipitation.")

    # 2) Aggregate to daily values
    print("Aggregating to daily values...")
    daily = daily_aggregate(pet_df=pet_df, ta_df=ta_df, prec_df=prec_df)


    # 3) Aggregate to seasonal values
    print("Aggregating to seasonal values...")
    seasonal = seasonal_aggregate(daily)

    # 4) Plot the time series for each season separately
    print("Plotting seasonal time series (one figure per season)...")
    plot_seasonal_timeseries(seasonal, out_dir=base_path / "figures")

if __name__ == "__main__":
    main()