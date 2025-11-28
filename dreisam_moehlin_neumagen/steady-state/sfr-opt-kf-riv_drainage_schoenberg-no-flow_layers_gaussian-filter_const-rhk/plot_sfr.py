from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import scipy as sp
import rasterio
import matplotlib.pyplot as plt
import click

@click.command("main")
def main():

    base_path = Path(__file__).parent

    reaches = pd.read_csv(base_path.parent / "input" / "sfr_packagedata_modified.csv", sep=";")
    reaches.iloc[:, 0] = reaches.iloc[:, 0].astype(int) - 1  # convert to zero-based indexing
    reaches.iloc[:, 1] = reaches.iloc[:, 1].astype(int) - 1 # convert to zero-based indexing
    reaches.iloc[:, 2] = reaches.iloc[:, 2].astype(int) - 1  # convert to zero-based indexing
    reaches.iloc[:, 3] = reaches.iloc[:, 3].astype(int) - 1  # convert to zero-based indexing
    reaches.iloc[:, 4] = reaches.iloc[:, 4].astype(float) 
    reaches.iloc[:, 5] = reaches.iloc[:, 5].astype(int)
    reaches.iloc[:, 6] = reaches.iloc[:, 6].astype(float)
    reaches.iloc[:, 7] = reaches.iloc[:, 7].astype(float)
    reaches.iloc[:, 8] = reaches.iloc[:, 8].astype(float)
    reaches.iloc[:, 9] = reaches.iloc[:, 9].astype(float)
    reaches.iloc[:, 10] = reaches.iloc[:, 10].astype(float)
    reaches.iloc[:, 11] = reaches.iloc[:, 11].astype(int)
    reaches.iloc[:, 12] = reaches.iloc[:, 12].astype(float)
    reaches.iloc[:, 13] = reaches.iloc[:, 13].astype(int)
    reaches.iloc[:, 15] = reaches.iloc[:, 15].astype(float)
    reaches.iloc[:, 16] = reaches.iloc[:, 16].astype(int)


    # plot distribution of reach hydraulic conductivity
    fig, axes = plt.subplots(figsize=(4, 4))
    axes.hist(reaches.iloc[:, 9], bins=np.logspace(np.log10(1e-8),np.log10(0.1), 70), color='blue', alpha=0.7)
    # set log scale for x axis
    axes.set_xscale('log')
    axes.set_xlabel('Hydraulic Conductivity (m/s)')
    axes.set_ylabel('Frequency')
    file = base_path / "figures" / "dist_rhk.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    # standardize reach hydraulic conductivity
    rhk_std = (reaches.iloc[:, 9] - reaches.iloc[:, 9].mean()) / reaches.iloc[:, 9].std()
    fig, axes = plt.subplots(figsize=(4, 4))
    axes.hist(rhk_std/2, bins=50, color='blue', alpha=0.7)
    # set log scale for x axis
    axes.set_xlabel('Hydraulic Conductivity (m/s)')
    axes.set_ylabel('Frequency')
    file = base_path / "figures" / "dist_rhk_std.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    rhk_new = 5e-6 * 10**(rhk_std/2) 
     # plot distribution of reach hydraulic conductivity
    fig, axes = plt.subplots(figsize=(4, 4))
    axes.hist(rhk_new, bins=50, color='blue', alpha=0.7)
    # set log scale for x axis
    axes.set_xscale('log')
    axes.set_xlabel('Hydraulic Conductivity (m/s)')
    axes.set_ylabel('Frequency')
    file = base_path / "figures" / "dist_rhk_new.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    return

if __name__ == "__main__":
    main()
