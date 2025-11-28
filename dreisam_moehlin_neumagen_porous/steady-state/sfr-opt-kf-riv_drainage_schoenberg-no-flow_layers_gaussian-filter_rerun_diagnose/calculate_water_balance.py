from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import rasterio
import click

@click.command("main", short_help="Run MODFLOW in steady-state mode")
def main():

    base_path = Path(__file__).parent


    # load the netcdf file
    output_file = base_path / "output" / "wsg_hausen.nc"
    ds_wsg_hausen = xr.open_dataset(output_file, engine="h5netcdf")
    mask_hausen = ds_wsg_hausen['mask'].values == 1
    _mask_hausen = np.array([mask_hausen for _ in range(ds_wsg_hausen.sizes['layer'])])

    output_file = base_path / "output" / "wsg_zartener_becken.nc"
    ds_wsg_zarten = xr.open_dataset(output_file, engine="h5netcdf")
    mask_zarten = ds_wsg_zarten['mask'].values == 1
    _mask_zarten = np.array([mask_zarten for _ in range(ds_wsg_zarten.sizes['layer'])])

    df_zarten = pd.DataFrame(index=[0], columns=["indirect_recharge", "direct_recharge", "gw_extraction", "streamflow", "gw_loss_to_surface_water", "lateral_outflow"])
    indirect_recharge_zarten = np.where(ds_wsg_zarten['indirect_recharge'].values < 0, 0, ds_wsg_zarten['indirect_recharge'].values)
    gw_loss_zarten = np.where(ds_wsg_zarten['indirect_recharge'].values > 0, 0, ds_wsg_zarten['indirect_recharge'].values * (-1))
    df_zarten.loc[0, "indirect_recharge"] = np.nansum(indirect_recharge_zarten[mask_zarten[np.newaxis, :, :]]) * 365 / 1000000  # in Mio m3/year
    df_zarten.loc[0, "gw_loss_to_surface_water"] = np.nansum(gw_loss_zarten[mask_zarten[np.newaxis, :, :]]) * 365 / 1000000  # in Mio m3/year
    df_zarten.loc[0, "direct_recharge"] = np.nansum(ds_wsg_zarten['direct_recharge'].values[mask_zarten[np.newaxis, :, :]]) * 365 / 1000000  # in Mio m3/year
    df_zarten.loc[0, "gw_extraction"] = np.nansum(ds_wsg_zarten['gw_extraction'].values[_mask_zarten[np.newaxis, :, :, :]]) * 365 / 1000000  # in Mio m3/year

    df_hausen = pd.DataFrame(index=[0], columns=["lateral_inflow", "indirect_recharge", "direct_recharge", "gw_extraction", "streamflow", "gw_loss_to_surface_water", "lateral_outflow"])
    indirect_recharge_hausen = np.where(ds_wsg_hausen['indirect_recharge'].values < 0, 0, ds_wsg_hausen['indirect_recharge'].values)
    gw_loss_hausen = np.where(ds_wsg_hausen['indirect_recharge'].values > 0, 0, ds_wsg_hausen['indirect_recharge'].values * (-1))
    df_hausen.loc[0, "indirect_recharge"] = np.nansum(indirect_recharge_hausen[mask_hausen[np.newaxis, :, :]]) * 365 / 1000000  # in Mio m3/year
    df_hausen.loc[0, "gw_loss_to_surface_water"] = np.nansum(gw_loss_hausen[mask_hausen[np.newaxis, :, :]]) * 365 / 1000000  # in Mio m3/year
    df_hausen.loc[0, "direct_recharge"] = np.nansum(ds_wsg_hausen['direct_recharge'].values[mask_hausen[np.newaxis, :, :]]) * 365 / 1000000  # in Mio m3/year
    df_hausen.loc[0, "gw_extraction"] = np.nansum(ds_wsg_hausen['gw_extraction'].values[_mask_hausen[np.newaxis, :, :, :]]) * 365 / 1000000  # in Mio m3/year

    file = base_path / "output" / "water_balance_wsg_zarten.csv"
    df_zarten.columns = [["[Mio m3/year] " for _ in df_zarten.columns],
                        df_zarten.columns]
    df_zarten.to_csv(file, sep=";", index=False)

    file = base_path / "output" / "water_balance_wsg_hausen.csv"
    df_hausen.columns = [["[Mio m3/year] " for _ in df_hausen.columns],
                        df_hausen.columns]
    df_hausen.to_csv(file, sep=";", index=False)

    return


if __name__ == "__main__":
    main()