import xarray as xr
import h5netcdf
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import click
import flopy

@click.option("-mr", "--model-run", type=int, default=0)
@click.option("--plot", type=int, is_flag=True, help="Print more output.")
@click.command("main", short_help="Run MODFLOW in steady-state mode")
def main(model_run, plot):
    try:
        path = Path(__file__).parent / "parameters_modflow.nc"
        ds_params = xr.open_dataset(path, engine="h5netcdf")

        path = Path(__file__).parent / "boundary_conditions.nc"
        ds_bc = xr.open_dataset(path, engine="h5netcdf")

        fhead = Path(__file__).parent  / "output" / "steady-state" / f"dmn_run_{model_run}.hds"
        hds = flopy.utils.HeadFile(fhead)
        head = hds.get_data()[1, :, :]

        topography = ds_params['elevations'].isel(z=0).values

        mask_bc = (ds_bc["mask_constant_head"] == 1)

        # calculate gradients in flow direction
        head_ = np.empty((head.shape[0]+4, head.shape[1]+4))
        head_gradients = np.empty((head.shape[0]+4, head.shape[1]+4, 8))
        head_[2:-2, 2:-2] = head

        # head_gradients[2:-2, 2:-2, 0] = head_[2:-2, 2:-2] - head_[1:-3, 2:-2]
        head_gradients[2:-2, 2:-2, 1] = head_[2:-2, 2:-2] - head_[3:-1, 2:-2]
        # head_gradients[2:-2, 2:-2, 2] = head_[2:-2, 2:-2] - head_[2:-2, 1:-3]
        head_gradients[2:-2, 2:-2, 3] = head_[2:-2, 2:-2] - head_[2:-2, 3:-1]
        # head_gradients[2:-2, 2:-2, 4] = head_[2:-2, 2:-2] - head_[1:-3, 1:-3]
        # head_gradients[2:-2, 2:-2, 5] = head_[2:-2, 2:-2] - head_[1:-3, 3:-1]
        # head_gradients[2:-2, 2:-2, 6] = head_[2:-2, 2:-2] - head_[3:-1, 1:-3]
        head_gradients[2:-2, 2:-2, 7] = head_[2:-2, 2:-2] - head_[3:-1, 3:-1]

        # calculate the maximum gradient
        head_gradient_ = np.empty((head.shape[0]+4, head.shape[1]+4))
        for x in range(head.shape[0]):
            for y in range(head.shape[1]):
                if np.abs(np.nanmin(head_gradients[x, y, :])) > np.abs(np.nanmax(head_gradients[x, y, :])):
                    head_gradient_[x, y] = np.nanmin(head_gradients[x, y, :])
                else:
                    head_gradient_[x, y] = np.nanmax(head_gradients[x, y, :])

        head_gradient = head_gradient_[2:-2, 2:-2]
        # limit the gradient to the grid resolution of 50 m
        head_gradient[head_gradient > 5] = 5
        head_gradient[head_gradient < -5] = -5

        if plot:
            fig, axes = plt.subplots(figsize=(4, 4))
            # axes.scatter(np.where(mask_bc)[1], np.where(mask_bc)[0], s=0.5, c='k', alpha=0.5)
            plt.imshow(head_gradient, cmap='RdYlBu', aspect='equal', vmin=-5, vmax=5)
            plt.colorbar(label='[m]', shrink=0.5)
            plt.grid(zorder=0)
            plt.xlabel('x-direction')
            plt.ylabel('y-direction')
            plt.tight_layout()
            file = Path(__file__).parent / "figures" / "steady-state" / "gradients_grid.png"
            fig.savefig(file, dpi=300)
            plt.close(fig)

        head_gradient[~mask_bc] = np.nan
        head[~mask_bc] = np.nan
        topography[~mask_bc] = np.nan

        # set new constant by adding the gradient to the initial constant head
        constant_head_new = head - head_gradient
        constant_head_new[constant_head_new >= topography] = topography[constant_head_new >= topography] - 1

        ds_bc.close()
        # write the new boundary condition to the netcdf file
        path = str(Path(__file__).parent / "boundary_conditions.nc")
        with h5netcdf.File(path, "a", decode_vlen_strings=False) as f:
            var_obj = f.variables.get("constant_head")
            var_obj[:, :] = constant_head_new

        # values for plotting
        if plot:
            topography_vals = topography.flatten()[np.isfinite(topography.flatten())]
            gradient_vals = head_gradient.flatten()[np.isfinite(head_gradient.flatten())]
            constant_head_initial_vals = head.flatten()[np.isfinite(head.flatten())]
            constant_head_new_vals = constant_head_initial_vals - gradient_vals
            constant_head_new_vals[constant_head_new_vals >= topography_vals] = topography_vals[constant_head_new_vals >= topography_vals] - 2
            constant_head_new = head + head_gradient
            constant_head_new_vals[constant_head_new_vals >= topography_vals] = topography_vals[constant_head_new_vals >= topography_vals] - 2

            fig, axes = plt.subplots(figsize=(6, 3))
            axes.plot(range(len(topography_vals)), topography_vals, label="constant head initial", color="black", linestyle="--", alpha=0.5)
            axes.plot(range(len(constant_head_new_vals)), constant_head_new_vals, label="constant head new", color="purple", alpha=0.5)
            axes.plot(range(len(constant_head_initial_vals)), constant_head_initial_vals, label="constant head initial", color="black")
            axes.set_ylabel("elevation [m.a.s.l.]")
            fig.tight_layout()
            file = Path(__file__).parent / "figures" / "steady-state" / "evaluation_of_constant_head.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

            fig, axes = plt.subplots(figsize=(6, 3))
            axes.plot(range(len(gradient_vals)), gradient_vals, label="constant head initial", color="black", linestyle="-")
            axes.set_ylabel("[m]")
            fig.tight_layout()
            file = Path(__file__).parent / "figures" / "steady-state" / "gradient_constant_head.png"
            fig.savefig(file, dpi=300)
            plt.close("all")
    except:
        pass

    return

if __name__ == "__main__":
    main()