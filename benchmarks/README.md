# Simple example to benchmark the computation time of RoGeR and MODFLOW

RoGeR and MODFLOW6 use a homogeneous paramterization (i.e. each grid cell uses the same parameters). In order to benchmark the computation time, the number of grid cell is increased stepwise

Short description of files and folders:
- `output/`: contains the model output
- `figures/`: contains the figures
- `config.yml`: File to set settings and parameters of MODFLOW6
- `modflow6_transient_for_benchmark.py`: Python script to run MODFLOW6 in transient mode using xmipy
- `modflow6_steady-state_for_benchmark.py`: Python script to run MODFLOW6 in steady-state mode using xmipy
- `roger_for_benchmark.py`: Python script to run RoGeR
- `roger_modflow6_transient_for_benchmark.py`: Python script to run RoGeR coupled with MODFLOW6 in transient mode using xmipy
- `plot_benchmarks.py`: Plots the computational times for different number of grid cells

Workflow:
1. Run `python modflow6_for_benchmark_steady-state.py`
2. Run `python modflow6_for_benchmark_transient.py`
3. Run `python roger_for_benchmark.py --backend numpy`
4. Run `python roger_for_benchmark.py--backend jax`
5. Run `python roger_modflow6_transient_for_benchmark.py --backend numpy`
6. Run `python roger_modflow6_transient_for_benchmark.py --backend jax`
Note: See https://roger.readthedocs.io/en/latest/ for backend