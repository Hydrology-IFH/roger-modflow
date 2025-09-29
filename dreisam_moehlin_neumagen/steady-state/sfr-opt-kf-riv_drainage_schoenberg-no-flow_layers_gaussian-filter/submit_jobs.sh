#!/bin/bash

module load devel/miniforge

conda activate roger-modflow

python make_parallel_jobs.py

cd batch_0
    sbatch -p cpu modflow6_steady-state_monte_carlo_slurm.sh
cd ../batch_1
    sbatch -p cpu modflow6_steady-state_monte_carlo_slurm.sh
cd ../batch_2
    sbatch -p cpu modflow6_steady-state_monte_carlo_slurm.sh
cd ../batch_3
    sbatch -p cpu modflow6_steady-state_monte_carlo_slurm.sh
cd ../batch_4
    sbatch -p cpu modflow6_steady-state_monte_carlo_slurm.sh
cd ../batch_5
    sbatch -p cpu modflow6_steady-state_monte_carlo_slurm.sh
cd ../batch_6
    sbatch -p cpu modflow6_steady-state_monte_carlo_slurm.sh
cd ../batch_7
    sbatch -p cpu modflow6_steady-state_monte_carlo_slurm.sh
cd ../batch_8
    sbatch -p cpu modflow6_steady-state_monte_carlo_slurm.sh
cd ../batch_9
    sbatch -p cpu modflow6_steady-state_monte_carlo_slurm.sh
cd ..