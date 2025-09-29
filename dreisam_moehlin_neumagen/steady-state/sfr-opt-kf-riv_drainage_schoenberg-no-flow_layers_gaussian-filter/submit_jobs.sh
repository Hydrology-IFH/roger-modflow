#!/bin/bash

module load devel/miniforge

conda activate roger-modflow

python make_parallel_jobs.py

cd batch_0
    sbatch -p batch_0_slurm.sh
cd ../batch_1
    sbatch -p batch_1_slurm.sh
cd ../batch_2
    sbatch -p batch_2_slurm.sh
cd ../batch_3
    sbatch -p batch_3_slurm.sh
cd ../batch_4
    sbatch -p batch_4_slurm.sh
cd ../batch_5
    sbatch -p batch_5_slurm.sh
cd ../batch_6
    sbatch -p batch_6_slurm.sh
cd ../batch_7
    sbatch -p batch_7_slurm.sh
cd ../batch_8
    sbatch -p batch_8_slurm.sh
cd ../batch_9
    sbatch -p batch_9_slurm.sh