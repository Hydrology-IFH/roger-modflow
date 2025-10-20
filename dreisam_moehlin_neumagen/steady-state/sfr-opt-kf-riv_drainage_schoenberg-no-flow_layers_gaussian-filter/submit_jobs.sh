#!/bin/bash

module load devel/miniforge
conda activate roger-modflow

python write_fudge_parameters.py
python make_parallel_jobs.py

cd batch_0
sbatch -p cpu_il batch_0_slurm.sh
cd ../batch_1
sbatch -p cpu_il batch_1_slurm.sh
cd ../batch_2
sbatch -p cpu_il batch_2_slurm.sh
cd ../batch_3
sbatch -p cpu_il batch_3_slurm.sh
cd ../batch_4
sbatch -p cpu_il batch_4_slurm.sh
cd ../batch_5
sbatch -p cpu_il batch_5_slurm.sh
cd ../batch_6
sbatch -p cpu_il batch_6_slurm.sh
cd ../batch_7
sbatch -p cpu_il batch_7_slurm.sh
cd ../batch_8
sbatch -p cpu_il batch_8_slurm.sh
cd ../batch_9
sbatch -p cpu_il batch_9_slurm.sh
cd ..