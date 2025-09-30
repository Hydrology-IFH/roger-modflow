from pathlib import Path
import os
import shutil
import subprocess

base_path = Path(__file__).parent

subprocess.Popen("python write_fudge_parameters.py", shell=True)

# make directories of parallel jobs
for i in range(10):
    path_dir = base_path / f"batch_{i}"
    if not os.path.exists(path_dir):
        os.mkdir(path_dir)
    path_output = base_path / f"batch_{i}" / "output"
    if not os.path.exists(path_output):
        os.mkdir(path_output)
    shutil.copy(base_path / "fudge_parameters_modflow.csv", path_dir / "fudge_parameters_modflow.csv")
    shutil.copy(base_path / "modflow6_steady-state_.py", path_dir / "modflow6_steady-state.py")
    shutil.copy(base_path / "write_binary_to_netcdf_steady-state.py", path_dir / "write_binary_to_netcdf_steady-state.py")
    shutil.copy(base_path / "cleanup.py", path_dir / "cleanup.py")

start = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
end = [999, 1999, 2999, 3999, 4999, 5999, 6999, 7999, 8999, 9999]
for j in range(10):
    path_dir = base_path / f"batch_{j}"
    script_name = "modflow6_steady-state_monte_carlo"
    lines = []
    lines.append("#!/bin/bash\n")
    lines.append("for i in {%s..%s}\n" % (start[j], end[j]))
    lines.append("do\n")
    lines.append("\tconverged=$(python modflow6_steady-state.py --model-run $i | grep 'converged: ' | awk '{print $NF}')\n")
    lines.append("\tpython write_binary_to_netcdf_steady-state.py --model-run $i --converged $converged\n")
    lines.append("\tpython cleanup.py --model-run $i\n")
    lines.append("\tprogress=$i/10000\n")
    lines.append('\techo "Model run $i done ($progress; $converged)"\n')
    lines.append("done\n")
    file_path = path_dir / f"{script_name}.sh"
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    script_name = f"batch_{j}_slurm"
    lines = []
    lines.append("#!/bin/bash\n")
    lines.append("#SBATCH --time=72:00:00\n")
    lines.append("#SBATCH --nodes=1\n")
    lines.append("#SBATCH --ntasks=1\n")
    lines.append("#SBATCH --cpus-per-task=1\n")
    lines.append("#SBATCH --mem=32000\n")
    lines.append("#SBATCH --mail-type=FAIL\n")
    lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
    lines.append(f"#SBATCH --job-name={script_name}\n")
    lines.append(f"#SBATCH --output={script_name}.out\n")
    lines.append(f"#SBATCH --error={script_name}_err.out\n")
    lines.append("#SBATCH --export=ALL\n")
    lines.append(" \n")
    lines.append('module load devel/miniforge\n')
    lines.append('eval "$(conda shell.bash hook)"\n')
    lines.append("conda activate roger-modflow\n")
    lines.append(f"cd {str(path_dir)}\n")
    lines.append(" \n")
    lines.append(
        './modflow6_steady-state_monte_carlo.sh\n'
    )
    file_path = path_dir / f"{script_name}.sh"
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    subprocess.Popen(f"chmod +x {file_path}", shell=True)



