from pathlib import Path
import yaml
import subprocess
import os
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    dir_name = os.path.basename(str(Path(__file__).parent))
    base_path_bwhpc = "/pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling"
    script_name = "evaluate_transient_simulation"

    # identifiers of the simulations
    lines = []
    lines.append("#!/bin/bash\n")
    lines.append("#SBATCH --time=01:00:00\n")
    lines.append("#SBATCH --nodes=1\n")
    lines.append("#SBATCH --ntasks=1\n")
    lines.append("#SBATCH --cpus-per-task=1\n")
    lines.append("#SBATCH --mem=164000\n")
    lines.append("#SBATCH --mail-type=FAIL\n")
    lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
    lines.append(f"#SBATCH --job-name={script_name}\n")
    lines.append(f"#SBATCH --output={script_name}.out\n")
    lines.append(f"#SBATCH --error={script_name}_err.out\n")
    lines.append("#SBATCH --export=ALL\n")
    lines.append("\n")
    lines.append('module load devel/miniforge\n')
    lines.append("conda activate roger-modflow\n")
    lines.append(f"cd {base_path_bwhpc}\n")
    lines.append("\n")
    lines.append('python evaluate_transient_simulation.py\n')
    file_path = base_path / f"{script_name}.sh"
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()
