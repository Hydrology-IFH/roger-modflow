from pathlib import Path
import subprocess
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    base_path_bwhpc = "/pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling"

    # identifiers of the simulations
    areas = ["dmn", "wsg_hausen", "wsg_zartener_becken", "wsg_boetzingen", "wsg_breisach", "wsg_ebringen", "wsg_eichstetten", "wsg_gottenheim", "wsg_krozinger_berg", "wsg_march", "wsg_schlatt", "wsg_tuniberg", "wsg_umkirch"]
    jobs = []
    for area in areas:
        script_name = f"calculate_daily_anomalies_{area}_roger"
        lines = []
        lines.append("#!/bin/bash\n")
        lines.append("#SBATCH --time=48:00:00\n")
        lines.append("#SBATCH --ntasks=1\n")
        lines.append("#SBATCH --cpus-per-task=1\n")
        lines.append("#SBATCH --mem=512000\n")
        lines.append("#SBATCH --mail-type=FAIL\n")
        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
        lines.append(f"#SBATCH --job-name={script_name}\n")
        lines.append(f"#SBATCH --output={script_name}.out\n")
        lines.append(f"#SBATCH --error={script_name}_err.out\n")
        lines.append("#SBATCH --export=ALL\n")
        lines.append("module load devel/miniforge\n")
        lines.append("conda activate roger-modflow\n")
        lines.append(f"cd {base_path_bwhpc}\n")
        lines.append("\n")
        lines.append('python calculate_daily_anomalies_roger.py --area %s\n' % (area))
        file_path = base_path / f"{script_name}.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)
        jobs.append(f"{script_name}.sh")

    for area in areas:
        script_name = f"calculate_daily_anomalies_{area}_modflow"
        lines = []
        lines.append("#!/bin/bash\n")
        lines.append("#SBATCH --time=72:00:00\n")
        lines.append("#SBATCH --ntasks=1\n")
        lines.append("#SBATCH --cpus-per-task=1\n")
        lines.append("#SBATCH --mem=512000\n")
        lines.append("#SBATCH --mail-type=FAIL\n")
        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
        lines.append(f"#SBATCH --job-name={script_name}\n")
        lines.append(f"#SBATCH --output={script_name}.out\n")
        lines.append(f"#SBATCH --error={script_name}_err.out\n")
        lines.append("#SBATCH --export=ALL\n")
        lines.append("module load devel/miniforge\n")
        lines.append("conda activate roger-modflow\n")
        lines.append(f"cd {base_path_bwhpc}\n")
        lines.append("\n")
        lines.append('python calculate_daily_anomalies_modflow.py --area %s\n' % (area))
        file_path = base_path / f"{script_name}.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)
        jobs.append(f"{script_name}.sh")

    file_path = base_path / "submit_daily_anomaly_jobs.sh"
    with open(file_path, "w") as job_file:
        job_file.write("#!/bin/bash\n")
        job_file.write("\n")
        for job in jobs:
                job_file.write(f"sbatch -p compute {job}\n")
    subprocess.Popen(f"chmod +x {file_path}", shell=True)


    jobs = []
    for area in areas:
        script_name = f"calculate_annual_anomalies_{area}_roger"
        lines = []
        lines.append("#!/bin/bash\n")
        lines.append("#SBATCH --time=48:00:00\n")
        lines.append("#SBATCH --ntasks=1\n")
        lines.append("#SBATCH --cpus-per-task=1\n")
        lines.append("#SBATCH --mem=512000\n")
        lines.append("#SBATCH --mail-type=FAIL\n")
        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
        lines.append(f"#SBATCH --job-name={script_name}\n")
        lines.append(f"#SBATCH --output={script_name}.out\n")
        lines.append(f"#SBATCH --error={script_name}_err.out\n")
        lines.append("#SBATCH --export=ALL\n")
        lines.append("module load devel/miniforge\n")
        lines.append("conda activate roger-modflow\n")
        lines.append(f"cd {base_path_bwhpc}\n")
        lines.append("\n")
        lines.append('python calculate_annual_anomalies_roger.py --area %s\n' % (area))
        file_path = base_path / f"{script_name}.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)
        jobs.append(f"{script_name}.sh")

    for area in areas:
        script_name = f"calculate_annual_anomalies_{area}_modflow"
        lines = []
        lines.append("#!/bin/bash\n")
        lines.append("#SBATCH --time=48:00:00\n")
        lines.append("#SBATCH --ntasks=1\n")
        lines.append("#SBATCH --cpus-per-task=1\n")
        lines.append("#SBATCH --mem=512000\n")
        lines.append("#SBATCH --mail-type=FAIL\n")
        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
        lines.append(f"#SBATCH --job-name={script_name}\n")
        lines.append(f"#SBATCH --output={script_name}.out\n")
        lines.append(f"#SBATCH --error={script_name}_err.out\n")
        lines.append("#SBATCH --export=ALL\n")
        lines.append("module load devel/miniforge\n")
        lines.append("conda activate roger-modflow\n")
        lines.append(f"cd {base_path_bwhpc}\n")
        lines.append("\n")
        lines.append('python calculate_annual_anomalies_modflow.py --area %s\n' % (area))
        file_path = base_path / f"{script_name}.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)
        jobs.append(f"{script_name}.sh")


    file_path = base_path / "submit_annual_anomaly_jobs.sh"
    with open(file_path, "w") as job_file:
        job_file.write("#!/bin/bash\n")
        job_file.write("\n")
        for job in jobs:
                job_file.write(f"sbatch -p compute {job}\n")
    subprocess.Popen(f"chmod +x {file_path}", shell=True)


    return


if __name__ == "__main__":
    main()
