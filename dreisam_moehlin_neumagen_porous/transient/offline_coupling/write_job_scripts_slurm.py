from pathlib import Path
import subprocess
import numpy as np
import os
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    dir_name = os.path.basename(str(Path(__file__).parent))
    base_path_bwhpc_roger = f"/pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed"
    base_path_bwhpc_modflow = f"/pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/{dir_name}"
    base_path_ws_modflow = Path(f"/pfs/10/work/fr_rs1092-workspace/roger-modflow/dreisam_moehlin_neumagen_porous/transient/{dir_name}")

    # identifiers of the simulations
    stress_tests_meteo = ["base", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]
    stress_test_meteo_magnitudes = [0, 2]
    stress_test_meteo_durations = [0, 3]
    scenario_flags = []
    script_names_modflow = []
    for stress_test_meteo in stress_tests_meteo:
        if stress_test_meteo == "base":
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --irrigation irrigation' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --yellow-mustard yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --irrigation irrigation --yellow-mustard yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --irrigation irrigation --yellow-mustard yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --grain-corn-only grain-corn-only' % (stress_test_meteo))

            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_no-yellow-mustard_no-soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_yellow-mustard_soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_irrigation_yellow-mustard_no-soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_irrigation_yellow-mustard_soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only' % (stress_test_meteo))

        elif stress_test_meteo == "base_2000-2024":
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --irrigation irrigation' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --yellow-mustard yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --irrigation irrigation --yellow-mustard yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --irrigation irrigation --yellow-mustard yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --grain-corn-only grain-corn-only' % (stress_test_meteo))

            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_no-yellow-mustard_no-soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_yellow-mustard_soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_irrigation_yellow-mustard_no-soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_irrigation_yellow-mustard_soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only' % (stress_test_meteo))

        elif stress_test_meteo == "spring-summer-wet":
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --yellow-mustard yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --yellow-mustard yellow-mustard' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --grain-corn-only grain-corn-only' % (stress_test_meteo))

            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_no-yellow-mustard_no-soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_irrigation_no-yellow-mustard_soil-compaction' % (stress_test_meteo))
            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_yellow-mustard_no-soil-compaction' % (stress_test_meteo)) 
            script_names_modflow.append('modflow_%s-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only' % (stress_test_meteo))

        elif stress_test_meteo in ["spring-drought", "summer-drought", "spring-summer-drought"]:
            for magnitude in stress_test_meteo_magnitudes:
                for duration in stress_test_meteo_durations:
                    if magnitude == 0 and duration == 0:
                        pass
                    else:
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction soil-compaction' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction soil-compaction --irrigation irrigation' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction soil-compaction --yellow-mustard yellow-mustard' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --irrigation irrigation --yellow-mustard yellow-mustard' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction soil-compaction --irrigation irrigation --yellow-mustard yellow-mustard' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction soil-compaction --grain-corn-only grain-corn-only' % (stress_test_meteo, magnitude, duration))

                        script_names_modflow.append('modflow_%s_magnitude%s_duration%s_no-irrigation_no-yellow-mustard_soil-compaction' % (stress_test_meteo, magnitude, duration))
                        script_names_modflow.append('modflow_%s_magnitude%s_duration%s_no-irrigation_no-yellow-mustard_no-soil-compaction' % (stress_test_meteo, magnitude, duration))
                        script_names_modflow.append('modflow_%s_magnitude%s_duration%s_irrigation_no-yellow-mustard_soil-compaction' % (stress_test_meteo, magnitude, duration))
                        script_names_modflow.append('modflow_%s_magnitude%s_duration%s_no-irrigation_yellow-mustard_soil-compaction' % (stress_test_meteo, magnitude, duration))
                        script_names_modflow.append('modflow_%s_magnitude%s_duration%s_irrigation_yellow-mustard_no-soil-compaction' % (stress_test_meteo, magnitude, duration))
                        script_names_modflow.append('modflow_%s_magnitude%s_duration%s_no-irrigation_yellow-mustard_no-soil-compaction' % (stress_test_meteo, magnitude, duration))
                        script_names_modflow.append('modflow_%s_magnitude%s_duration%s_no-irrigation_no-yellow-mustard_soil-compaction_grain-corn-only' % (stress_test_meteo, magnitude, duration))
    jobs = []
    years = np.arange(2013, 2024).tolist()
    # xx = [0, 1, 2, 3, 4, 5, 6, 70, 71, 72, 73, 74, 75, 76, 21, 22, 23, 24, 25, 26, 27, 42, 43, 44, 45, 46, 47, 48, 77, 78, 79, 80, 81]
    # scenario_flags_ = [scenario_flags[i] for i in xx]
    # script_names_modflow_ = [script_names_modflow[i] for i in xx]
    # for scenario_flag, script_name in zip(scenario_flags_, script_names_modflow_):
    for scenario_flag, script_name in zip(scenario_flags, script_names_modflow):
        output_path_ws = base_path_ws_modflow / "output" / script_name
        lines = []
        lines.append("#!/bin/bash\n")
        lines.append("#SBATCH --time=12:00:00\n")
        lines.append("#SBATCH --ntasks=1\n")
        lines.append("#SBATCH --cpus-per-task=1\n")
        lines.append("#SBATCH --mem=32000\n")
        lines.append("#SBATCH --mail-type=FAIL\n")
        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
        lines.append(f"#SBATCH --job-name={script_name}\n")
        lines.append(f"#SBATCH --output={script_name}.out\n")
        lines.append(f"#SBATCH --error={script_name}_err.out\n")
        lines.append("#SBATCH --export=ALL\n")
        lines.append("module load devel/miniforge\n")
        lines.append("conda activate roger-modflow\n")
        lines.append(f"cd {base_path_bwhpc_modflow}\n")
        lines.append("\n")
        lines.append("mkdir ${TMPDIR}/roger-modflow\n")
        lines.append("mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous\n")
        lines.append("mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient\n")
        lines.append("mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling\n")
        lines.append("mkdir ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output\n")
        lines.append("cp -r /pfs/10/work/fr_rs1092-workspace/roger-modflow/bin ${TMPDIR}/roger-modflow\n")
        lines.append("cp -r %s/roger-oneD_modflow6_transient_with_well_extraction.py ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling\n" % (base_path_bwhpc_modflow)) 
        lines.append("cp -r %s/config_modflow.yml ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient\n" % (str(base_path_ws_modflow.parent)))
        lines.append("cp -r %s/input ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient\n" % (str(base_path_ws_modflow.parent)))
        for year in years:
            input_file = script_name.replace("modflow_", "recharge_") + f"_year{year}.nc"
            lines.append("cp -r %s/output/%s ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/input/\n" % (base_path_bwhpc_roger, input_file))
        for year in years:
            input_file = script_name.replace("modflow_", "capillary_rise_") + f"_year{year}.nc"
            lines.append("cp -r %s/output/%s ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/input/\n" % (base_path_bwhpc_roger, input_file))
        for year in years:
            if "_irrigation_" in script_name:
                input_file = script_name.replace("modflow_", "irrigation_") + f"_year{year}.nc"
                lines.append("cp -r %s/output/%s ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/input/\n" % (base_path_bwhpc_roger, input_file))
        lines.append('sleep 120\n')
        lines.append("cd ${TMPDIR}/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling\n")
        lines.append('python roger-oneD_modflow6_transient_with_well_extraction.py %s\n' % (scenario_flag))
        lines.append("# Move output from local SSD to global workspace\n")
        lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
        lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
        lines.append('mv -v "${TMPDIR}"/roger-modflow/dreisam_moehlin_neumagen_porous/transient/offline_coupling/output/%s %s' % (script_name, output_path_ws.as_posix()))
        file_path = base_path / f"{script_name}.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)
        jobs.append(f"{script_name}.sh")

    file_path = base_path / "submit_jobs.sh"
    with open(file_path, "w") as job_file:
        job_file.write("#!/bin/bash\n")
        job_file.write("\n")
        for job in jobs:
            job_file.write(f"sbatch -p compute {job}\n")
    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()
