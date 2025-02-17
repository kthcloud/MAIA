import json
import argparse
from pathlib import Path

def main(server_config_file, job_config_file,script_file):
    # Read the configuration file
    with open(server_config_file, "r") as f:
        server_config = json.load(f)

    server_name = Path(server_config_file).name.split(".")[0]

    with open(job_config_file, "r") as f:
        job_config = json.load(f)

    job_name = job_config["job_name"]
    project_id = server_config["project_id"]
    wall_time = job_config["wall_time"]
    nodes = job_config["nodes"]
    error_file = server_config["error_file"]
    output_file = server_config["output_file"]
    # Create the header of the job script
    header = f"""#!/bin/bash

#SBATCH -J {job_name}
#SBATCH -A {project_id}
#SBATCH -t {wall_time}
#SBATCH --nodes={nodes}
#SBATCH -e {error_file}%j.err
#SBATCH -o {output_file}%j.out

"""
    
    if server_config["partition"] is not None:
        header += f"#SBATCH -p {server_config['partition']}\n"
    
    if "cpus_per_task" in job_config[server_name]:
        header += f"#SBATCH -c {job_config[server_name]['cpus_per_task']}\n"

    if "gpus_per_node" in job_config[server_name]:
        header += f"#SBATCH --gpus-per-node={job_config[server_name]['gpus_per_node']}\n"

    if "tasks_per_node" in job_config[server_name]:
        header += f"#SBATCH --ntasks-per-node={job_config[server_name]['tasks_per_node']}\n"

    if "gpus" in job_config[server_name]:
        header += f"#SBATCH --gres=gpu:{job_config[server_name]['gpus']}\n"
        header += f"#SBATCH --gpus={job_config[server_name]['gpus']}\n"

    if "constraint" in job_config[server_name]:
        header += f"#SBATCH -C {job_config[server_name]['constraint']}\n"

    if "no_gpu" in job_config[server_name]:
        header += f"#SBATCH -C NOGPU\n"

    

    # Create the body of the job script

    if job_config[server_name]["env_variables"] is not None:
        for key, value in job_config[server_name]["env_variables"].items():
            header += f"export {key}={value}\n"

    if "load_modules" in job_config[server_name]:
        for module in job_config[server_name]["load_modules"]:
            header += f"ml {module}\n"

    cmd = "srun singularity run"

    if "nvidia_gpu" in server_config:
        cmd += f" --nv"

    if "amd_gpu" in server_config:
        cmd += f" --rocm"

    if job_config["project_dir"] is not None:
        cmd += f" -B {server_config['project_dir']}:{job_config['project_dir']}"
    
    cmd += f" {server_config['project_dir']}/{job_config['singularity_image']}"

    cmd += f" {job_config['command']}"

    with open(script_file, "w") as f:
        f.write(header + cmd)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-config-file", type=str, required=True)
    parser.add_argument("--job-config-file", type=str, required=True)
    parser.add_argument("--script-file", type=str, required=True)
    main(**vars(parser.parse_args()))
