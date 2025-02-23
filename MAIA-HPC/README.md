# MAIA-HPC

MAIA High Performance Computing (HPC) is a collection of tools and scripts to facilitate the use of the MAIA in HPC (High Performance Computing) environments. Even though these tools are designed to work with the MAIA, they can be used to independently connect to any generic remote server accessible via SSH.

## Installation

To install the MAIA-HPC tools, clone this repository and run the `install.sh` script. This script will install the necessary dependencies and create the necessary configuration files.

```bash
install.sh
```

## Configuration

To configure the MAIA-HPC tools, run the script `configure_MAIA-HPC.sh`. This script will guide you through the configuration process.


## Connecting to a remote server

After the installation and configuration, you can connect to a remote server using the `run_session` script. This script will connect to the remote server and start a new session.

```bash
run_session.sh <server_name>
```

Since the sessions are created using tmux, you can detach from the session and later reattach using the same command used to create the session.

## Mounting a remote directory

To mount a remote directory, use the `mount_remote` script. This script will mount the remote $HOME directory to a local directory.

```bash
mount_remote.sh <server_name> <remote_directory> <local_directory>
```
If you are trying to mount the remote directory into your local machine,remember to copy the SSH configuration file to your local machine, and add the `ProxyJump` configuration to it for the remote server.

```config
Host maia
    HostName <maia_server>
    User maia-user
    Port <port>
    IdentityFile <path_to_ssh_key>

Host <server_name>
    HostName <server_ip>
    User <username>
    IdentityFile <path_to_ssh_key>
    ProxyJump maia
```

## SFTP File Transfer
To transfer files between the local machine and the remote server, use the `sftp` command. This command will open an sftp session to the remote server.

```bash
sftp <server_name>
```

## Slurm Job Configuration

To configure the Slurm job, use the `configure_experiment.sh` script. This script will guide you through the configuration process.

### Edit Slurm Job Configuration
To edit the Slurm job configuration, use the `edit_experiment.sh` script. This script will open the configuration file in the default editor.

```bash
edit_experiment.sh <experiment_name>
```

## Example Usage

The following is an example of how to submit a job to the remote server and monitor the job status.


Add the following configuration to the $HOME/ssh_config.ini file.

Note: to account for multiple experiments running on the same server, SERVER_NAME can be the SSH server name followed by a sequential number. 

For example, SERVER_NAME=maia-1, maia-2, etc.

```config
[SERVER_NAME]
EXPERIMENT_NAME=train_Iris_Dataset_Classification
REMOTE_PATH=~/Iris
LOCAL_PATH=~/.maia-hpc/examples/Iris
```

## Code and Script Folder

REMOTE_PATH and LOCAL_PATH are the folder paths on the remote server and local machine, where the experiment files (scripts, code, etc.) are stored.

DO NOT include data files in these paths! Data files should be stored in a separate folder.

As a good practice, create a separate data folder on the remote server and store the data files there.


### Data Folder
The proper data folder path should be created under the corresponding Project folder, which can be inspected by running:

```bash
ssh_to_project_dir.sh SERVER_NAME
```

### Singularity Image

Finally, create your own Singularity Image (or use an existing one) and copy it to the remote server in the corresponding Project folder.
Follow the instructions in the [Singularity](Singularity.md) section to learn more and create your own Singularity image.

For this tutorial, you can create a Singularity image from this recipe:

```python
#!/usr/bin/env python

import hpccm

Stage0 += baseimage(image='python:3.10')
Stage0 += gnu()
Stage0 += python(pip_packages=['numpy', 'scikit-learn'])
Stage0 += runscript(commands=['python $ROOT_DIR/iris_dataset.py'])
```
and then build the Singularity image with the following commands:

```bash
hpccm --recipe recipe.py --format singularity > iris_dataset.def
singularity build iris_dataset.sif iris_dataset.def
```


To properly work, the Singularity image should be visible when running the following command:

```bash
ssh_to_project_dir.sh SERVER_NAME
```
### Data Transfer

To transfer the data files (and optionally the Singularity Image) to the remote server, use the `sftp` command. This command will open an sftp session to the remote server.

```bash
sftp SERVER_NAME:$PROJ_DIR
```

### Experiment Configuration

Finally, you can create the experiment configuration file with  `train_Iris_Dataset_Classification` name and the command to run the experiment:

```json
{
    "singularity_image": "YOUR_SINGULARITY_IMAGE",
    "command": "python $ROOT_DIR/iris_dataset.py"
}
```

ROOT_DIR  should be added as an environment variable in the experiment and it should match the REMOTE_PATH in the ssh_config.ini file:

```bash
configure_experiment.sh
```
and then edit the experiment configuration file:

```json
{
    "env_variables": {
        "ROOT_DIR": "<REMOTE_PATH>"
    }
}
```

## Submit Job

To submit the job, run the following command.

```bash
submit_job.sh SERVER_NAME
```

To monitor the job status, run the following command.

```bash
log_out.sh SERVER_NAME
log_err.sh SERVER_NAME
```


