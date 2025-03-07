!#!/bin/bash

# Request the user to provide the server name
echo "Please provide the server name: "
read SERVER_NAME
# Request the user to provide the Job Name
echo "Please provide the job name: "
read JOB_NAME

# Request the user to provide the wall time, default is 12 hours

echo "Please provide the wall time (default is 12 hours): "
read WALL_TIME
WALL_TIME=${WALL_TIME:-"12:00:00"}

# Request the user to provide the number of nodes, default is 1
echo "Please provide the number of nodes (default is 1): "
read NODES
NODES=${NODES:-1}

# Request the user to provide the number of cpus per task, default is 8
echo "Please provide the number of cpus per task (default is 8): "
read CPUS_PER_TASK
CPUS_PER_TASK=${CPUS_PER_TASK:-8}

# Request the user to provide the number of gpus per node, default is 1
echo "Please provide the number of gpus per node (default is 1): "
read GPUS_PER_NODE
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

# Request the user to provide the number of tasks per node, default is 1
echo "Please provide the number of tasks per node, equal to the number of gpus per node (default is 1): "
read TASKS_PER_NODE
TASKS_PER_NODE=${TASKS_PER_NODE:-1}

# Request the user to provide the Singularity image
echo "Please provide the Singularity image: "
read SINGULARITY_IMAGE

# Request the user to provide the command to run from the Singularity image
echo "Please provide the command to run from the Singularity image: "
read COMMAND



## Create the job configuration file
cat > ~/.maia-hpc/experiments/$JOB_NAME.json <<EOF
{
    "job_name": "$JOB_NAME",
    "wall_time": "$WALL_TIME",
    "nodes": $NODES,
    "$SERVER_NAME": {
        "cpus_per_task": $CPUS_PER_TASK,
        "gpus_per_node": $GPUS_PER_NODE,
        "tasks_per_node": $TASKS_PER_NODE,
        "env_variables": {
        
    }
    },
    
    "project_dir": "/mnt/proj-dir",
    "singularity_image": "$SINGULARITY_IMAGE",
    "command": "$COMMAND"
}