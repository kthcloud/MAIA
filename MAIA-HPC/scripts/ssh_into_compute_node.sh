#!/bin/bash


if [ -z "$1" ]; then
  echo "Usage: $0 SERVER_NAME JOB_NAME"
  exit 1
fi

export SSH_SERVER=$1


source read_config.sh $SSH_SERVER


# Strip out the suffix if SERVER_NAME ends with -NUMBER, where NUMBER can be anything
SSH_SERVER=$(echo $SSH_SERVER | sed 's/-.*$//')
echo "Server Name: $SSH_SERVER"
echo "Experiment Name: $EXPERIMENT_NAME"

export JOB_NAME=$(echo  $EXPERIMENT_NAME | tr '[:upper:]' '[:lower:]' | tr '_' '-')


export JOB_ID=$(ssh $SSH_SERVER "squeue -u \$USER --json" | jq -r --arg JOB_NAME "$JOB_NAME" '.jobs[] | select(.name == $JOB_NAME) | .job_id' | tail -n 1)
echo "JOB_ID: $JOB_ID"

export RUNNING_NODE=$(ssh $SSH_SERVER "squeue -u \$USER --json" | jq -r --arg JOB_ID "$JOB_ID" '.jobs[] | select(.job_id == ($JOB_ID | tonumber)) | .nodes')
echo "RUNNING_NODE: $RUNNING_NODE"
  
if ssh $SSH_SERVER "[ -f \$HOME/logs/slurm-$JOB_ID.out ]"; then
  
echo "JOB IS RUNNING"

echo "Run the following command to ssh into the Login node:"
echo "      ssh $SSH_SERVER"
echo "and then, from the Login node:"
echo "      ssh $RUNNING_NODE"

server_project_dir=$(jq -r '.project_dir' $HOME/.maia-hpc/server_configs/$SSH_SERVER.json)
singularity_project_dir=$(jq -r '.project_dir' $HOME/.maia-hpc/experiments/$EXPERIMENT_NAME.json)

singularity_image=$(jq -r '.singularity_image' $HOME/.maia-hpc/experiments/$EXPERIMENT_NAME.json)

echo "To access the Singularity image environment, run the following command:"
echo "      singularity run -B $server_project_dir:$singularity_project_dir $SINGULARITY_IMAGE $server_project_dir/$singularity_image bash"

fi
