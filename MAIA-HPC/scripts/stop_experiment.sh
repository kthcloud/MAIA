#!/bin/bash


if [ -z "$1" ]; then
  echo "Usage: $0 SERVER_NAME JOB_NAME"
  exit 1
fi

export SSH_SERVER=$1


source read_config.sh $SSH_SERVER


# Strip out the number if SERVER_NAME ends with -NUMBER
SSH_SERVER=$(echo $SSH_SERVER | sed 's/-[0-9]*$//')
echo "Server Name: $SSH_SERVER"
echo "Experiment Name: $EXPERIMENT_NAME"

export JOB_NAME=$(echo  $EXPERIMENT_NAME | tr '[:upper:]' '[:lower:]' | tr '_' '-')


export JOB_ID=$(ssh $SSH_SERVER "squeue -u \$USER --json"| jq -r --arg JOB_NAME "$JOB_NAME" '.jobs[] | select(.name == $JOB_NAME) | .job_id' | tail -n 1)
echo "JOB_ID: $JOB_ID"
  
if ssh $SSH_SERVER "[ -f \$HOME/logs/slurm-$JOB_ID.out ]"; then
  
  echo "JOB IS RUNNING"
  ssh $SSH_SERVER scancel $JOB_ID
  fi
echo "Job $JOB_ID has been cancelled."
echo "To check the status of the job, run the following command:"
echo "        watch 'squeue --format="%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" -u $USER'"
