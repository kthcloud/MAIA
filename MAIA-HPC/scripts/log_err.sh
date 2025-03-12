#!/bin/bash


if [ -z "$1" ]; then
  echo "Usage: $0 SERVER_NAME JOB_NAME"
  exit 1
fi

export SSH_SERVER=$1


source read_config.sh $SSH_SERVER


# Strip out everything after the first hyphen
SSH_SERVER=$(echo $SSH_SERVER | sed 's/-.*$//')
echo "Server Name: $SSH_SERVER"
echo "Experiment Name: $EXPERIMENT_NAME"

export JOB_NAME=$(echo  $EXPERIMENT_NAME | tr '[:upper:]' '[:lower:]' | tr '_' '-')

while true; do
  export JOB_ID=$(ssh $SSH_SERVER "squeue -u \$USER --json"| jq -r --arg JOB_NAME "$JOB_NAME" '.jobs[] | select(.name == $JOB_NAME) | .job_id' | tail -n 1)
  echo "JOB_ID: $JOB_ID"
  
  if ssh $SSH_SERVER "[ -f \$HOME/logs/slurm-$JOB_ID.err ]"; then
  
    echo "JOB IS RUNNING"
    ssh $SSH_SERVER "tail -f \$HOME/logs/slurm-$JOB_ID.err"
  else
    export JOB_ID=$(ssh $SSH_SERVER "sacct -u \$USER --format=JobID,nodelist,Partition,AllocCPUs,State,start --json" | jq -r --arg JOB_NAME "$JOB_NAME" '.jobs[] | select(.name == $JOB_NAME) | .job_id' | tail -n 1 )

    JOB_STATE=$(ssh $SSH_SERVER "squeue -j $JOB_ID -h -o %T")
    echo "JOB_STATE: $JOB_STATE"
    if [ "$JOB_STATE" == "PENDING" ]; then
      echo "PENDING JOB_ID: $JOB_ID"
      exit 0
    else
      echo "TERMINATED JOB_ID: $JOB_ID"
      ssh $SSH_SERVER "tail -f \$HOME/logs/slurm-$JOB_ID.err"
      exit 0
    fi
  fi

done
