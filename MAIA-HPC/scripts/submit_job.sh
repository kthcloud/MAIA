#!/bin/bash


if [ -z "$1" ]; then
  echo "Usage: $0 SERVER_NAME"
  exit 1
fi

# Read the argument and set it as an environment variable
export SERVER_NAME=$1

# Read variables from the output of the script
source read_config.sh $SERVER_NAME



# Strip out the suffix if SERVER_NAME ends with -NUMBER, where NUMBER can be anything
SERVER_NAME=$(echo $SERVER_NAME | sed 's/-.*$//')
echo "Server Name: $SERVER_NAME"
echo "Experiment Name: $EXPERIMENT_NAME"
echo "Remote Path: $REMOTE_PATH"
echo "Local Path: $LOCAL_PATH"

mkdir -p $HOME/.maia-hpc/job_scripts
python /usr/local/bin/create_job_script.py \
--server-config-file ~/.maia-hpc/server_configs/$SERVER_NAME.json \
--job-config-file ~/.maia-hpc/experiments/$EXPERIMENT_NAME.json --script-file $HOME/.maia-hpc/job_scripts/$EXPERIMENT_NAME.sh

ssh $SERVER_NAME mkdir -p $REMOTE_PATH
rsync $HOME/.maia-hpc/job_scripts/* $SERVER_NAME:$REMOTE_PATH'/scripts/' --progress -r
cd $(eval echo $LOCAL_PATH) && rsync . $SERVER_NAME:$REMOTE_PATH --progress -r
export JOB_ID=$(ssh $SERVER_NAME sbatch $REMOTE_PATH'/scripts/'$EXPERIMENT_NAME.sh | rev | cut -d ' '  -f1 | rev)

echo "Job ID: $JOB_ID"