#!/bin/bash

# Get the server name
export SERVER_NAME=$1

# Get the Project Directory from the JSON file
export PROJECT_DIR=$(jq -r ".project_dir" ~/.maia-hpc/server_configs/$SERVER_NAME.json)

echo "SSHing to $SERVER_NAME at $PROJECT_DIR"
# SSH to the project directory
ssh $SERVER_NAME "cd $PROJECT_DIR; ls -l;"
