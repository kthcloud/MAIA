#!/bin/bash

# Usage: ./script_name.sh <remote_name> <local_folder>

# Exit if any command fails
set -e

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <remote_server> <remote_folder> <local_folder>"
    exit 1
fi

REMOTE_NAME=$1
REMOTE_FOLDER=$2
LOCAL_FOLDER=$3

# Ensure the local folder exists
sudo mkdir -p "$LOCAL_FOLDER"
sudo chmod -R 777 "$LOCAL_FOLDER"

# Mount the remote folder using sshfs
sshfs -o allow_other,default_permissions,uid=$(id -u),gid=$(id -g) "${REMOTE_NAME}:${REMOTE_FOLDER}" "$LOCAL_FOLDER"