#!/bin/bash

# Request user input to provide the HPC server name
echo "Please provide the HPC server name: "
read HPC_SERVER_NAME
# Request user input to provide the HPC username
echo "Please provide the HPC username: "
read HPC_USERNAME
# Request user input to provide the HPC server address
echo "Please provide the HPC server address: "
read HPC_SERVER_ADDRESS
# Request user input to provide the HPC server port and assign 22 as the default value
echo "Please provide the HPC server port (default: 22): "
read HPC_SERVER_PORT
# Assign 22 as the default value for the HPC server port if no input is provided
HPC_SERVER_PORT=${HPC_SERVER_PORT:-22}
# Request user input to provide the path to the SSH key file
echo "Please provide the path to the SSH key file: "
read SSH_KEY_PATH
# Add the provided HPC server information to the SSH configuration file
echo "Host $HPC_SERVER_NAME" >> ~/.ssh/config
echo "    HostName $HPC_SERVER_ADDRESS" >> ~/.ssh/config
echo "    User $HPC_USERNAME" >> ~/.ssh/config
echo "    Port $HPC_SERVER_PORT" >> ~/.ssh/config
if [ -n "$SSH_KEY_PATH" ]; then
    echo "    IdentityFile $SSH_KEY_PATH" >> ~/.ssh/config
fi
# Add Option to persist the SSH ControlMaster configuration
echo "    ControlMaster auto" >> ~/.ssh/config
echo "    ControlPath ~/.ssh/%r@%h-%p" >> ~/.ssh/config
echo "    ControlPersist 240h" >> ~/.ssh/config
# Display a message to inform the user that the configuration has been completed
echo "HPC server configuration completed successfully."

## Create the JSON file for the server configuration

# Ask the user if they want to create a JSON configuration file
echo "Do you want to create a JSON configuration file for the server? (yes/no): "
read CREATE_JSON

if [ "$CREATE_JSON" == "yes" ]; then
    # Request user input to provide the HPC Project ID
    echo "Please provide the HPC Project ID: "
    read PROJECT_ID
    # Request user input to provide the HPC Partition
    echo "Please provide the HPC Partition: "
    read PARTITION
    # Request user input to provide the path to the error file, default value is "logs/slurm-" if no input is provided
    echo "Please provide the path to the error file (default: logs/slurm-): "
    read ERROR_FILE
    ERROR_FILE=${ERROR_FILE:-logs/slurm-}
    # Request user input to provide the path to the output file, default value is "logs/slurm-" if no input is provided
    echo "Please provide the path to the output file (default: logs/slurm-): "
    read OUTPUT_FILE
    OUTPUT_FILE=${OUTPUT_FILE:-logs/slurm-}
    # Request user input to specify if the HPC server has NVIDIA GPUs
    echo "Does the HPC server have NVIDIA GPUs? (true/false): "
    read NVIDIA_GPU
    # Request user input to provide the path to the project directory
    echo "Please provide the path to the project directory: "
    read PROJECT_DIR
    # Create a JSON file with the provided HPC server configuration
    mkdir -p ~/.maia-hpc/server_configs
    echo "{" > ~/.maia-hpc/server_configs/$HPC_SERVER_NAME.json
    echo "    \"project_id\": \"$PROJECT_ID\"," >> ~/.maia-hpc/server_configs/$HPC_SERVER_NAME.json
    echo "    \"partition\": \"$PARTITION\"," >> ~/.maia-hpc/server_configs/$HPC_SERVER_NAME.json
    echo "    \"error_file\": \"$ERROR_FILE\"," >> ~/.maia-hpc/server_configs/$HPC_SERVER_NAME.json
    echo "    \"output_file\": \"$OUTPUT_FILE\"," >> ~/.maia-hpc/server_configs/$HPC_SERVER_NAME.json
    echo "    \"nvidia_gpu\": $NVIDIA_GPU," >> ~/.maia-hpc/server_configs/$HPC_SERVER_NAME.json
    echo "    \"project_dir\": \"$PROJECT_DIR\"" >> ~/.maia-hpc/server_configs/$HPC_SERVER_NAME.json
    echo "}" >> ~/.maia-hpc/server_configs/$HPC_SERVER_NAME.json
    # Display a message to inform the user that the server configuration file has been created
    echo "Server configuration file created successfully."
else
    echo "Skipping JSON configuration file creation."
fi

