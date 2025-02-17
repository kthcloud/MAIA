#!/bin/bash

# Make the scripts executable
chmod +x scripts/*
# Install the scripts in the /usr/local/bin directory

cp scripts/* /usr/local/bin/

# Create MAIA-HPC directory
mkdir -p ~/.maia-hpc
mkdir -p ~/.maia-hpc/experiments
mkdir -p ~/.maia-hpc/server_configs
mkdir -p ~/.maia-hpc/examples

cp -r Examples/* ~/.maia-hpc/examples
