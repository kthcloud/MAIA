#!/bin/bash

## Check if an experiment name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_file>"
    exit 1
fi

filename=$(basename "$1")
cp "$1" "$HOME/.maia-hpc/experiments/$filename"