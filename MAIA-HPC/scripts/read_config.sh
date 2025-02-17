#!/bin/bash

# Function to read all key-value pairs for a specific entry
read_config_section() {
    local config_file="$HOME/ssh_config.ini"
    local section="$1"

    # Extract all key-value pairs for the specified section
    awk -v section="$section" '
        /^\[.*\]/ { in_section = 0 } 
        $0 == "[" section "]" { in_section = 1; next }
        in_section && /^[a-zA-Z0-9_]+=.+/ { print $0 }
    ' "$config_file"
}

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <section_name>"
    exit 1
fi

# Read the section name from the argument
entry="$1"

# Get all key-value pairs for the section
config_entries=$(read_config_section "$entry")

if [ -n "$config_entries" ]; then
    echo "Configuration entries for [$entry]:"
    echo "$config_entries"
    
    # Set the entries as environment variables
    while IFS='=' read -r key value; do
        export "$key=$value"
    done <<< "$config_entries"
else
    echo "Section '$entry' not found in the configuration file."
fi
