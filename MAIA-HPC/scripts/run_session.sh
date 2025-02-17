#!/bin/bash

# Check if a session name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <session_name>"
    exit 1
fi

# Define the session name
SESSION_NAME="$1"

# Check if the session already exists
if tmux has-session -t $SESSION_NAME 2>/dev/null; then
    echo "Attaching to existing tmux session '$SESSION_NAME'."
    tmux attach-session -t $SESSION_NAME
else
    echo "Creating a new tmux session named '$SESSION_NAME'."
    tmux new-session -s $SESSION_NAME \; send-keys "echo 'Welcome to $SESSION_NAME!'" C-m \; send-keys "ssh $SESSION_NAME" C-m
fi
