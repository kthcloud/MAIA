#!/bin/bash

/opt/conda/bin/conda init bash
/opt/conda/bin/conda init zsh

exec "$@" &
/usr/bin/supervisord &


if [ "$INSTALL_ZSH" = "1" ]; then
    /etc/install_zsh.sh
fi

#if [ ! -f "$HOME/.zshrc" ]; then
  cp /etc/.zshrc "$HOME/"
#fi

#if [ ! -f "$HOME/.tmux.conf" ]; then
  cp /etc/.tmux.conf "$HOME/"
#fi

sleep infinity