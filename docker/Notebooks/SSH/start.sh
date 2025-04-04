#!/bin/bash

/opt/conda/bin/conda init bash

python3 /opt/generate_user_environment.py --authorized-keys "$ssh_publickey"

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

/opt/conda/bin/conda init zsh
sleep infinity

