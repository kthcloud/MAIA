#!/bin/bash -e

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

trap "echo TRAPed signal" HUP INT QUIT TERM

sudo chown maia-user:maia-user /home/maia-user || sudo chown maia-user:maia-user /home/maia-user/* || { echo "Failed to change maia-user directory permissions. There may be permission issues."; }
# Change operating system password to environment variable
if [ ! -d /home/maia-user/Tutorials ]; then
  sudo cp -r /etc/Tutorials /home/maia-user
  sudo chmod -R 777 /home/maia-user/Tutorials
fi

if [ ! -f /home/maia-user/Welcome.ipynb ]; then
  sudo cp /etc/Welcome.ipynb /home/maia-user
  sudo chmod 777 /home/maia-user/Welcome.ipynb
fi


#if [ ! -d /home/maia-user/Shared ]; then
#    ln -s /mnt/shared /home/maia-user/Shared
#fi
touch /home/maia-user/.env
JUPYTERHUB_POD_NAME=jupyter-$(echo "$JUPYTERHUB_USER" sed -r 's/[-]+/-2d/g' | sed -r 's/[@]+/-40/g' | sed -r 's/[.]+/-2e/g')
if grep -Fx "JUPYTERHUB_POD_NAME=${JUPYTERHUB_POD_NAME}" /home/maia-user/.env;
then
    echo "Environment variable already set"
else
  echo "JUPYTERHUB_POD_NAME=${JUPYTERHUB_POD_NAME}" >> /home/maia-user/.env
fi

echo "maia-user:$PASSWD" | sudo chpasswd

if [ ! -f /home/maia-user/.bash_profile ]; then
  cp /etc/.bash_profile /home/maia-user/
fi
cp /etc/MAIA.png /home/maia-user/
echo "Session Running. Press [Return] to exit."
read


if [ "$INSTALL_ZSH" = "1" ]; then
    /etc/install_zsh.sh
fi

#if [ ! -f "$HOME/.zshrc" ]; then
  cp /etc/.zshrc "$HOME/"
#fi

#if [ ! -f "$HOME/.tmux.conf" ]; then
  cp /etc/.tmux.conf "$HOME/"
#fi
# Only for debugging and development
#exec "$@"
