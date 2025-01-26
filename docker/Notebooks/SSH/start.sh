#!/bin/bash

#!/bin/bash

/opt/conda/bin/conda init bash
/opt/conda/bin/conda init zsh

python3 /opt/generate_user_environment.py --authorized-keys "$ssh_publickey"

exec "$@" &
/usr/bin/supervisord &

sleep 30

until [ -d "$HOME/Desktop" ]; do
  sleep 1
done

bash /etc/change_desktop_wallpaper.sh

if [ "$INSTALL_ZSH" = "1" ]; then
    /etc/install_zsh.sh
fi

if [ "$INSTALL_SLICER" = "1" ]; then
    /etc/install_slicer.sh
fi

if [ "$INSTALL_FREESURFER" = "1" ]; then
    /etc/install_freesurfer.sh
fi

if [ "$INSTALL_ITKSNAP" = "1" ]; then
    /etc/install_itksnap.sh
fi

if [ "$INSTALL_QUPATH" = "1" ]; then
    /etc/install_qupath.sh
fi

if [ ! -f "$HOME/.zshrc" ]; then
  cp /etc/.zshrc "$HOME/"
fi

if [ ! -f "$HOME/.tmux.conf" ]; then
  cp /etc/.tmux.conf "$HOME/"
fi

sleep infinity

