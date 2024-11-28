#!/bin/bash -e

# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

trap "echo TRAPed signal" HUP INT QUIT TERM

# Create and modify permissions of XDG_RUNTIME_DIR
sudo -u maia-user mkdir -pm700 /tmp/runtime-user
sudo chown maia-user:maia-user /tmp/runtime-user
sudo -u maia-user chmod 700 /tmp/runtime-user
# Make user directory owned by the user in case it is not
sudo chown maia-user:maia-user /home/maia-user || sudo chown maia-user:maia-user /home/maia-user/* || { echo "Failed to change maia-user directory permissions. There may be permission issues."; }
# Change operating system password to environment variable
sudo cp -r /etc/Tutorials /home/maia-user
sudo cp /etc/Welcome.ipynb /home/maia-user
sudo chmod -R 777 /home/maia-user/Tutorials
sudo chmod 777 /home/maia-user/Welcome.ipynb
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
# Remove directories to make sure the desktop environment starts
sudo rm -rf /tmp/.X* ~/.cache
# Change time zone from environment variable
sudo ln -snf "/usr/share/zoneinfo/$TZ" /etc/localtime && echo "$TZ" | sudo tee /etc/timezone > /dev/null
# Add Lutris and VirtualGL directories to path
export PATH="${PATH}:/usr/local/games:/usr/games:/opt/VirtualGL/bin"
# Add LibreOffice to library path
export LD_LIBRARY_PATH="/usr/lib/libreoffice/program:${LD_LIBRARY_PATH}"

# Start DBus without systemd
sudo /etc/init.d/dbus start

# Default display is :0 across the container
export DISPLAY=":0"
# Run Xvfb server with required extensions
/usr/bin/Xvfb "${DISPLAY}" -ac -screen "0" "8192x4096x${CDEPTH}" -dpi "${DPI}" +extension "COMPOSITE" +extension "DAMAGE" +extension "GLX" +extension "RANDR" +extension "RENDER" +extension "MIT-SHM" +extension "XFIXES" +extension "XTEST" +iglx +render -nolisten "tcp" -noreset -shmem &

# Wait for X11 to start
echo "Waiting for X socket"
until [ -S "/tmp/.X11-unix/X${DISPLAY/:/}" ]; do sleep 1; done
echo "X socket is ready"

# Resize the screen to the provided size
bash -c ". /opt/gstreamer/gst-env && /usr/local/bin/selkies-gstreamer-resize ${SIZEW}x${SIZEH}"

# Run the x11vnc + noVNC fallback web interface if enabled
if [ "${NOVNC_ENABLE,,}" = "true" ]; then
  if [ -n "$NOVNC_VIEWPASS" ]; then export NOVNC_VIEWONLY="-viewpasswd ${NOVNC_VIEWPASS}"; else unset NOVNC_VIEWONLY; fi
  /usr/local/bin/x11vnc -display "${DISPLAY}" -passwd "${BASIC_AUTH_PASSWORD:-$PASSWD}" -shared -forever -repeat -xkb -snapfb -threads -xrandr "resize" -rfbport 5900 ${NOVNC_VIEWONLY} &
  /opt/noVNC/utils/novnc_proxy --vnc localhost:5900 --listen 8082 --heartbeat 10 &
fi

# Use VirtualGL to run the KDE desktop environment with OpenGL if the GPU is available, otherwise use OpenGL with llvmpipe
if [ -n "$(nvidia-smi --query-gpu=uuid --format=csv | sed -n 2p)" ]; then
  export VGL_DISPLAY="${VGL_DISPLAY:-egl}"
  export VGL_REFRESHRATE="$REFRESH"
  /usr/bin/vglrun +wm /usr/bin/dbus-launch /usr/bin/startplasma-x11 &
else
  /usr/bin/dbus-launch /usr/bin/startplasma-x11 &
fi

# Start Fcitx input method framework
/usr/bin/fcitx &

# Add custom processes right below this line, or within `supervisord.conf` to perform service management similar to systemd


sudo sed -i 's|worker_processes .*|worker_processes 1;|' /etc/nginx/nginx.conf

envsubst '${JUPYTERHUB_USER},${NAMESPACE}' < /etc/default.template > default
sudo mv default /etc/nginx/sites-enabled/
sudo nginx -c /etc/nginx/nginx.conf -g 'daemon off;' &




cp /etc/.bash_profile /home/maia-user/
cp /etc/MAIA.png /home/maia-user/
echo "Session Running. Press [Return] to exit."
read

# Only for debugging and development
#exec "$@"
