FROM ghcr.io/selkies-project/nvidia-egl-desktop:22.04-20240225071130


USER root
RUN apt update && apt install nginx -y

COPY supervisord.conf   /etc/supervisord.conf
RUN chmod 755 /etc/supervisord.conf

COPY ui.js /opt/noVNC/app/ui.js
RUN usermod --login maia-user --move-home --home /home/maia-user user && groupmod --new-name maia-user user

RUN cd /etc && wget https://download.slicer.org/bitstream/66653094c8a295ea3286e97a -O slicer.tar.gz

COPY entrypoint.sh /etc/entrypoint.sh
RUN chmod 755 /etc/entrypoint.sh
RUN echo "maia-user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
COPY Tutorials /etc/Tutorials
COPY Welcome.ipynb /etc/Welcome.ipynb
RUN chmod -R 777 /etc/Tutorials
RUN chmod 777 /etc/Welcome.ipynb

COPY change_desktop_wallpaper.sh /etc/change_desktop_wallpaper.sh
RUN chmod 777 /etc/change_desktop_wallpaper.sh
RUN chmod a+x /etc/change_desktop_wallpaper.sh

COPY 70-maia /etc/update-motd.d/70-maia
RUN chmod +x /etc/update-motd.d/70-maia && chmod 777 /etc/update-motd.d/70-maia

COPY Slicer.desktop /etc/Slicer.desktop
RUN chmod 777 /etc/Slicer.desktop
RUN chmod a+x /etc/Slicer.desktop

USER 1000
ENV SHELL /bin/bash

COPY default.template /etc/default.template
ENV USER maia-user
WORKDIR /home/maia-user
COPY .bash_profile /etc/.bash_profile

COPY MAIA.png /etc/MAIA.png

COPY bitmap.png /etc/bitmap.png

COPY 300px-3DSlicerLogo-app-icon512.svg.png /etc/300px-3DSlicerLogo-app-icon512.svg.png

ENTRYPOINT ["bash", "/etc/entrypoint.sh"]

