#FROM ghcr.io/selkies-project/nvidia-egl-desktop:22.04-20241222100454
FROM ghcr.io/selkies-project/nvidia-egl-desktop:22.04-20240225071130


USER root
RUN apt update && apt install nginx -y
RUN apt update && apt install -y zsh

COPY supervisord.conf   /etc/supervisord.conf
RUN chmod 755 /etc/supervisord.conf
COPY entrypoint.sh /etc/entrypoint.sh
RUN chmod 755 /etc/entrypoint.sh
COPY ui.js /opt/noVNC/app/ui.js
COPY default.template /etc/default.template
COPY MAIA.png /etc/MAIA.png
COPY bitmap.png /etc/bitmap.png
COPY entrypoint_debug.sh /etc/entrypoint_debug.sh

RUN usermod --login maia-user --move-home --home /home/maia-user user && groupmod --new-name maia-user user
RUN chsh -s /bin/zsh maia-user
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


COPY Tools/Slicer/Slicer.desktop /etc/Slicer.desktop
COPY Tools/Slicer/300px-3DSlicerLogo-app-icon512.svg.png /etc/300px-3DSlicerLogo-app-icon512.svg.png
RUN chmod 777 /etc/Slicer.desktop
RUN chmod a+x /etc/Slicer.desktop
COPY Tools/Slicer/install_slicer.sh /etc/install_slicer.sh
RUN chmod 777 /etc/install_slicer.sh
RUN chmod a+x /etc/install_slicer.sh

COPY Tools/ZSH/install_zsh.sh /etc/install_zsh.sh
RUN chmod 777 /etc/install_zsh.sh
RUN chmod a+x /etc/install_zsh.sh

COPY Tools/FreeSurfer/install_freesurfer.sh /etc/install_freesurfer.sh
RUN chmod 777 /etc/install_freesurfer.sh
RUN chmod a+x /etc/install_freesurfer.sh
COPY Tools/FreeSurfer/FreeView.desktop /etc/FreeView.desktop
COPY Tools/FreeSurfer/FreeSurfer.png /etc/FreeSurfer.png
RUN chmod 777 /etc/FreeView.desktop
RUN chmod a+x /etc/FreeView.desktop


COPY Tools/QuPath/QuPath.png /etc/QuPath.png
COPY Tools/QuPath/install_qupath.sh /etc/install_qupath.sh
COPY Tools/QuPath/QuPath.desktop /etc/QuPath.desktop
RUN chmod 777 /etc/QuPath.desktop
RUN chmod a+x /etc/QuPath.desktop
RUN chmod 777 /etc/install_qupath.sh
RUN chmod a+x /etc/install_qupath.sh


COPY Tools/ITKSnap/ITKSnap.png /etc/ITKSnap.png
COPY Tools/ITKSnap/install_itksnap.sh /etc/install_itksnap.sh
COPY Tools/ITKSnap/ITKSnap.desktop /etc/ITKSnap.desktop
RUN chmod 777 /etc/install_itksnap.sh
RUN chmod a+x /etc/install_itksnap.sh
RUN chmod 777 /etc/ITKSnap.desktop
RUN chmod a+x /etc/ITKSnap.desktop

USER 1000
ENV SHELL /bin/zsh


ENV USER maia-user
WORKDIR /home/maia-user
COPY .bash_profile /etc/.bash_profile
COPY .zshrc /etc/.zshrc
COPY .tmux.conf /etc/.tmux.conf


ENTRYPOINT ["bash", "/etc/entrypoint.sh"]

