#FROM ghcr.io/selkies-project/nvidia-egl-desktop:22.04-20241222100454
FROM ghcr.io/selkies-project/nvidia-egl-desktop:22.04-20240225071130


USER root
RUN apt update && apt install nginx -y
RUN apt update && apt install -y zsh

COPY supervisord.conf   /etc/supervisord.conf
RUN chmod 755 /etc/supervisord.conf
COPY entrypoint.sh /etc/entrypoint.sh
RUN chmod 755 /etc/entrypoint.sh

COPY MAIA.png /etc/MAIA.png

RUN usermod --login maia-user --move-home --home /home/maia-user user && groupmod --new-name maia-user user
RUN chsh -s /bin/zsh maia-user
RUN echo "maia-user ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

COPY Tutorials /etc/Tutorials
COPY Welcome.ipynb /etc/Welcome.ipynb
RUN chmod -R 777 /etc/Tutorials
RUN chmod 777 /etc/Welcome.ipynb

COPY 70-maia /etc/update-motd.d/70-maia
RUN chmod +x /etc/update-motd.d/70-maia && chmod 777 /etc/update-motd.d/70-maia

COPY 90-qotd /etc/update-motd.d/90-qotd
RUN chmod +x /etc/update-motd.d/90-qotd && chmod 777 /etc/update-motd.d/90-qotd

COPY Tools/ZSH/install_zsh.sh /etc/install_zsh.sh
RUN chmod 777 /etc/install_zsh.sh
RUN chmod a+x /etc/install_zsh.sh


USER 1000
ENV SHELL /bin/zsh


ENV USER maia-user
WORKDIR /home/maia-user
COPY .bash_profile /etc/.bash_profile
COPY .zshrc /etc/.zshrc
COPY .tmux.conf /etc/.tmux.conf


ENTRYPOINT ["bash", "/etc/entrypoint.sh"]

