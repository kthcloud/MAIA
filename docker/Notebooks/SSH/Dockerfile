ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive
USER root

RUN apt-get update && apt-get install -y openssh-server build-essential
RUN apt-get install -y sudo screen git nano tmux vim curl tree htop zsh wget rsync btop

COPY sshd-1.service /etc/systemd/system/sshd-1.service
COPY start.sh /opt/start.sh
COPY generate_user_environment.py /opt/generate_user_environment.py
RUN mkdir /opt/ssh
COPY sshd_config /opt/ssh/
RUN ssh-keygen -q -N "" -t dsa -f /opt/ssh/ssh_host_dsa_key
RUN ssh-keygen -q -N "" -t rsa -b 4096 -f /opt/ssh/ssh_host_rsa_key
RUN ssh-keygen -q -N "" -t ecdsa -f /opt/ssh/ssh_host_ecdsa_key
RUN ssh-keygen -q -N "" -t ed25519 -f /opt/ssh/ssh_host_ed25519_key
RUN mkdir /run/sshd
RUN chmod 600 /opt/ssh/*
RUN chmod 644 /opt/ssh/sshd_config
RUN chown -R maia-user. /opt/ssh/
RUN chown maia-user: /etc/systemd/system/sshd.service
#RUN systemctl enable sshd-1
#RUN python -m pip install  supervisor supervisord-dependent-startup
#COPY supervisord.conf /etc/supervisord.conf
COPY sshd.conf /etc/supervisord.d/sshd.conf


USER 1000
CMD ["jupyterhub-singleuser"]
ENTRYPOINT ["bash", "/opt/start.sh"]
