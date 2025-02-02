ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG DEBIAN_FRONTEND=noninteractive
USER root

COPY dependencies.txt /tmp/dependencies.txt

RUN apt-get update && apt-get install --no-install-recommends -y `cat /tmp/dependencies.txt` \
    && apt-get clean && apt-get -y autoremove && rm -rf /var/lib/apt/lists/*

RUN wget https://www.mathworks.com/mpm/glnxa64/mpm && chmod +x mpm

USER 1000
RUN sudo mkdir /opt/matlab
RUN sudo chmod -R 777 /opt/matlab
RUN sudo ./mpm install --release R2024b --products MATLAB --destination=/opt/matlab
RUN sudo ln -s /opt/matlab/bin/matlab /usr/bin/matlab

USER root
RUN apt update && apt install xvfb -y
RUN pip install jupyter-matlab-proxy

RUN apt-get install r-base -y && apt-get install gdebi-core -y && wget https://download2.rstudio.org/server/jammy/amd64/rstudio-server-2024.04.2-764-amd64.deb
RUN gdebi rstudio-server-2024.04.2-764-amd64.deb -n
RUN pip install jupyter-rsession-proxy

#RUN pip install conda-store-server jupyterlab-conda-store

#RUN apt-get install python3-pyqt5.qtquick python3-pyqt5.qtsvg -y && pip install glueviz[recommended,qt]  glue-jupyterlab && pip install git+https://github.com/glue-viz/glue-jupyter.git


RUN pip install jupyter-ai jupyter-vscode-proxy

RUN pip install panel 

# watchfiles&& jupyter serverextension enable panel.io.jupyter_server_extension

RUN curl -fsSL https://code-server.dev/install.sh | sh


COPY Tools/Matlab/Matlab.desktop /etc/Matlab.desktop
COPY Tools/Matlab/Matlab_Logo.png /etc/Matlab_Logo.png
RUN chmod 777 /etc/Matlab.desktop
RUN chmod a+x /etc/Matlab.desktop

RUN pip install jupyterlab_nvdashboard "elyra[all]"
COPY start.sh /opt/start.sh
USER 1000
CMD ["jupyterhub-singleuser"]
ENTRYPOINT ["bash", "/opt/start.sh"]
