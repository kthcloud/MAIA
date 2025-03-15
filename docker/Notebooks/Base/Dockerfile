ARG BASE_IMAGE
FROM ${BASE_IMAGE}



ARG DEBIAN_FRONTEND=noninteractive

USER root
RUN curl -sL https://deb.nodesource.com/setup_20.x -o /tmp/nodesource_setup.sh
RUN bash /tmp/nodesource_setup.sh
RUN apt install nodejs


RUN mkdir -p /opt/conda
RUN chmod -R 777 /opt/conda/
USER 1000
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/conda/miniconda.sh
RUN bash /opt/conda/miniconda.sh -b -u -p /opt/conda
RUN rm -rf /opt/conda/miniconda.sh

RUN . /opt/conda/etc/profile.d/conda.sh

USER root
#RUN pip install jupyterhub==4.0.2  notebook==6.* jupyterlab==4.1.5
RUN pip install jupyterhub  notebook jupyterlab
RUN pip install jupyterlab-git jupyter_scheduler
RUN pip install jupyter_contrib_nbextensions  && pip install nbgitpuller 
#&& jupyter contrib nbextension install --user
RUN pip install python-dotenv jupyter-server-proxy
RUN pip install light-the-torch
ENV PATH="$PATH:/root/.local/bin"

USER 1000
COPY start.sh /etc/start.sh
RUN sudo chmod 755 /etc/start.sh


ENTRYPOINT ["/etc/start.sh"]