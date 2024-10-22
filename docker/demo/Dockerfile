FROM jupyter/scipy-notebook:latest
#RUN conda install -c conda-forge conda-store-server && conda install -c conda-forge jupyterlab-conda-store
#RUN pip install panel watchfiles
#USER root
CMD ["jupyter", "lab", "--ip", "0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=$(echo $2 | python -c \"from jupyter_server.auth import passwd; import os;print(passwd(os.environ['PW']))\")" ]
