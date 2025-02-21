# Singularity
Singularity is a containerization tool that allows you to run your code in a containerized environment. This means that you can run your code in a controlled environment, which is isolated from the host system. This is useful when you want to run your code on a different system, where the dependencies are not available.

## Singularity Images

Singularity images are the files that contain the environment in which your code will run. You can create your own Singularity image or use an existing one.
You can find existing Singularity images on the [Singularity Hub](https://singularity-hub.org/).
To create your own Singularity image from an existing Docker image, you can use the `singularity pull` command.

```bash
singularity pull --name python-3.8.sif docker://python:3.8
```
and to run the Singularity image, you can use the `singularity run` command.

```bash
singularity run python-3.8.sif <COMMAND>
```

## Singularity Definition File
 To create a Singularity image, you need a Singularity definition file, which is a text file that describes the environment that you want to create.
[HPPCM](https://github.com/NVIDIA/hpc-container-maker), a tool to create container images for HPC applications from given recipes.
You only need to provide a recipe file that describes the environment you want to create, and HPC Container Maker will generate the Singularity definition file for you.

To install HPPCM, you can run the following command:

```bash
pip install hpccm
```

Recipe example to create a Singularity image with PyTorch, MONAI and and JupyterLab:

```python
#!/usr/bin/env python

import hpccm

Stage0 += baseimage(image='nvcr.io/nvidia/pytorch:23.10-py3')
Stage0 += gnu()
Stage0 += pip(ospackages=[""], packages=[
                                         "monai",
                                         "jupyter",
                                         "jupyterlab"
                                         ])
Stage0 += runscript(commands=[''])
```
To create the Singularity image, you can run the following command:

```bash
hpccm --recipe recipe.py --format singularity > PyMAIA.def
singularity build MONAI.sif MONAI.def
```
or, to create the corresponding Docker image, you can run the following command:

```bash
hpccm --recipe recipe.py --format docker > PyMAIA.docker
docker build -t MONAI -f PyMAIA.docker .
```
