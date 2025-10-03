#!/bin/bash
export CONDA_AGREE=yes
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Activate conda - remove following lines if not needed
source /opt/conda/bin/activate
echo -e "\nif [ -f ~/.bashrc ]; then\n    source ~/.bashrc\nfi" >> ~/.bash_profile
source ~/.bashrc
# Update PATH
export PATH="/opt/conda/bin:$PATH"
# Remove until here

# Create the conda environment with only PyTorch
conda create --prefix ~/.conda/envs/DL_labs_GPU -y
conda activate DL_labs_GPU
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install monai

# Update the environment with the rest of the packages from the YAML file
conda env update -n DL_labs_GPU -f "$SCRIPT_DIR/environment.yml"

# Install ipykernel
pip install ipykernel

# Install the kernel for Jupyter
python -m ipykernel install --user --name=DL_labs_GPU

echo "Setup complete!"