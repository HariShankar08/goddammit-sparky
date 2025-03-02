#!/bin/bash

# Create your own conda environment, then...
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# Pytorch geometric
pip install torch_scatter==2.0.9 torch_sparse==0.6.12 torch_cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch_geometric==2.0.4
# Additional libraries
pip install -r requirements.txt
