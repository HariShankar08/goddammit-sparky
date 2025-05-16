# Things to do for setup

I'm using a conda environment, so my setup looks like this:

```
# Create your own conda environment, then...
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# Pytorch geometric
pip install torch_scatter==2.0.9 torch_sparse==0.6.12 torch_cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch_geometric==2.0.4
# Additional libraries
pip install -r requirements.txt

# Downgrade protobuf - IMPORTANT.
pip install --upgrade --force-reinstall protobuf==3.20.*

# also install higher
pip install higher
```

***Important: After the change the data_dir in `GSAT/configs/global_config.yml`***

Currently, the `ba2_motifs` and `Graph-SST2` datasets work as expected. Of these, `ba2` will be 
downloaded normally as a part of the codebase. The files for `Graph-SST2` need to be downloaded;
Use this link to download: [Download Graph-SST2.zip here](https://drive.google.com/drive/folders/1dt0aGMBvCEUYzaG00TYu1D03GPO7305z)
Download and unzip the zip file in the `data_dir`.

Refer to the commands in `scripts/gsat.sh` to run and train the models.



