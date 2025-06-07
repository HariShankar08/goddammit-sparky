SpMotif-0.7 Logs linked here: [Google Drive](https://drive.google.com/drive/folders/1fPsHZWV-GeazKkiC1md2W-dk81UAGXBr?usp=sharing)

## Preparation

### Environment Setup

We mainly use the following key libraries with the cuda version of 11.3:

```
torch==1.10.1+cu113
torch_cluster==1.6.0
torch_scatter==2.0.9
torch_sparse==0.6.12
torch_geometric==2.0.4
```

To setup the environment, one may use the following commands under the conda environments:

```
# Create your own conda environment, then...
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# Pytorch geometric
pip install torch_scatter==2.0.9 torch_sparse==0.6.12 torch_cluster==1.6.0 -f https://data.pyg.org/whl/torch-1.10.1+cu113.html
pip install torch_geometric==2.0.4
# Additional libraries
pip install -r requirements.txt
```

### Datasets

To prepare the datasets of regular graphs, following the [instructions in GSAT](https://github.com/Graph-COM/GSAT?tab=readme-ov-file#instructions-on-acquiring-datasets).

To prepare the datasets of geometric graphs, following the [instructions in LRI](https://github.com/Graph-COM/LRI?tab=readme-ov-file#datasets).

## Experiments on Regular Graphs

`/GSAT` contains the codes for running on regular graphs. The instructions to reproduce our results are given in [scripts/gsat.sh](scripts/gsat.sh).

### Sample Commands

For `GSAT`

```
python run_gmt.py --dataset spmotif_0.5 --backbone GIN --cuda 0 -fs 1 -mt 0
```

For `GMT-lin`

```
python run_gmt.py --dataset spmotif_0.5 --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 0.5
```

For `GMT-sam`

```
# train subgraph extractor
python run_gmt.py --dataset spmotif_0.5 --backbone GIN --cuda 0 -fs 1 -mt 5 -st 200 -ie 0.5 -sm 
# train subgraph classifier
python run_gmt.py --dataset spmotif_0.5 --backbone GIN --cuda 0 -fs 1 -mt 5550 -st 200 -ie 0.5 -fm -sr 0.8
```

## Experiments on Geometric Graphs

`/LRI` contains the codes for running on geometric graphs. The instructions to reproduce our results are given in [scripts/lri.sh](scripts/lri.sh).

### Sample Commands

For `LRI-Bern`

```
python trainer.py -ba --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -mt 0
```

For `GMT-lin`

```
python trainer.py -ba --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -mt 0 -ie 0.1
```

For `GMT-sam`

```
# train subgraph extractor
python trainer.py -ba -smt 55 -ie 0.1 -fr 0.7 --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -mt 55 -ir 1
# train subgraph classifier
python trainer.py -ba -smt 55 -ie 0.1 -fr 0.7 --cuda 0 --backbone egnn --dataset actstrack_2T --method lri_bern -mt 5553
```
