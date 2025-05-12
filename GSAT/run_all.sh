#!/bin/bash

# Function to run commands with and without --meta

# log_prefix="logs/${backbone}_${dataset}_${mode}"
mkdir -p logs

run_command() {
    local backbone=$1
    local dataset=$2
    local mode=$3
    local cmd=$4
    local log_prefix="logs/${backbone}_${dataset}_${mode}"
    # Normal GMT
    echo "Running $mode with backbone: $backbone on dataset: $dataset"
    eval "$cmd" > "${log_prefix}.out" 2> "${log_prefix}.err"
    echo

    # Meta GMT
    echo "Running Meta$mode with backbone: $backbone on dataset: $dataset"
    eval "$cmd --meta" > "${log_prefix}_meta.out" 2> "${log_prefix}_meta.err"
    echo
}

# All commands with structured info
commands=(
"GIN ba_2motifs GMT-sam 'python run_gmt.py --dataset ba_2motifs --backbone GIN --cuda 0 -fs 1 -mt 8 -st 10 -ie 0.1 -r 0.5 -sm'"
"PNA ba_2motifs GMT-lin 'python run_gmt.py --dataset ba_2motifs --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 0.1'"
"PNA ba_2motifs GMT-sam 'python run_gmt.py --dataset ba_2motifs --backbone PNA --cuda 0 -fs 1 -mt 5 -st 20 -ie 0.1 -r 0.5 -sm'"

"GIN mutag GMT-lin 'python run_gmt.py --dataset mutag --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 0.1'"
"GIN mutag GMT-sam 'python run_gmt.py --dataset mutag --backbone GIN --cuda 0 -fs 1 -mt 5 -st 100 -ie 0.1 -sm'"
"PNA mutag GMT-lin 'python run_gmt.py --dataset mutag --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 0.5'"
"PNA mutag GMT-sam 'python run_gmt.py --dataset mutag --backbone PNA --cuda 0 -fs 1 -mt 5 -st 1 -ie 0.5 -r 0.6 -sm'"

"GIN spmotif_0.5 GMT-lin 'python run_gmt.py --dataset spmotif_0.5 --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 0.5'"
"GIN spmotif_0.5 GMT-sam 'python run_gmt.py --dataset spmotif_0.5 --backbone GIN --cuda 0 -fs 1 -mt 5 -di 10 -st 80 -ie 0.5 -sm'"
"PNA spmotif_0.5 GMT-lin 'python run_gmt.py --dataset spmotif_0.5 --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 1'"
"PNA spmotif_0.5 GMT-sam 'python run_gmt.py --dataset spmotif_0.5 --backbone PNA --cuda 0 -fs 1 -mt 5 -di -1 -st 80 -ie 1 -sm'"

"GIN spmotif_0.7 GMT-lin 'python run_gmt.py --dataset spmotif_0.7 --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 1'"
"GIN spmotif_0.7 GMT-sam 'python run_gmt.py --dataset spmotif_0.7 --backbone GIN --cuda 0 -fs 1 -mt 5 -di 20 -st 200 -ie 0.1 -sm'"
"PNA spmotif_0.7 GMT-lin 'python run_gmt.py --dataset spmotif_0.7 --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 1'"
"PNA spmotif_0.7 GMT-sam 'python run_gmt.py --dataset spmotif_0.7 --backbone PNA --cuda 0 -fs 1 -mt 5 -di -1 -st 100 -ie 0.1 -r 0.8 -sm'"

"GIN spmotif_0.9 GMT-lin 'python run_gmt.py --dataset spmotif_0.9 --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 0.5'"
"GIN spmotif_0.9 GMT-sam 'python run_gmt.py --dataset spmotif_0.9 --backbone GIN --cuda 0 -fs 1 -mt 5 -di 20 -st 200 -ie 0.1 -sm'"
"PNA spmotif_0.9 GMT-lin 'python run_gmt.py --dataset spmotif_0.9 --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 1'"
"PNA spmotif_0.9 GMT-sam 'python run_gmt.py --dataset spmotif_0.9 --backbone PNA --cuda 0 -fs 1 -mt 5 -di -1 -st 80 -ie 1 -sm'"

"GIN ogbg_molhiv GMT-lin 'python run_gmt.py --dataset ogbg_molhiv --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 1'"
"GIN ogbg_molhiv GMT-sam 'python run_gmt.py --dataset ogbg_molhiv --backbone GIN --cuda 0 -fs 1 -mt 5 -di 20 -st 100 -ie 0.1 -sm'"
"PNA ogbg_molhiv GMT-lin 'python run_gmt.py --dataset ogbg_molhiv --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 1'"
"PNA ogbg_molhiv GMT-sam 'python run_gmt.py --dataset ogbg_molhiv --backbone PNA --cuda 0 -fs 1 -mt 5 -di 10 -st 100 -ie 0.5 -sm'"

"GIN Graph-SST2 GMT-lin 'python run_gmt.py --dataset Graph-SST2 --backbone GIN --cuda 0 -fs 1 -mt 3 -ie 1'"
"GIN Graph-SST2 GMT-sam 'python run_gmt.py --dataset Graph-SST2 --backbone GIN --cuda 0 -fs 1 -mt 5 -r 0.7 -st 100 -ie 1 -sm'"
"PNA Graph-SST2 GMT-lin 'python run_gmt.py --dataset Graph-SST2 --backbone PNA --cuda 0 -fs 1 -mt 3 -ie 0.1'"
"PNA Graph-SST2 GMT-sam 'python run_gmt.py --dataset Graph-SST2 --backbone PNA --cuda 0 -fs 1 -mt 5 -di 20 -st 100 -ie 0.5 -sm'"
)

# Run all commands
for entry in "${commands[@]}"; do
    eval run_command $entry
done
