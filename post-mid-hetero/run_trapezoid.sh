#!/bin/bash

SEEDS=(42 123 101112)
# SEEDS=(101112)
METHODS=("otsu" "adaptive")
ALPHA_SPARSE=0.005
ALPHA_ENTROPY=0.001
ALPHA_MASK=0.1

mkdir -p results_t_2_authors_mask

for SEED in "${SEEDS[@]}"
do
  for METHOD in "${METHODS[@]}"
  do
    echo "🚀 Running: seed=$SEED, method=$METHOD"
    python3 trapezoid_motif.py \
      --seed $SEED \
      --method $METHOD \
      --alpha_sparse $ALPHA_SPARSE \
      --alpha_entropy $ALPHA_ENTROPY \
      --alpha_mask $ALPHA_MASK \
      --outdir results_t_2_authors_mask
  done
done
