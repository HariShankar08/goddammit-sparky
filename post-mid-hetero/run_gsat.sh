#!/usr/bin/env bash
# Grid-search over (gamma_writes × gamma_cites × alpha × beta × motif × seed)

OUT=gsat_hetero_results4_seed7_motif1
mkdir -p "$OUT"

# ── swept hyper-parameters ────────────────────────────────────────────────
gw_arr=(0.3 0.5 0.7 0.9)      # gamma_writes
gc_arr=(0.3 0.5 0.7 0.9)      # gamma_cites
alpha_arr=(1e-4)        # alpha values
beta_arr=(20)           # beta  values
motifs=(1)              # 0 = trapezoid, 1 = simple motif
seeds=(7)    # five seeds

# ── fixed hyper-parameters ────────────────────────────────────────────────
HIDDEN=128
LR=1e-3
TEMP=0.5
EPOCHS=20
PER_CLASS=100

# ── launch grid ───────────────────────────────────────────────────────────
for seed in "${seeds[@]}";  do
  for motif in "${motifs[@]}";  do
    for gw in "${gw_arr[@]}";   do
      for gc in "${gc_arr[@]}"; do
        for a in "${alpha_arr[@]}"; do
          for b in "${beta_arr[@]}";  do

            python main_gsat_hetero.py \
              --seed "$seed" \
              --motif "$motif" \
              --gamma_writes "$gw" \
              --gamma_cites  "$gc" \
              --alpha "$a" \
              --beta  "$b" \
              --hidden "$HIDDEN" \
              --lr "$LR" \
              --temp "$TEMP" \
              --epochs "$EPOCHS" \
              --per_class "$PER_CLASS" \
              --out_dir "$OUT"

          done
        done
      done
    done
  done
done
