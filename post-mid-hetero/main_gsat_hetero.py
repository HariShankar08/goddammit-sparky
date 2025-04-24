#!/usr/bin/env python
"""
Train a GSAT-HeteroGNN on either the *trapezoid* (motif-0) or the *simple*
(motif-1) synthetic dataset, then save visual explanations and metrics for the
chosen hyper-parameter setting.

Usage example
-------------
python main.py --motif 0 --hidden 128 --lr 5e-4
"""

import os, argparse, pickle, torch, numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# ---------- motif-specific helpers -----------------------------------------
import trapezoid_motif   as motif0      # has generate_simple_dataset_with_noise
import simple_motif      as motif1      # has generate_simple_dataset_with_noise
from simple_motif import set_seed       # common RNG helper

# ---------- local GSAT utilities -------------------------------------------
from gsathet2 import (
    convert_to_hetero_data,  GSAT_HeteroGNN, GSATTrainer,
    visualize_explanation_topk, evaluate_explanations2
)
from utils_wandb import start_run, log_epoch, log_history, log_final

# ─────────────────────────────────────────────────────────────────────────────
def choose_motif(motif_id: int):
    """
    Returns (dataset_generator_fn,   # callable(path, per_class)
             top_k_dict,             # e.g. {'writes':2,'cites':4}
             motif_label)            # int label that means “motif present”
    """
    if motif_id == 0:   # trapezoid
        return motif0.generate_simple_dataset_with_noise, {'writes':0.5,'cites':0.5}, 0
    elif motif_id == 1: # simple 3-edge motif
        return motif1.generate_simple_dataset_with_noise, {'writes':0.5,'cites':0}, 0
    else:
        raise ValueError("motif must be 0 or 1")
# ---------- helper: did_GSAT_hit_all_edges? -------------------------------
def gsat_full_hit(graph, mask_edges, trainer, ratio):
    """
    Returns True iff *all* edges kept by sparse_topk belong to mask_edges.
    """
    from gsathet2 import sparse_topk, convert_to_hetero_data
    data = convert_to_hetero_data(graph)
    _, alpha = trainer.sample_edges(data, train=False)

    authors = [n for n,d in graph.nodes(data=True) if d['type']=='Author']
    papers  = [n for n,d in graph.nodes(data=True) if d['type']=='Paper']

    for rel, probs in alpha.items():
        rtype = rel[1]
        if rtype not in ratio or probs.numel()==0:
            continue
        keep = sparse_topk(probs, data.edge_index_dict[rel], ratio=ratio[rtype])
        ei   = data.edge_index_dict[rel][:, keep]
        srcL = authors if rel[0]=='author' else papers
        dstL = papers
        for i in range(ei.size(1)):
            edge = (srcL[ei[0,i]], dstL[ei[1,i]])
            if edge not in mask_edges and (edge[1],edge[0]) not in mask_edges:
                return False        # found a wrong edge
    return True                     # every kept edge is correct

# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--seed", type=int,   default=42)
    ap.add_argument("--per_class", type=int, default=100,
                    help="graphs per class to generate")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr",     type=float, default=1e-3)
    ap.add_argument("--alpha",  type=float, default=1e-4)
    ap.add_argument("--beta",   type=float, default=20.0)
    ap.add_argument("--temp",   type=float, default=0.5)
    ap.add_argument("--gamma_writes", type=float, default=0.5)
    ap.add_argument("--gamma_cites",  type=float, default=0.5)
    ap.add_argument("--out_dir", default="gsat_hetero_results")
    ap.add_argument("--motif",  type=int, choices=[0,1], default=0,
                    help="0 = trapezoid motif, 1 = simple 3-edge motif")
    args = ap.parse_args()

    # 1.  reproducibility, output folder
    set_seed(args.seed)
    tag = (f"seed{args.seed}motif{args.motif}_"
           f"a{args.alpha}_b{args.beta}_gr{args.gamma_writes}_"
           f"gc{args.gamma_cites}")
    run_dir = os.path.join(args.out_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    run = start_run(args, run_dir)


    # 2.  choose dataset generator / vis settings
    gen_fn, ratio, motif_label = choose_motif(args.motif)

    # 3.  cache dataset to run-specific folder
    ds_file = os.path.join(run_dir, "dataset.pkl")
    if not os.path.exists(ds_file):
        gen_fn(ds_file, args.per_class)       # writes the pickle
    with open(ds_file, "rb") as f:
        graphs, labels, masks = pickle.load(f)

    # 4.  convert to HeteroData
    data_list = []
    for G, y in zip(graphs, labels):
        d = convert_to_hetero_data(G)
        d.y = torch.tensor(y, dtype=torch.long)
        data_list.append(d)

    train, test = train_test_split(data_list, test_size=0.2,
                                   random_state=args.seed, stratify=labels)
    train_loader = DataLoader(train, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test,  batch_size=1)

    # 5.  model & trainer
    in_dim = data_list[0]['author'].x.size(1)
    rels   = [
        ('author','writes','paper'),
        ('paper','cites','paper'),
        ('author','self_loop','author'),
        ('paper','self_loop','paper')
    ]
    model   = GSAT_HeteroGNN(in_dim, args.hidden, 2, rels, temp=args.temp)
    trainer = GSATTrainer(
        model, lr=args.lr, temp=args.temp,
        gamma_dict={'writes': args.gamma_writes, 'cites': args.gamma_cites},
        alpha=args.alpha, beta=args.beta
    )

    # 6.  training
    loss_h, acc_h, auc_h = [], [], []
    for epoch in range(args.epochs):
        loss = trainer.train_epoch(train_loader)
        acc,auc  = trainer.test(test_loader)
        print(f"[{tag}] epoch {epoch:02d}: loss={loss:.4f} acc={acc:.3f} auc={auc:.3f}")
        loss_h.append(loss)
        acc_h.append(acc)
        auc_h.append(auc)
        log_epoch(epoch, loss, acc, auc)

    log_history(loss_h, acc_h, auc_h)

    # 7.  save explanations for first 4 graphs
    import matplotlib.pyplot as plt
    import random
    random.seed(args.seed)
    # 7.  visualise only perfect-recall explanations

    # 8.  quantitative metrics
    metrics = evaluate_explanations2(graphs,masks,labels,trainer,
                                     ratio=ratio, motif_class=motif_label)
    perfect = metrics['full_mask_covered_ids']
    import matplotlib.pyplot as plt
    for idx in perfect[:4]:                     # save up to 4 examples
        fig = visualize_explanation_topk(
                graphs[idx], trainer,
                ratio=ratio, mask_edges=masks[idx])
        fig.savefig(os.path.join(run_dir, f"graph_{idx}.png"))
        plt.close(fig)
    # 9.  persist hyper-parameters & metrics
    with open(os.path.join(run_dir,"metrics.txt"),'w') as fp:
        for k,v in metrics.items():
            fp.write(f"{k}: {v}\n")
    print("★ finished | graph ROC-AUC =", metrics['graph_roc_auc'],
          "| edge ROC-AUC =", metrics['edge_roc_auc'])
    log_final(metrics); run.finish()

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
