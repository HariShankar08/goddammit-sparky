# eval.py
"""
Edge‑level ROC‑AUC is still global (every edge, every class).
Accuracy / fidelity are computed *per‑class* and reported in a dict.
"""
from sklearn.metrics import roc_auc_score
import numpy as np, torch

@torch.no_grad()
def evaluate_explanations_multi(graphs,masks,labels,trainer,convert_fn,
                                top_k={'writes':2,'cites':4}):
    per_class_hits={c:[] for c in set(labels)}
    y_true=[]; y_score=[]
    fid_pos={c:[] for c in set(labels)}; fid_neg={c:[] for c in set(labels)}

    rels=trainer.model.relations
    for G,mask,y in zip(graphs,masks,labels):
        data=convert_fn(G)
        _,alph=trainer.sample_edges(data,train=False)

        # --- ROC bookkeeping (all edges) ----------------
        auth=[n for n,d in G.nodes(data=True) if d['type']=='Author']
        pap =[n for n,d in G.nodes(data=True) if d['type']=='Paper']
        for rel,pr in alph.items():
            ei=data.edge_index_dict[rel]; srcL=auth if rel[0]=='author' else pap
            for i,p in enumerate(pr):
                u,v=srcL[ei[0,i]], pap[ei[1,i]]
                y_true.append(int((u,v) in mask or (v,u) in mask))
                y_score.append(p.item())

        # skip explanation metrics if no motif in this graph
        if y==3 or len(mask)==0: continue

        # --- build top‑k edge set -----------------------
        pred=set(); topbool={}
        for rel,pr in alph.items():
            k=top_k.get(rel[1],0); m=torch.zeros_like(pr,dtype=torch.bool)
            if k and pr.numel():
                idx=torch.topk(pr,min(k,pr.numel())).indices; m[idx]=1
                ei=data.edge_index_dict[rel]; srcL=auth if rel[0]=='author' else pap
                for idx in idx: pred.add((srcL[ei[0,idx]],pap[ei[1,idx]]))
            topbool[rel]=m

        if pred:
            per_class_hits[y].append(sum(e in mask or (e[1],e[0]) in mask for e in pred)/len(pred))

        # --- fidelity (same as earlier) -----------------
        drop,keep={},{}
        for rel in rels:
            ei=data.edge_index_dict[rel]; m=~topbool.get(rel,torch.ones(ei.size(1),dtype=torch.bool))
            drop[rel]=ei[:,m]; keep[rel]=ei[:,~m] if (~m).any() else ei[:,:0]
            if rel[1]=='self_loop': keep[rel]=ei
        log_full=trainer.model(data.x_dict,data.edge_index_dict)[0]
        log_drop=trainer.model(data.x_dict,drop)[0]
        log_keep=trainer.model(data.x_dict,keep)[0]
        fid_pos[y].append((log_full.argmax()==log_drop.argmax()).item())
        fid_neg[y].append((log_full.argmax()==log_keep.argmax()).item())

    return dict(
        roc_auc=roc_auc_score(y_true,y_score),
        edge_accuracy={c:np.mean(per_class_hits[c]) for c in per_class_hits},
        fidelity_pos ={c:np.mean(fid_pos[c])       for c in fid_pos},
        fidelity_neg ={c:np.mean(fid_neg[c])       for c in fid_neg},
    )
