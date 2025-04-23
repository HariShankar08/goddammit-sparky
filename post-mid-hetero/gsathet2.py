import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import networkx as nx
import pickle
from simple_motif import set_seed



def convert_to_hetero_data(G):
    """
    G      – a NetworkX MultiDiGraph with node attr 'type' ∈ {'Author','Paper'}
    mask   – optional set of (u,v) motif edges. If given, the function prints
             and returns a boolean tensor aligned with the *writes* edges.
    """
    # print("\n==== Converting NetworkX graph to HeteroData ====")
    data = HeteroData()

    # 1.  build node lists and ID maps ──────────────────────────────
    authors = [n for n, d in G.nodes(data=True) if d["type"] == "Author"]
    papers  = [n for n, d in G.nodes(data=True) if d["type"] == "Paper"]
    A, P    = len(authors), len(papers)
    a_map   = {n: i for i, n in enumerate(authors)}
    p_map   = {n: i for i, n in enumerate(papers)}

    # print(f"Authors (|A|={A}):", authors)
    # print(f"Papers  (|P|={P}):", papers)
    # print("author-ID → row:", a_map)
    # print("paper-ID  → row:", p_map)

    # 2.  66-dim node features  (64-D one-hot  + degree + clustering coeff)
    def make_feats(nodes):
        oh  = torch.eye(len(nodes), 32)
        deg = torch.tensor([G.degree(n) for n in nodes]).float().view(-1, 1)
        cc  = torch.tensor([nx.clustering(nx.Graph(G), n) for n in nodes]).float().view(-1, 1)
        return torch.cat([oh, deg, cc], dim=1)

    data["author"].x = make_feats(authors)
    data["paper" ].x = make_feats(papers)
    # print("author.x shape :", tuple(data['author'].x.shape))
    # print("paper.x  shape :", tuple(data['paper' ].x.shape))

    # 3.  writes  (Author → Paper) ─────────────────────────────────
    writes = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "writes"]
    src_w  = [a_map[u] for u, v in writes]
    dst_w  = [p_map[v] for u, v in writes]
    data["author", "writes", "paper"].edge_index = torch.tensor([src_w, dst_w])
    # print("\nwrites edges           :", writes)
    # print("writes edge_index      :\n", data["author","writes","paper"].edge_index)


    # 4.  cites   (Paper  → Paper) ─────────────────────────────────
    cites = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "cites"
             and u in p_map and v in p_map]
    src_c  = [p_map[u] for u, v in cites]
    dst_c  = [p_map[v] for u, v in cites]
    data["paper", "cites", "paper"].edge_index = torch.tensor([src_c, dst_c])
    # print("\ncites edges            :", cites)
    # print("cites edge_index       :\n", data["paper","cites","paper"].edge_index)

    # 5.  explicit self-loops ──────────────────────────────────────
    for ntype, N in [("author", A), ("paper", P)]:
        idx = torch.arange(N)
        data[ntype, "self_loop", ntype].edge_index = torch.stack([idx, idx])
    # print("\nself-loops added for every node.")

    # print("Current keys in HeteroData:", list(data.keys()))
    # print("==== conversion complete ====\n")
    return data
class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        layers = []
        for i in range(len(dims)-1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        self.net = nn.Sequential(*layers[:-1])
    def forward(self, x):
        return self.net(x)

class GSAT_HeteroGNN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, relations, temp=1.0):
        super().__init__()
        self.relations = relations
        self.temp      = temp

        self.edge_mlps = nn.ModuleDict({
            '_'.join(rel): MLP([2*in_dim, hidden, 1])
            for rel in relations
        })
        self.conv1 = HeteroConv({rel: SAGEConv(in_dim, hidden) for rel in relations}, aggr='sum')
        self.conv2 = HeteroConv({rel: SAGEConv(hidden, hidden) for rel in relations}, aggr='sum')
        self.skip  = nn.ModuleDict({ntype: nn.Linear(in_dim, hidden) for ntype in ['author','paper']})
        self.cls   = MLP([2*hidden, hidden, out_dim])

    def forward(self, x_dict, edge_index_dict, edge_attn=None):
        # debug: print input feature shapes
        # print("forward: x_dict shapes:", {k: v.shape for k,v in x_dict.items()})
        # print("forward: edge_index_dict shapes:", {k: v.shape for k,v in edge_index_dict.items()})

        # Layer 1
        h1 = self.conv1(x_dict, edge_index_dict)
        # print("after conv1, h1 shapes:", {k: v.shape for k,v in h1.items()})
        h1 = {k: F.relu(v) for k,v in h1.items()}

        # Layer 2 + skip
        h2 = self.conv2(h1, edge_index_dict)
        # print("after conv2, h2 pre-skip shapes:", {k: v.shape for k,v in h2.items()})
        for ntype in self.skip:
            if ntype in h2:
                h2[ntype] = F.relu(h2[ntype] + self.skip[ntype](x_dict[ntype]))
        # print("after skip & relu, h2 shapes:", {k: v.shape for k,v in h2.items()})

        # pooling
        # print(x_dict['author'].size(0))
        
        a_batch = x_dict['author'].new_zeros(x_dict['author'].size(0), dtype=torch.long)
        p_batch = x_dict['paper'].new_zeros(x_dict['paper'].size(0),  dtype=torch.long)
        a_pool  = global_mean_pool(h2['author'], a_batch)
        p_pool  = global_mean_pool(h2['paper'],  p_batch)
        # print("a_pool shape:", a_pool.shape)
        # print("p_pool shape:", p_pool.shape)
        # logits
        logits = self.cls(torch.cat([a_pool, p_pool], dim=-1))
        # print("logits shape:", logits.shape)
        return logits, {}

# -----------------------------------------
# 3) GSATTrainer (with debug prints)
# -----------------------------------------
class GSATTrainer:
    def __init__(self, model, alpha = 0.5, beta = 0.1, lr=0.01, gamma_dict=None, temp=1.0):
        self.model      = model
        self.optim      = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma_dict = gamma_dict or {}
        self.temp       = temp
        self.alpha = alpha
        self.beta = beta

    def sample_edges(self, data, train=True):
        # print("sample_edges: x_dict shapes:", {k: v.shape for k,v in data.x_dict.items()})
        # print("sample_edges: edge_index_dict shapes:", {k: v.shape for k,v in data.edge_index_dict.items()})
        edge_masks = {}
        alpha_dict = {}

        for rel in self.model.relations:
            if rel not in data.edge_index_dict or data.edge_index_dict[rel].size(1) == 0:
                continue

            edge_index = data.edge_index_dict[rel]
            # print(f"original edge_index[{rel}] shape:", edge_index.shape)
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                edge_index = edge_index.view(2, -1)
                # print(f"  reshaped to:", edge_index.shape)
                data.edge_index_dict[rel] = edge_index

            src_t, _, dst_t = rel
            src, dst = edge_index
            h_src = data.x_dict[src_t][src]
            h_dst = data.x_dict[dst_t][dst]
            # print(f"{rel} h_src shape:", h_src.shape, "h_dst shape:", h_dst.shape)

            s = self.model.edge_mlps['_'.join(rel)](torch.cat([h_src,h_dst],-1)).view(-1)
            # print(f"{rel} s (logit input) shape:", s.shape)
            logits = torch.log(torch.sigmoid(s)+1e-10)
            # print(f"{rel} logits shape:", logits.shape)

            if train:
                u = torch.rand_like(logits)
                g = -torch.log(-torch.log(u+1e-10)+1e-10)
                thresh = torch.sigmoid((logits+g)/self.temp)
                hard   = (thresh>=0.5).float()
                mask   = (hard - thresh.detach() + thresh).view(-1)
            else:
                # mask = torch.sigmoid(logits).float()
                mask = (torch.sigmoid(logits) >= 0.5).float()
            # print(f"{rel} mask shape:", mask.shape)

            edge_masks[rel] = mask
            alpha_dict[rel] = torch.sigmoid(logits)

        # build masked_edges
        masked_edges = {}
        for rel, idx in data.edge_index_dict.items():
            if idx.dim() != 2 or idx.size(0) != 2:
                idx = idx.view(2, -1)
            if rel in edge_masks:
                m = edge_masks[rel].bool().view(-1)
                masked_edges[rel] = idx[:, m]
            else:
                masked_edges[rel] = idx

        # print("masked_edges shapes:", {k: v.shape for k,v in masked_edges.items()})
        # print("alpha_dict shapes:",    {k: v.shape for k,v in alpha_dict.items()})
        return masked_edges, alpha_dict

    def kl_loss(self, alpha, rel_type):
        gam = self.gamma_dict.get(rel_type, 0.5)
        return ( alpha*torch.log(alpha/gam+1e-10) +
                 (1-alpha)*torch.log((1-alpha)/(1-gam)+1e-10)
               ).mean()

    def train_epoch(self, loader):
        self.model.train()
        tot = 0
        for data in loader:
            # print("train_epoch: data.x_dict shapes:", {k: v.shape for k,v in data.x_dict.items()})
            # print("train_epoch: data.edge_index_dict shapes:", {k: v.shape for k,v in data.edge_index_dict.items()})

            me, alph = self.sample_edges(data, train=True)
            # print("train_epoch: masked_edges shapes:", {k: v.shape for k,v in me.items()})
            # print("train_epoch: alpha_dict shapes:",   {k: v.shape for k,v in alph.items()})

            logits, _ = self.model(data.x_dict, me)
            # print("train_epoch: logits shape:", logits.shape)

            y = data.y
            if y.dim() == 0:
                y = y.unsqueeze(0)
            y = y.long()
            # print("train_epoch: labels shape:", y.shape)

            task_loss = F.cross_entropy(logits, y)
            reg_loss  = sum(self.kl_loss(a, r[1]) for r,a in alph.items() if a.numel()>0)
            loss = self.alpha* task_loss + self.beta* reg_loss
            # print("train_epoch: task_loss =", task_loss.item(), "reg_loss =", reg_loss.item(), "total=", loss.item())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            tot += loss.item()
        return tot / len(loader)

    def test(self, loader):
        self.model.eval()

        correct = tot = 0
        y_true, y_score = [], []

        with torch.no_grad():
            for data in loader:
                masked_edges, _ = self.sample_edges(data, train=False)
                logits, _ = self.model(data.x_dict, masked_edges)   # shape (B, 2)

                probs  = torch.softmax(logits, dim=-1)              # (B, 2)
                preds  = logits.argmax(dim=-1)
                correct += (preds == data.y).sum().item()
                tot     += data.y.size(0)
                y_true.extend(data.y.cpu().numpy())
                y_score.extend(probs[:, 1].cpu().numpy())           # P(class=1)

        acc = correct / tot if tot else 0.0
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:              # all labels identical in this split
            auc = float('nan')

        print(f"test: correct={correct} total={tot}  acc={acc:.3f}  roc_auc={auc:.3f}")
        return acc, auc


# want to visualize the explanations 
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_explanation_topk(graph, trainer, convert_to_hetero_data,
                                mask_edges, top_k = {'writes': 2, 'cites': 4}):
    """
    Like visualize_explanation, but selects exactly `top_k` edges
    per relation (writes/cites), ranked by attention score.
    """
    data = convert_to_hetero_data(graph)
    _, alpha_dict = trainer.sample_edges(data, train=False)

    H = nx.Graph()
    H.add_nodes_from(graph.nodes(data=True))
    H.add_edges_from(graph.edges(data=True))
    pos = nx.spring_layout(H)

    # collect the top_k edges
    gsat_edges = []
    for rel, alpha in alpha_dict.items():
        if rel[1] not in {'writes','cites'}:
            continue
        scores = alpha.cpu().detach().numpy()
        if scores.size == 0:
            continue
        if top_k[rel[1]] == 0:
            continue

        # get indices of top_k scores
        idxs = np.argsort(scores)[::-1][:top_k[rel[1]]]
        ei = data.edge_index_dict[rel]
        authors = [n for n,d in graph.nodes(data=True) if d['type']=='Author']
        papers  = [n for n,d in graph.nodes(data=True) if d['type']=='Paper']
        src_list = authors if rel[0]=='author' else papers
        dst_list = papers

        for idx in idxs:
            u = src_list[ ei[0,idx].item() ]
            v = dst_list[ ei[1,idx].item() ]
            gsat_edges.append((u, v))

    # plot
    plt.figure(figsize=(6,4))
    authors = [n for n,d in graph.nodes(data=True) if d['type']=='Author']
    papers  = [n for n,d in graph.nodes(data=True) if d['type']=='Paper']

    nx.draw_networkx_nodes(H, pos, nodelist=authors,
                           node_color='skyblue', node_shape='o', label='Author')
    nx.draw_networkx_nodes(H, pos, nodelist=papers,
                           node_color='lightgreen', node_shape='s', label='Paper')
    nx.draw_networkx_labels(H, pos, font_size=8)
    nx.draw_networkx_edges(H, pos, edge_color='lightgrey', width=1)

    if gsat_edges:
        nx.draw_networkx_edges(H, pos,
            edgelist=gsat_edges, edge_color='red', width=2, label='GSAT top-k')

    if mask_edges:
        nx.draw_networkx_edges(H, pos,
            edgelist=mask_edges, edge_color='black',
            style='dashed', width=2, label='Motif (GT)')

    plt.title(f"GSAT Explanation (top {top_k} edges)")
    plt.legend()
    plt.axis('off')
    # plt.show()
    return plt.gcf()


from sklearn.metrics import roc_auc_score
import torch, numpy as np

@torch.no_grad()
def evaluate_explanations2(
        graphs, masks, labels,
        trainer, convert_fn,
        top_k={'writes':2, 'cites':4},
        motif_class=0                 # which label corresponds to “motif present”
):
    hits_per_graph, fid_pos, fid_neg = [], [], []
    y_true, y_score                   = [], []   # <- now filled for *all* graphs
    rels = trainer.model.relations

    for G, gt_mask, y in zip(graphs, masks, labels):
        data = convert_fn(G)
        _, alph = trainer.sample_edges(data, train=False)

        # ---------- ROC bookkeeping (all graphs, all edges) ---------------
        authors = [n for n,d in G.nodes(data=True) if d['type']=='Author']
        papers  = [n for n,d in G.nodes(data=True) if d['type']=='Paper']
        for rel, probs in alph.items():
            ei   = data.edge_index_dict[rel]
            srcL = authors if rel[0]=='author' else papers
            dstL = papers
            for i,p in enumerate(probs):
                u,v = srcL[ei[0,i]], dstL[ei[1,i]]
                y_true.append(int((u,v) in gt_mask or (v,u) in gt_mask))
                y_score.append(p.item())

        # ---------- Skip accuracy / fidelity if no motif in this graph ----
        if y != motif_class or len(gt_mask) == 0:
            continue

        # ---------- build top‑k masks & pred edge list --------------------
        topk_bool, pred_edges = {}, []
        for rel, probs in alph.items():
            rtype = rel[1]; k = top_k.get(rtype, 0)
            m = torch.zeros_like(probs, dtype=torch.bool)
            if k > 0 and probs.numel():
                idxs = torch.topk(probs, k=min(k, probs.numel())).indices
                m[idxs] = True
                ei   = data.edge_index_dict[rel]
                srcL = authors if rel[0]=='author' else papers
                dstL = papers
                for idx in idxs:
                    pred_edges.append((srcL[ei[0,idx]], dstL[ei[1,idx]]))
            topk_bool[rel] = m

        # ---------- edge‑level accuracy -----------------------------------
        if pred_edges:
            hits = sum(e in gt_mask or (e[1],e[0]) in gt_mask for e in pred_edges)
            hits_per_graph.append(hits / len(pred_edges))

        # ---------- build drop / keep dicts for fidelity ------------------
        drop_dict, keep_dict = {}, {}
        for rel in rels:
            ei = data.edge_index_dict[rel]
            if rel in topk_bool:
                m = ~topk_bool[rel]
                drop_dict[rel] = ei[:, m]
                keep_dict[rel] = ei[:, ~m] if (~m).any() else ei[:, :0]
            else:
                drop_dict[rel] = ei
                keep_dict[rel] = ei[:, :0]
            if rel[1] == 'self_loop':
                keep_dict[rel] = ei

        log_full = trainer.model(data.x_dict, data.edge_index_dict)[0]
        log_drop = trainer.model(data.x_dict, drop_dict)[0]
        log_keep = trainer.model(data.x_dict, keep_dict)[0]
        fid_pos.append((log_full.argmax() == log_drop.argmax()).item())
        fid_neg.append((log_full.argmax() == log_keep.argmax()).item())

    # ------------- aggregation -------------------------------------------
    roc  = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else float('nan')
    return dict(
        edge_accuracy = float(np.mean(hits_per_graph)) if hits_per_graph else float('nan'),
        roc_auc       = roc,
        fidelity_pos  = float(np.mean(fid_pos)) if fid_pos else float('nan'),
        fidelity_neg  = float(np.mean(fid_neg)) if fid_neg else float('nan'),
        per_graph_hits = hits_per_graph,
        evaluated_graphs = len(hits_per_graph)
    )



# -----------------------------------------
# 4) Example usage (unchanged)
# -----------------------------------------
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per_class", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma_writes", type=float, default=0.5)
    parser.add_argument("--gamma_cites", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.0001)
    parser.add_argument("--beta", type=float, default=20.0)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--img_dir", type=str, default="gsat_hetero_explanations")
    parser.add_argument("--motif", type=int, default=0)

    args = parser.parse_args()
    if args.motif == 0:
        from trapezoid_motif import generate_simple_dataset_with_noise
        filename = "trapezoid_motif_gsat.pkl"
        top_k = {'writes': 2, 'cites': 4}
    elif args.motif == 1:
        from simple_motif import generate_simple_dataset_with_noise
        filename = "simple_motif_gsat.pkl"
        top_k = {'writes': 2, 'cites': 0}
    else: 
        raise ValueError("motif must be 0 or 1")
        
    set_seed(61)
    generate_simple_dataset_with_noise("hit.pkl",100)
    with open("hit.pkl","rb") as f:
        graphs, labels, masks = pickle.load(f)

    data_list = []
    for G,y in zip(graphs,labels):
        d = convert_to_hetero_data(G)
        d.y = torch.tensor(y, dtype=torch.long)
        data_list.append(d)

    train, test = train_test_split(data_list, test_size=0.2)
    train_loader = DataLoader(train, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test, batch_size=1)

    in_dim = data_list[0]['author'].x.size(1)
    rels   = [
      ('author','writes','paper'),
      ('paper','cites','paper'),
      ('author','self_loop','author'),
      ('paper','self_loop','paper')
    ]
    model   = GSAT_HeteroGNN(in_dim, 128, 2, rels, temp=0.5)
    trainer = GSATTrainer(model, lr=0.001,
                          gamma_dict={'writes':0.5,'cites':0.5}, temp=0.5,
                          alpha = 0.00001, beta = 20.0)
    from tqdm import tqdm
    for epoch in tqdm(range(20)):
        loss = trainer.train_epoch(train_loader)
        acc,auc  = trainer.test(test_loader)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}, ROC AUC={auc:.4f}")
    # fig = visualize_explanation_topk(graphs[0], trainer, convert_to_hetero_data,mask_edges=masks[0], top_k=topk
    # fig.savefig(os.path.join(img_dir, f"graph_{i+1}.png"))
    # plt.close(fig)
    # visualize_explanation_topk(graphs[1], trainer, convert_to_hetero_data, mask_edges=masks[1])
    # visualize_explanation_topk(graphs[2], trainer, convert_to_hetero_data,mask_edges=masks[2])
    # visualize_explanation_topk(graphs[3], trainer, convert_to_hetero_data, mask_edges=masks[3])

    metrics = evaluate_explanations2(
            graphs, masks, labels, trainer, convert_to_hetero_data,
            top_k=top_k, motif_class = 0) 

    for k, v in metrics.items():
        if isinstance(v, list):
            print(f"{k}: mean={np.mean(v):.3f}, per‑graph={v}")
        else:
            print(f"{k}: {v:.3f}")

