import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import networkx as nx
import pickle

# --------------------------------------------------
# 1) Graph → HeteroData converter (with self‑loops)
# --------------------------------------------------
def convert_to_hetero_data(G, mask=None):
    data = HeteroData()
    # collect and index nodes
    author_nodes = [n for n,d in G.nodes(data=True) if d['type']=='Author']
    paper_nodes  = [n for n,d in G.nodes(data=True) if d['type']=='Paper']
    A = len(author_nodes); P = len(paper_nodes)
    a_map = {n:i for i,n in enumerate(author_nodes)}
    p_map = {n:i for i,n in enumerate(paper_nodes)}

    # one‐hot features (64 dim)
    data['author'].x = torch.eye(A,64)
    data['paper'].x  = torch.eye(P,64)

    # writes edges
    writes = [(u,v) for u,v,d in G.edges(data=True) if d['type']=='writes']
    if writes:
        src = [a_map[u] for u,v in writes]
        dst = [p_map[v] for u,v in writes]
        data['author','writes','paper'].edge_index = torch.tensor([src,dst], dtype=torch.long)
        mask_tensor = torch.zeros(len(writes))
        if mask:
            for i,(u,v) in enumerate(writes):
                if (u,v) in mask: mask_tensor[i]=1
        data['author','writes','paper'].edge_mask = mask_tensor
    else:
        data['author','writes','paper'].edge_index = torch.empty(2,0, dtype=torch.long)
        data['author','writes','paper'].edge_mask = torch.empty(0)

    # cites edges
    cites = [(u,v) for u,v,d in G.edges(data=True) if d['type']=='cites']
    if cites:
        valid_cites = [(u,v) for u,v in cites if u in p_map and v in p_map]
        if valid_cites:
            src = [p_map[u] for u,v in valid_cites]
            dst = [p_map[v] for u,v in valid_cites]
            data['paper','cites','paper'].edge_index = torch.tensor([src,dst], dtype=torch.long)
            mask_tensor = torch.zeros(len(valid_cites))
            if mask:
                for i,(u,v) in enumerate(valid_cites):
                    if (u,v) in mask: mask_tensor[i]=1
            data['paper','cites','paper'].edge_mask = mask_tensor
        else:
            data['paper','cites','paper'].edge_index = torch.empty(2,0, dtype=torch.long)
            data['paper','cites','paper'].edge_mask = torch.empty(0)
    else:
        data['paper','cites','paper'].edge_index = torch.empty(2,0, dtype=torch.long)
        data['paper','cites','paper'].edge_mask = torch.empty(0)

    # add self‑loops so every node gets messages
    for ntype, N in [('author',A), ('paper',P)]:
        idx = torch.arange(N, dtype=torch.long)
        data[ntype, 'self_loop', ntype].edge_index = torch.stack([idx,idx], dim=0)

    return data

# -----------------------------------------
# 2) GSAT‑style HeteroGNN (with debug prints)
# -----------------------------------------
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
                hard   = (thresh>=0.8).float()
                mask   = (hard - thresh.detach() + thresh).view(-1)
            else:
                # mask = torch.sigmoid(logits).float()
                mask = (torch.sigmoid(logits) >= 0.8).float()
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
        with torch.no_grad():
            for data in loader:
                me, _ = self.sample_edges(data, train=False)
                logits, _ = self.model(data.x_dict, me)
                preds = logits.argmax(dim=-1)
                correct += (preds==data.y).sum().item()
                tot += data.y.size(0)
        print("test: correct =", correct, "total =", tot)
        return correct/tot if tot else 0.0


# want to visualize the explanations 
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_explanation_topk(graph, trainer, convert_to_hetero_data,
                                mask_edges, top_k=3):
    """
    Like visualize_explanation, but selects exactly `top_k` edges
    per relation (writes/cites), ranked by attention score.
    """
    data = convert_to_hetero_data(graph)
    _, alpha_dict = trainer.sample_edges(data, train=False)

    H = nx.Graph()
    H.add_nodes_from(graph.nodes(data=True))
    H.add_edges_from(graph.edges(data=True))
    pos = nx.spring_layout(H, seed=42)

    # collect the top_k edges
    gsat_edges = []
    for rel, alpha in alpha_dict.items():
        if rel[1] not in {'writes','cites'}:
            continue
        scores = alpha.cpu().numpy()
        if scores.size == 0:
            continue

        # get indices of top_k scores
        idxs = np.argsort(scores)[-top_k:]
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
    plt.show()




# -----------------------------------------
# 4) Example usage (unchanged)
# -----------------------------------------
if __name__ == "__main__":
    from simple_motif import generate_simple_dataset_with_noise
    generate_simple_dataset_with_noise("hi.pkl",50)
    with open("hi.pkl","rb") as f:
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
    model   = GSAT_HeteroGNN(in_dim, 256, 2, rels, temp=1.0)
    trainer = GSATTrainer(model, lr=0.01,
                          gamma_dict={'writes':0.2,'cites':0.1}, temp=1.0,
                          alpha = 0.5, beta = 1.0)

    for epoch in range(10):
        loss = trainer.train_epoch(train_loader)
        acc  = trainer.test(test_loader)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")
    
    visualize_explanation_topk(graphs[0], trainer, convert_to_hetero_data,mask_edges=masks[0])
    visualize_explanation_topk(graphs[1], trainer, convert_to_hetero_data, mask_edges=masks[1])
    visualize_explanation_topk(graphs[2], trainer, convert_to_hetero_data,mask_edges=masks[2])
    visualize_explanation_topk(graphs[3], trainer, convert_to_hetero_data, mask_edges=masks[3])
