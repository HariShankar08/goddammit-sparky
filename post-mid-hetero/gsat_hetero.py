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
        # Ensure edge_index is 2D (2 x num_edges)
        data['author','writes','paper'].edge_index = torch.tensor([src,dst], dtype=torch.long)
        # optional mask per edge
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
        if valid_cites:  # Only proceed if there are valid edges
            src = [p_map[u] for u,v in valid_cites]
            dst = [p_map[v] for u,v in valid_cites]
            # Ensure edge_index is 2D (2 x num_edges)
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
# 2) GSAT‑style HeteroGNN (refactored)
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

        # per‐relation edge‐MLPs
        self.edge_mlps = nn.ModuleDict({
            '_'.join(rel): MLP([2*in_dim, hidden, 1])
            for rel in relations
        })

        # two layers of HeteroConv
        self.conv1 = HeteroConv({
            rel: SAGEConv(in_dim,  hidden) for rel in relations
        }, aggr='sum')
        self.conv2 = HeteroConv({
            rel: SAGEConv(hidden, hidden) for rel in relations
        }, aggr='sum')
        # final classifier
        self.skip = nn.ModuleDict({
            ntype: nn.Linear(in_dim, hidden)
            for ntype in {'author','paper'}
        })
        self.cls = MLP([2*hidden, hidden, out_dim])

    def forward(self, x_dict, edge_index_dict, edge_attn=None):
        # (a) sample_edges → masked_edges, alpha_dict lives in Trainer
        # (b) here we just run convs & pool

        h1 = self.conv1(x_dict, edge_index_dict)  
        h1 = {k: F.relu(v) for k,v in h1.items()}

        # 2) Layer 2 + skip
        h2 = self.conv2(h1, edge_index_dict)
        # add skip only for the types we have skips for
        for ntype in self.skip:
            if ntype in h2:
                h2[ntype] = F.relu(h2[ntype] + self.skip[ntype](x_dict[ntype]))

        # global mean pool (batch=all nodes as one graph)
        
        a_batch = x_dict['author'].new_zeros(x_dict['author'].size(0), dtype=torch.long)
        p_batch = x_dict['paper'].new_zeros(x_dict['paper'].size(0),  dtype=torch.long)
        # if you’re using batch_size>1, swap the above for `data['author'].batch`

        a_pool = global_mean_pool(h2['author'], a_batch)
        p_pool = global_mean_pool(h2['paper'],  p_batch)

        logits = self.cls(torch.cat([a_pool, p_pool], dim=-1))
        # print(123, logits.shape)
        return logits, {}

# -----------------------------------------
# 3) Your GSATTrainer (unchanged logic)
# -----------------------------------------
class GSATTrainer:
    def __init__(self, model, lr=0.01, gamma_dict=None, temp=1.0):
        self.model      = model
        self.optim      = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma_dict = gamma_dict or {}
        self.temp       = temp

    def sample_edges(self, data, train=True):
        edge_masks = {}
        alpha_dict = {}
        for rel in self.model.relations:
            if rel not in data.edge_index_dict or data.edge_index_dict[rel].size(1) == 0:
                continue
            
            edge_index = data.edge_index_dict[rel]
            # Check and fix edge_index dimensions if needed
            if edge_index.dim() != 2 or edge_index.size(0) != 2:
                print(f"Warning: Edge index for {rel} has incorrect shape {edge_index.shape}. Attempting to reshape.")
                if edge_index.dim() > 2:
                    edge_index = edge_index.view(2, -1)
                data.edge_index_dict[rel] = edge_index
                
            src_t, _, dst_t = rel
            src, dst = edge_index
            
            h_src = data.x_dict[src_t][src]
            h_dst = data.x_dict[dst_t][dst]

            s = self.model.edge_mlps['_'.join(rel)](torch.cat([h_src,h_dst],-1)).squeeze()
            logits = torch.log(torch.sigmoid(s)+1e-10)

            if train:
                u = torch.rand_like(logits)
                g = -torch.log(-torch.log(u+1e-10)+1e-10)
                thresh = torch.sigmoid((logits+g)/self.temp)
                hard = (thresh>=0.8).float()
                mask = hard - thresh.detach() + thresh
            else:
                mask = torch.sigmoid(logits).float()

            edge_masks[rel] = mask
            alpha_dict[rel] = torch.sigmoid(logits)

        # build masked_edges
        masked_edges = {}
        for rel, idx in data.edge_index_dict.items():
            # Ensure idx is 2D
            if idx.dim() != 2 or idx.size(0) != 2:
                if idx.dim() > 2:
                    idx = idx.view(2, -1)
                    
            if rel in edge_masks:
                m = edge_masks[rel].bool().squeeze()
                masked_edges[rel] = idx[:, m]
            else:
                masked_edges[rel] = idx
                
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
            # Debug prints to understand the data structure
            # print("X dict shapes:", {k: v.shape for k, v in data.x_dict.items()})
            # print("Edge dict shapes:", {k: v.shape for k, v in data.edge_index_dict.items()})
            
            me, alph = self.sample_edges(data, train=True)
            
            # Debug prints for masked edges
            # print("Masked edge shapes:", {k: v.shape for k, v in me.items()})
            
            # Forward pass
            logits, _ = self.model(data.x_dict, me)
            
            # Ensure label has right shape
            if data.y.dim() == 0:
                data.y = data.y.unsqueeze(0)
            data.y = data.y.long()  # Use long type for cross entropy
            
            # Calculate loss
            task_loss = F.cross_entropy(logits, data.y)
            reg_loss = sum(self.kl_loss(a, r[1]) for r, a in alph.items() if a.numel() > 0)
            loss = task_loss + reg_loss
            
            # Optimization step
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
        return correct/tot if tot else 0.0

# -----------------------------------------
# 4) Example usage
# -----------------------------------------
if __name__ == "__main__":
    # pretend you have graphs,labels pickled
    from simple_motif import generate_simple_dataset_with_noise
    generate_simple_dataset_with_noise("hi.pkl",50)
    with open("hi.pkl","rb") as f:
        graphs, labels, _ = pickle.load(f)

    data_list = []
    for G,y in zip(graphs,labels):
        d = convert_to_hetero_data(G)
        d.y = torch.tensor(y, dtype=torch.long)
        data_list.append(d)

    train, test = train_test_split(data_list, test_size=0.2, random_state=42)
    train_loader = DataLoader(train, batch_size=1, shuffle=True)
    test_loader  = DataLoader(test, batch_size=1)

    in_dim = data_list[0]['author'].x.size(1)
    rels   = [
      ('author','writes','paper'),
      ('paper','cites','paper'),
      ('author','self_loop','author'),
      ('paper','self_loop','paper')
    ]
    model   = GSAT_HeteroGNN(in_dim, 128, 2, rels, temp=1.0)
    trainer = GSATTrainer(model, lr=0.001,
                          gamma_dict={'writes':0.7,'cites':0.5}, temp=1.0)

    for epoch in range(10):
        loss = trainer.train_epoch(train_loader)
        acc  = trainer.test(test_loader)
        print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")
