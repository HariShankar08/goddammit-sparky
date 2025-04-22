import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool, MessagePassing
from torch_geometric.data import HeteroData, DataLoader
from sklearn.model_selection import train_test_split

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
    def __init__(self, in_dim, hidden, out_dim, relations):
        super().__init__()
        self.relations = relations
        print(22)
        # Convert tuple relations to string keys for ModuleDict
        self.edge_mlps = nn.ModuleDict({
            '_'.join(rel): MLP([2*in_dim, hidden, 1]) for rel in relations
        })
        print(23)
        
        # First layer convs
        self.conv1 = HeteroConv({
            rel: SAGEConv(in_dim, hidden) for rel in relations
        }, aggr='sum')
        
        # Second layer convs with skip
        self.conv2 = HeteroConv({
            rel: SAGEConv(hidden, hidden) for rel in relations
        }, aggr='sum')
        
        # Should be ModuleDict instead of ParameterDict since we're storing modules
        self.skip = nn.ModuleDict({
            'author': nn.Linear(in_dim, hidden),
            'paper': nn.Linear(in_dim, hidden)
        })
        
        self.cls = MLP([2*hidden, hidden, out_dim])

    def forward(self, x_dict, edge_index_dict, edge_attn=None):
        # Compute edge attention scores
        #print(49, in_dim)
        print(50, x_dict)
        print(51, edge_index_dict)
        if edge_attn is not None:
            print(52, edge_attn)

        # alpha_dict = {}
        # if edge_attn is None:
        #     for rel in self.relations:
        #         #print(58, rel)
        #         # Skip if relation doesn't exist in edge_index_dict
        #         if rel not in edge_index_dict or edge_index_dict[rel].size(1) == 0:
        #             continue
                    
        #         src_type, _, dst_type = rel
        #         src, dst = edge_index_dict[rel]
        #         h_src = x_dict[src_type][src]
        #         h_dst = x_dict[dst_type][dst]
        #         # Convert tuple key to string for ModuleDict lookup

        #         s = self.edge_mlps['_'.join(rel)](torch.cat([h_src, h_dst], -1)).squeeze()
        #         print(64, s)
        #         alpha_dict[rel] = torch.sigmoid(s)
        # else:
        #     alpha_dict = edge_attn
        #     print(74, alpha_dict)
        
        # Layer 1 with masked edges
        h1_dict = self.conv1(x_dict, edge_index_dict)
        # h1_dict = self.conv1(x_dict, edge_index_dict)
        print(78, h1_dict)
        h1_dict = {k: F.relu(v) for k,v in h1_dict.items()}
        print(80, h1_dict)
        
        # Layer 2 with skip
        h2_dict = self.conv2(h1_dict, edge_index_dict)
        # h2_dict = self.conv2(h1_dict, edge_index_dict)

        print(83, h2_dict)
        h2_dict = {
            k: F.relu(v + self.skip[k](x_dict[k])) 
            for k,v in h2_dict.items() if k in self.skip
        }
        print(89, h2_dict)
        
        # Readout - use batch index if available, otherwise assume batch size 1
        author_batch = torch.zeros(h2_dict['author'].size(0), dtype=torch.long, device=h2_dict['author'].device)
        print(82, h2_dict['author'].size(0), h2_dict['author'].device)
        paper_batch = torch.zeros(h2_dict['paper'].size(0), dtype=torch.long, device=h2_dict['paper'].device)
        print(84, h2_dict['paper'].size(0), h2_dict['paper'].device)

        
        author_pool = global_mean_pool(h2_dict['author'], author_batch)
        paper_pool = global_mean_pool(h2_dict['paper'], paper_batch)
        print(87, "pooling done")
        alpha_dict = {}
        return self.cls(torch.cat([author_pool, paper_pool], -1)), alpha_dict


class GSATTrainer:
    def __init__(self, model, lr=0.01, gamma_dict={'writes':0.5, 'cites':0.5, 'self_loop':0.5}, temp=1.0):
        self.model = model
        self.optim = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma_dict = gamma_dict  # Relation-specific gamma
        self.temp = temp

    def sample_edges(self, data, train=True):
        """Sample edges using Gumbel-Bernoulli with straight-through estimator"""
        edge_masks = {}
        alpha_dict = {}
        
        # 1. Compute edge probabilities
        for rel in self.model.relations:
            if rel not in data.edge_index_dict or data.edge_index_dict[rel].size(1) == 0:
                continue
                
            # Get node features for this relation
            src_type, _, dst_type = rel
            src, dst = data.edge_index_dict[rel]
            h_src = data.x_dict[src_type][src]
            h_dst = data.x_dict[dst_type][dst]
            
            # Compute logits through MLP
            s = self.model.edge_mlps['_'.join(rel)](torch.cat([h_src, h_dst], -1)).squeeze()
            logits = torch.log(torch.sigmoid(s) + 1e-10)  # log(p/(1-p))
            
            # 2. Sample using Gumbel-Bernoulli
            if train:
                # Gumbel noise
                u = torch.rand_like(logits)
                g = -torch.log(-torch.log(u + 1e-10) + 1e-10)
                
                # Temperature-scaled sampling
                threshold = torch.sigmoid((logits + g) / self.temp)
                hard_mask = (threshold >= 0.5).float()
                
                # Straight-through estimator
                edge_mask = hard_mask - threshold.detach() + threshold
            else:
                # Deterministic for evaluation
                edge_mask = (torch.sigmoid(logits) >= 0.5).float()

            edge_masks[rel] = edge_mask
            alpha_dict[rel] = torch.sigmoid(logits)  # Store probabilities for KL
            
        # 3. Create masked edge indices
        masked_edges = {}
        for rel in data.edge_index_dict:
            if rel not in edge_masks:
                masked_edges[rel] = data.edge_index_dict[rel]
                continue
                
            mask = edge_masks[rel].bool().squeeze()
            masked_edges[rel] = data.edge_index_dict[rel][:, mask]
            
        return masked_edges, alpha_dict

    def kl_loss(self, alpha, rel_type):
        gamma = self.gamma_dict.get(rel_type, 0.5)
        return (alpha * torch.log(alpha/gamma + 1e-10) + 
               (1-alpha) * torch.log((1-alpha)/(1-gamma) + 1e-10)).mean()

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0
        
        for data in loader:
            # 1. Sample subgraph
            masked_edges, alpha_dict = self.sample_edges(data, train=True)
            
            # 2. Forward pass with masked edges
            logits, _= self.model(data.x_dict, masked_edges)
            
            # 3. Calculate losses
            loss_cls = F.cross_entropy(logits, data.y)
            loss_kl = sum(self.kl_loss(alpha, rel[1]) for rel, alpha in alpha_dict.items())
            
            # 4. Optimize
            loss = loss_cls + loss_kl
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            
            total_loss += loss.item()
            
        return total_loss / len(loader)

    def test(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data in loader:
                masked_edges, _ = self.sample_edges(data, train=False)
                logits, _ = self.model(data.x_dict, masked_edges)
                pred = logits.argmax(dim=-1)
                correct += (pred == data.y).sum().item()
                total += data.y.size(0)
                
        return correct / total if total > 0 else 0.0

# Example Usage
from simple_motif import convert_to_hetero_data, generate_simple_dataset_with_noise
import pickle
# 1. Generate and load dataset
generate_simple_dataset_with_noise("hi.pkl", 50)
with open("hi.pkl", 'rb') as f:
    graphs, labels, masks = pickle.load(f)

# 2. Convert to HeteroData list
data_list = []
for G, y in zip(graphs, labels):
    data = convert_to_hetero_data(G)
    data.y = torch.tensor(y, dtype=torch.long)
    data_list.append(data)

# 3. Train-test split
train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)

# 4. Create data loaders
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1)

# 5. Verify feature dimensions (adjust based on your convert_to_hetero_data)
sample_data = data_list[0]
in_dim = sample_data['author'].x.size(1)  # Should match for paper nodes too

# 6. Initialize model and trainer
model = GSAT_HeteroGNN(
    in_dim=in_dim,
    hidden=128,
    out_dim=2,
    relations=[
        ('author','writes','paper'),
        ('paper','cites','paper')
        # ('author','self_loop','author')
    ]
)
trainer = GSATTrainer(model, gamma_dict = {'writes':0.5, 'cites':0.5})

# 7. Training loop
for epoch in range(1):
    loss = trainer.train_epoch(train_loader)
    acc = trainer.test(test_loader)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Acc={acc:.4f}")
