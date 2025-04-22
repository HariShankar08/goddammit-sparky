import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import random

# 1. Load the dataset
print("Loading dataset...")
with open("trapezoid_dataset_with_noise.pkl", 'rb') as f:
    graphs, labels = pickle.load(f)

print(f"Loaded {len(graphs)} graphs with labels distribution: {np.bincount(labels)}")

# 2. Convert NetworkX graphs to PyTorch Geometric HeteroData objects
def convert_to_hetero_data(G):
    data = HeteroData()
    
    # Identify nodes by type
    author_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'Author']
    paper_nodes  = [n for n, d in G.nodes(data=True) if d['type'] == 'Paper']
    
    # Create mappings for node IDs
    author_mapping = {n: i for i, n in enumerate(author_nodes)}
    paper_mapping  = {n: i for i, n in enumerate(paper_nodes)}

    data['author'].x = torch.ones((len(author_nodes), 64)) * 0.1
    print(f"Author nodes: {len(author_nodes)}, Paper nodes: {len(paper_nodes)}")
    data['paper'].x  = torch.ones((len(paper_nodes), 64)) * 0.1
    
    # Add identity information (one-hot style) for a few dimensions
    for i in range(len(author_nodes)):
        if i < 64:  # Ensure we do not run out of dimensions
            data['author'].x[i, i] = 1.0
    
    for i in range(len(paper_nodes)):
        if i < 64:
            data['paper'].x[i, i] = 1.0

    # Extract and add edges for the 'writes' and 'cites' relations
    writes_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'writes']
    cites_edges  = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'cites']
    
    if writes_edges:
        writes_src = [author_mapping[u] for u, v in writes_edges]
        writes_dst = [paper_mapping[v] for u, v in writes_edges]
        data['author', 'writes', 'paper'].edge_index = torch.tensor([writes_src, writes_dst])
    else:
        data['author', 'writes', 'paper'].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    if cites_edges:
        cites_src = [paper_mapping[u] for u, v in cites_edges if u in paper_mapping and v in paper_mapping]
        cites_dst = [paper_mapping[v] for u, v in cites_edges if u in paper_mapping and v in paper_mapping]
        data['paper', 'cites', 'paper'].edge_index = torch.tensor([cites_src, cites_dst])
    else:
        data['paper', 'cites', 'paper'].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # --- Add Self-Loop for author nodes ---
    if author_nodes:
        self_loop_edges = [(n, n) for n in author_nodes]
        self_loop_src = [author_mapping[u] for u, v in self_loop_edges]
        self_loop_dst = [author_mapping[v] for u, v in self_loop_edges]
        data['author', 'self_loop', 'author'].edge_index = torch.tensor([self_loop_src, self_loop_dst])
    else:
        data['author', 'self_loop', 'author'].edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return data

# Convert all graphs
hetero_data_list = []
for i, G in enumerate(graphs):
    try:
        hetero_data = convert_to_hetero_data(G)
        # Store the graph-level label
        hetero_data.y = torch.tensor([labels[i]], dtype=torch.long)
        hetero_data_list.append(hetero_data)
    except Exception as e:
        print(f"Error converting graph {i}: {e}")

print(f"Successfully converted {len(hetero_data_list)} graphs")

# Split dataset into training and testing sets
random.seed(42)
random.shuffle(hetero_data_list)
split_idx = int(0.8 * len(hetero_data_list))
train_dataset = hetero_data_list[:split_idx]
test_dataset  = hetero_data_list[split_idx:]

print(f"Training graphs: {len(train_dataset)}, Test graphs: {len(test_dataset)}")

# Prepare data loaders using PyG's DataLoader
from torch_geometric.loader import DataLoader
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8)

# Define the heterogeneous GNN model with self-loop update for authors.
class HeteroGNN(nn.Module):
    def __init__(self, hidden_dim=64, out_dim=2):
        super(HeteroGNN, self).__init__()
        
        # First layer: include 'writes', 'cites', and self-loop for 'author'
        self.conv1 = HeteroConv({
            ('author', 'writes', 'paper'): SAGEConv(64, hidden_dim),
            ('paper', 'cites', 'paper'):   SAGEConv(64, hidden_dim),
            ('author', 'self_loop', 'author'): SAGEConv(64, hidden_dim)
        }, aggr='sum')
        
        # Second layer: same idea, include self-loop for authors
        self.conv2 = HeteroConv({
            ('author', 'writes', 'paper'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ('paper', 'cites', 'paper'):   SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ('author', 'self_loop', 'author'): SAGEConv((hidden_dim, hidden_dim), hidden_dim)
        }, aggr='sum')
        
        # Fully connected layers for final graph-level classification.
        # We concatenate pooled features for 'author' and 'paper' types.
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, x_dict, edge_index_dict, batch_dict):
        # First convolution layer with ReLU activation
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        # Second convolution layer with ReLU activation
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        
        # Global mean pooling per node type; pooling separately for 'author' and 'paper'
        if 'author' in batch_dict:
            x_author = global_mean_pool(x_dict['author'], batch_dict['author'])
        else:
            x_author = torch.zeros(batch_dict['paper'].max().item() + 1, x_dict['paper'].size(1)).to(x_dict['paper'].device)
        x_paper = global_mean_pool(x_dict['paper'], batch_dict['paper'])
        
        # Concatenate the pooled features
        x = torch.cat([x_author, x_paper], dim=1)
        
        # Classification MLP head
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Set up device, model, optimizer and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = HeteroGNN(hidden_dim=64, out_dim=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Training function
def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
        
        # Get graph-level labels from the batch
        y = torch.cat([data.y for data in batch.to_data_list()]).to(device)
        
        # Compute loss and update weights
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Testing function
def test(loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass
            out = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
            
            # Extract graph-level labels
            y = torch.cat([data.y for data in batch.to_data_list()]).to(device)
            
            # Get predictions
            _, preds = torch.max(out, 1)
            
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    accuracy = correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    return accuracy, precision, recall, f1

# Training loop
print("Starting training...")
epochs = 5
train_losses = []
test_accuracies = []

for epoch in range(1, epochs + 1):
    loss = train()
    train_losses.append(loss)
    
    train_acc, train_prec, train_rec, train_f1 = test(train_loader)
    test_acc, test_prec, test_rec, test_f1 = test(test_loader)
    test_accuracies.append(test_acc)
    
    print(f'Epoch {epoch:02d} | Loss: {loss:.4f}')
    print(f'Train - Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Rec: {train_rec:.4f}, F1: {train_f1:.4f}')
    print(f'Test  - Acc: {test_acc:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}, F1: {test_f1:.4f}')

# Plot training loss and test accuracy curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig('hgnn_trapezoid.png')
plt.show()

# Final evaluation
final_test_acc, final_test_prec, final_test_rec, final_test_f1 = test(test_loader)
print("\nFinal Test Results:")
print(f"Accuracy: {final_test_acc:.4f}")
print(f"Precision: {final_test_prec:.4f}")
print(f"Recall: {final_test_rec:.4f}")
print(f"F1 Score: {final_test_f1:.4f}")
