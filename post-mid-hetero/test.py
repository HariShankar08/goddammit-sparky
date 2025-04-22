import pickle
import torch
from torch_geometric.data import HeteroData

# Load the dataset
with open('data.pkl', 'rb') as f:
    graphs, labels, masks = pickle.load(f)
from gsat_hetero import convert_to_hetero_data
# Verify edge indices
errors = []
for idx, G in enumerate(graphs):
    data = convert_to_hetero_data(G, masks[idx])
    # Check each relation's edge_index bounds against node feature sizes
    for rel, edge_index in data.edge_index_dict.items():
        # Determine number of nodes of source type
        src_type = rel[0]
        num_src_nodes = data[src_type].x.size(0)
        if edge_index.numel() == 0:
            continue
        max_index = int(edge_index.max())
        if max_index >= num_src_nodes or edge_index.min() < 0:
            errors.append(f"Graph {idx}, relation {rel}: index out-of-bounds (max={max_index}, num_src={num_src_nodes})")

if errors:
    print("Found index errors:")
    for e in errors:
        print("  -", e)
else:
    print("All edge_index tensors are within valid bounds.")
