import matplotlib.pyplot as plt
import networkx as nx
import random
import pickle
from tqdm import tqdm

def plot_motif_graph(G, label, node_size=800):
    pos = nx.spring_layout(G, seed=42)

    # Color nodes by type
    color_map = []
    shape_map = {'Author': 's', 'Paper': 'o'}
    node_shapes = {'s': [], 'o': []}
    for n, d in G.nodes(data=True):
        ntype = d['type']
        color_map.append('lightgreen' if ntype == 'Author' else 'skyblue')
        node_shapes[shape_map[ntype]].append(n)

    # Create subplots for different shapes
    plt.figure(figsize=(8, 6))
    for shape, nodes in node_shapes.items():
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=node_size,
                               node_color=['lightgreen' if shape == 's' else 'skyblue'],
                               node_shape=shape)

    # Draw edges
    edge_colors = {'writes': 'black', 'cites': 'gray'}
    writes = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'writes']
    cites = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'cites']
    nx.draw_networkx_edges(G, pos, edgelist=writes, edge_color='black', arrows=True)
    nx.draw_networkx_edges(G, pos, edgelist=cites, edge_color='gray', arrows=True, style='dashed')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Add title
    titles = {
        0: "With trapezoid motif",
        1: "Without trapezoid motif",
    }
    plt.title(f"Motif Type {label}: {titles[label]}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Load previously saved multi-class dataset
with open("trapezoid_dataset_with_noise.pkl", "rb") as f:
    graphs, labels = pickle.load(f)

# Plot one example per motif class
shown = set()
for i, (g, label) in enumerate(zip(graphs, labels)):
    if label not in shown:
        plot_motif_graph(g, label)
        shown.add(label)
    if len(shown) == 5:
        break
