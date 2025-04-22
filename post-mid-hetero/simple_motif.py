import networkx as nx
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool
from torch_geometric.data import HeteroData, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os
import sys
from datetime import datetime

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # For reproducibility
    torch.backends.cudnn.benchmark = False

def add_nodes(G, num_authors, num_papers, offset=0):
    author_ids = list(range(offset, offset + num_authors))
    paper_ids = list(range(offset + num_authors, offset + num_authors + num_papers))
    for aid in author_ids:
        G.add_node(aid, type='Author')
    for pid in paper_ids:
        G.add_node(pid, type='Paper')
    return author_ids, paper_ids


def generate_simple_motif_graph():
    G = nx.MultiDiGraph()
    authors, papers = add_nodes(G, 2, 1)
    a1, a2 = authors
    # print(papers)
    p1 = papers[0]
    motif_edges = {
        (a1, p1), (a2, p1)
    }
    for u, v in motif_edges:
        edge_type = 'writes' if G.nodes[u]['type'] == 'Author' else 'cites'
        G.add_edge(u, v, type=edge_type)
    G.graph['motif_mask'] = motif_edges
    return G



def contains_simple_motif(G):
    authors = [n for n, d in G.nodes(data=True) if d['type'] == 'Author']
    papers = [n for n, d in G.nodes(data=True) if d['type'] == 'Paper']
    
    for a1 in authors:
        for a2 in authors:
            if a1 == a2:
                continue
            for p1 in papers:
                # Both a1 and a2 write p1
                if G.has_edge(a1, p1) and G.has_edge(a2, p1):
                    # p1 should cite some p2 written by a1
                    return True
    return False
   

# ----- add noise -------- 
def add_random_noise(G, num_extra_authors=2, num_extra_papers=2, max_writes=2, max_cites=3, motif_checkers=[]):
    offset = max(G.nodes) + 1 if G.nodes else 0
    authors, papers = add_nodes(G, num_extra_authors, num_extra_papers, offset)

    old_authors = [n for n, d in G.nodes(data=True) if d['type'] == 'Author' and n not in authors]
    old_papers = [n for n, d in G.nodes(data=True) if d['type'] == 'Paper' and n not in papers]

    for a in authors:
        potential_papers = old_papers + papers
        for p in random.sample(potential_papers, random.randint(1, max_writes)):
            G.add_edge(a, p, type='writes')

    all_papers = old_papers + papers
    for i in range(len(all_papers)):
        for j in range(len(all_papers)):
            if i != j and random.random() < 0.4:
                G.add_edge(all_papers[i], all_papers[j], type='cites')

    for checker in motif_checkers:
        if checker(G):
            # print("checker", checker.__name__, "found a motif after adding noise")
            return False

    return True
def generate_random_non_motif_graph():
    while True:
        G = nx.MultiDiGraph()
        authors, papers = add_nodes(G, 2, 3)
        G.graph['motif_mask'] = set()
        for a in authors:
            for p in random.sample(papers, random.randint(1, 3)):
                G.add_edge(a, p, type='writes')
        for i in range(len(papers)):
            for j in range(len(papers)):
                if i != j and random.random() < 0.3:
                    G.add_edge(papers[i], papers[j], type='cites')
        # if not contains_trapezoid_motif(G):
        if not contains_simple_motif(G):
            return G



def generate_simple_dataset_with_noise(filename="simple_dataset_with_noise.pkl", max_per_class=500):
    graphs, labels, masks = [], [], []
    label_counts = {0: 0, 1: 0}
    for _ in tqdm(range(10000)):
        if label_counts[0] < max_per_class:
            G = generate_simple_motif_graph()
            if add_random_noise(G, motif_checkers=[]):
                graphs.append(G)
                labels.append(0)
                masks.append(G.graph['motif_mask'])
                label_counts[0] += 1
        elif label_counts[1] < max_per_class:
            G = generate_random_non_motif_graph()
            # print(G.nodes(data=True))
            if add_random_noise(G, motif_checkers=[contains_simple_motif]):
                graphs.append(G)
                labels.append(1)
                masks.append(set())
                label_counts[1] += 1
        else:
            break
    with open(filename, 'wb') as f:
        pickle.dump((graphs, labels, masks), f)
    return label_counts

def convert_to_hetero_data(G, mask=None):
    data = HeteroData()
    author_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'Author']
    paper_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'Paper']
    author_mapping = {n: i for i, n in enumerate(author_nodes)}
    paper_mapping = {n: i for i, n in enumerate(paper_nodes)}
    data['author'].x = torch.eye(len(author_nodes), 64)
    data['paper'].x = torch.eye(len(paper_nodes), 64)

    writes_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'writes']
    cites_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'cites']

    if writes_edges:
        writes_src = [author_mapping[u] for u, v in writes_edges]
        writes_dst = [paper_mapping[v] for u, v in writes_edges]
        data['author', 'writes', 'paper'].edge_index = torch.tensor([writes_src, writes_dst])
        
        edge_mask = torch.zeros(len(writes_edges))
        if mask:
            for idx, (src, dst) in enumerate(writes_edges):
                if (src, dst) in mask:
                    edge_mask[idx] = 1.0
        data['author', 'writes', 'paper'].edge_mask = edge_mask
    else:
        data['author', 'writes', 'paper'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['author', 'writes', 'paper'].edge_mask = torch.empty((0,), dtype=torch.float32)

    # Add 'cites' edges
    if cites_edges:
        cites_src = [paper_mapping[u] for u, v in cites_edges if u in paper_mapping and v in paper_mapping]
        cites_dst = [paper_mapping[v] for u, v in cites_edges if u in paper_mapping and v in paper_mapping]
        data['paper', 'cites', 'paper'].edge_index = torch.tensor([cites_src, cites_dst])

        edge_mask = torch.zeros(len(cites_edges))
        if mask:
            for idx, (src, dst) in enumerate(cites_edges):
                if (src, dst) in mask:
                    edge_mask[idx] = 1.0
        data['paper', 'cites', 'paper'].edge_mask = edge_mask
    else:
        data['paper', 'cites', 'paper'].edge_index = torch.empty((2, 0), dtype=torch.long)
        data['paper', 'cites', 'paper'].edge_mask = torch.empty((0,), dtype=torch.float32)


    data['author', 'self_loop', 'author'].edge_index = torch.stack([torch.arange(len(author_nodes))]*2)
    # not including this in the mask ... 
    return data


class ExplainableHeteroGNN(nn.Module):
    def __init__(self, hidden_dim1 = 64, hidden_dim2=64, out_dim=2, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim2

        self.conv1 = HeteroConv({
            ('author', 'writes', 'paper'): SAGEConv(hidden_dim1, hidden_dim2),
            ('paper', 'cites', 'paper'): SAGEConv(hidden_dim1, hidden_dim2),
            ('author', 'self_loop', 'author'): SAGEConv(hidden_dim1, hidden_dim2)
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('author', 'writes', 'paper'): SAGEConv(hidden_dim2, hidden_dim2),
            ('paper', 'cites', 'paper'): SAGEConv(hidden_dim2, hidden_dim2),
            ('author', 'self_loop', 'author'): SAGEConv(hidden_dim2, hidden_dim2)
        }, aggr='sum')
        
        # Skip connection
        self.skip = nn.ModuleDict({
            'author': nn.Linear(hidden_dim1, hidden_dim2),
            'paper': nn.Linear(hidden_dim1, hidden_dim2)
        })
        
        # Edge attention mechanism with richer parameterization
        self.att_writes = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )
        
        self.att_cites = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )
        
        # Graph-level classification
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim2 * 2, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, out_dim)
        )
        # Global context for structure-aware attention
        self.global_context = nn.Parameter(torch.randn(hidden_dim2))
        # Temperature parameter for sharpening attention
        self.temperature = nn.Parameter(torch.ones(1))

    def compute_attention_scores(self, h_src, h_dst, edge_type):
        h_cat = torch.cat([h_src, h_dst], dim=-1)  # [E, 2H]
        
        if edge_type == ('author', 'writes', 'paper'):
            scores = self.att_writes(h_cat).view(-1)  # [E]
        else:  # ('paper', 'cites', 'paper')
            scores = self.att_cites(h_cat).view(-1)  # [E]
            
        # Structure awareness: Add dot product with global context
        # what does the global context vector train on: during loss.backward, and .step() all gradients propogate
        # to all the parameters: so it comes here. 
        structure_score = (h_src * self.global_context).sum(dim=-1) + (h_dst * self.global_context).sum(dim=-1)
        scores = scores + 0.1 * structure_score
        
        # Apply temperature for sharper attention
        scores = scores / (self.temperature + 1e-6)
        
        return scores

    def forward(self, x_dict, edge_index_dict, batch_dict, return_attention=False, edge_probs=None):
        x_dict_skip = {k: self.skip[k](v) for k, v in x_dict.items()}
        x_dict = {k: F.relu(v) for k, v in self.conv1(x_dict, edge_index_dict).items()}
        x_dict = {k: F.relu(v) + x_dict_skip[k] for k, v in self.conv2(x_dict, edge_index_dict).items()}
        
        if batch_dict is not None:
            x_author = global_mean_pool(x_dict['author'], batch_dict['author'])
            x_paper = global_mean_pool(x_dict['paper'], batch_dict['paper'])
        else:
            x_author = x_dict['author'].mean(dim=0, keepdim=True)
            x_paper = x_dict['paper'].mean(dim=0, keepdim=True)
    
        class_logits = self.fc(torch.cat([x_author, x_paper], dim=-1))
        
        if return_attention:
            edge_scores = []
            edge_tuples = []
            
            if ('author', 'writes', 'paper') in edge_index_dict:
                src, dst = edge_index_dict[('author', 'writes', 'paper')]
                h_src = x_dict['author'][src]
                h_dst = x_dict['paper'][dst]
                scores = self.compute_attention_scores(h_src, h_dst, ('author', 'writes', 'paper'))
                edge_scores.append(scores)
                edge_tuples += [('author', 'writes', 'paper')] * scores.shape[0]
            
            if ('paper', 'cites', 'paper') in edge_index_dict:
                src, dst = edge_index_dict[('paper', 'cites', 'paper')]
                h_src = x_dict['paper'][src]
                h_dst = x_dict['paper'][dst]
                scores = self.compute_attention_scores(h_src, h_dst, ('paper', 'cites', 'paper'))
                edge_scores.append(scores)
                edge_tuples += [('paper', 'cites', 'paper')] * scores.shape[0]
            
            final_scores = torch.cat(edge_scores, dim=0).sigmoid() if edge_scores else torch.tensor([])
            
            if edge_probs is not None and final_scores.numel() > 0:
                final_scores = final_scores * edge_probs
            
            return class_logits, final_scores, edge_tuples
        else:
            return class_logits

def train_model(alpha_sparse=0.005, alpha_entropy=0.001, alpha_mask=0.0):
    with open("simple_dataset_with_noise.pkl", 'rb') as f:
        graphs, labels, masks = pickle.load(f)

    hetero_data_list = []
    for G, label, mask in zip(graphs, labels, masks):
        data = convert_to_hetero_data(G, mask)
        data.y = torch.tensor(label, dtype=torch.long)
        hetero_data_list.append(data)

    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(hetero_data_list, test_size=0.2, stratify=[d.y.item() for d in hetero_data_list])
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ExplainableHeteroGNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    best_acc = 0
    patience_counter = 0
    
    for epoch in range(1, 20):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            class_logits, edge_preds, _ = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, return_attention=True)

            loss_cls = F.cross_entropy(class_logits, batch.y)

            writes_len = batch['author', 'writes', 'paper'].edge_index.size(1) if ('author', 'writes', 'paper') in batch.edge_types else 0
            cites_len = batch['paper', 'cites', 'paper'].edge_index.size(1) if ('paper', 'cites', 'paper') in batch.edge_types else 0
            
            loss_mask = 0
            if writes_len > 0:
                edge_pred_writes = edge_preds[:writes_len]
                loss_mask += F.binary_cross_entropy(edge_pred_writes, batch['author', 'writes', 'paper'].edge_mask.to(device))
            
            if cites_len > 0:
                edge_pred_cites = edge_preds[writes_len:writes_len + cites_len]
                loss_mask += F.binary_cross_entropy(edge_pred_cites, batch['paper', 'cites', 'paper'].edge_mask.to(device))

            # Continuity/structure loss - encourage connected explanations
            loss_continuity = 0
            if writes_len > 0:
                edge_pred_writes = edge_preds[:writes_len].view(-1)
                loss_continuity += torch.var(edge_pred_writes)
            
            # GNNExplainer-style regularization (sparse and non-trivial)
            edge_preds = edge_preds.clamp(min=1e-6, max=1-1e-6)
            loss_sparse = edge_preds.sum() / edge_preds.size(0)  # Sparsity
            entropy = -(edge_preds * torch.log(edge_preds) + (1-edge_preds) * torch.log(1-edge_preds))
            loss_entropy = entropy.mean()  # Non-trivial explanations

            # Combine losses with balanced weights
            loss = loss_cls + alpha_mask * loss_mask + alpha_sparse * loss_sparse + alpha_entropy * loss_entropy + 0.01 * loss_continuity
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluation
        model.eval()
        y_true, y_pred = [], []
        for batch in test_loader:
            batch = batch.to(device)
            with torch.no_grad():
                class_logits = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
                preds = class_logits.argmax(dim=-1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        acc_cls = accuracy_score(y_true, y_pred)
        print(f"Epoch {epoch}: Loss={total_loss / len(train_loader):.4f}, Acc={acc_cls:.4f}")
        
        scheduler.step(total_loss / len(train_loader))
        
        # Early stopping
        if acc_cls > best_acc:
            best_acc = acc_cls
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:  # 5 epochs without improvement
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load("best_model.pt"))
    return model

# explanation threshold approach
def visualize_explanation(model, graph, method='otsu'):
    data = convert_to_hetero_data(graph)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # Get attention scores from the model
    model.eval()
    with torch.no_grad():
        _, edge_scores, edge_types = model(data.x_dict, data.edge_index_dict, None, return_attention=True)

    if edge_scores.numel() > 0:
        edge_scores = edge_scores.cpu().numpy()
        
        # Adaptive thresholding based on method
        if method == 'otsu':
            # Otsu's method - finds optimal threshold that minimizes intra-class variance
            from skimage.filters import threshold_otsu
            threshold = threshold_otsu(edge_scores)
        elif method == 'adaptive':
            # Use gap/elbow detection
            sorted_scores = np.sort(edge_scores)[::-1]  # Sort in descending order
            gaps = sorted_scores[:-1] - sorted_scores[1:]
            elbow_idx = np.argmax(gaps) + 1
            threshold = sorted_scores[elbow_idx]
        elif method == 'relative':
            # Use relative thresholding - select top scores that sum to 50% of total
            sorted_scores = np.sort(edge_scores)[::-1]  # Sort in descending order
            cumsum = np.cumsum(sorted_scores)
            threshold_idx = np.searchsorted(cumsum, 0.5 * cumsum[-1])
            threshold = sorted_scores[threshold_idx]
        else:
            # Fallback - use top 20% of edges
            threshold = np.quantile(edge_scores, 0.8)
            
        # print(f"Chosen threshold: {threshold:.4f}")

        # Match score order to edge order
        writes_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'writes']
        cites_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'cites']
        all_edges = writes_edges + cites_edges

        # Important edge mask (based on adaptive threshold)
        important_edges = [edge for i, edge in enumerate(all_edges)
                           if i < len(edge_scores) and edge_scores[i] > threshold]

        # Draw
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(graph, seed=42)
        author_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'Author']
        paper_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'Paper']
        nx.draw_networkx_nodes(graph, pos, nodelist=author_nodes, node_color='lightblue', label='Authors')
        nx.draw_networkx_nodes(graph, pos, nodelist=paper_nodes, node_color='lightgreen', label='Papers')
        nx.draw_networkx_edges(graph, pos, alpha=0.1) 
        edge_labels = {}
        for i, edge in enumerate(all_edges):
            if i < len(edge_scores):
                edge_labels[edge] = f"{edge_scores[i]:.2f}"

        if important_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=important_edges,
                                   edge_color='red', width=2, label='Important Edges')
            nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8)

        motif_edges = graph.graph.get('motif_mask', [])
        if motif_edges:
            nx.draw_networkx_edges(graph, pos, edgelist=motif_edges,
                                   edge_color='black', width=2, style='dashed', label='Ground Truth')

        nx.draw_networkx_labels(graph, pos)
        plt.title(f"Graph Explanation using {method} thresholding")
        plt.legend()
        plt.tight_layout()
        # plt.show()
        return plt.gcf()
    else:
        print("No edge scores available for visualization")

# Evaluate explanations with the model
def evaluate_explanation(model, graph, method='otsu'):
    motif_mask = graph.graph.get("motif_mask", set())
    if not motif_mask:
        print("⚠️ No ground truth motif_mask found in graph.")
        return {}

    # Convert motif edges to (u, v, type)
    true_edges = {(u, v, d['type']) for u, v, d in graph.edges(data=True) if (u, v) in motif_mask}

    data = convert_to_hetero_data(graph)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        _, edge_scores, edge_types = model(data.x_dict, data.edge_index_dict, None, return_attention=True)

    edge_scores = edge_scores.cpu().numpy()

    if method == 'otsu':
        from skimage.filters import threshold_otsu
        threshold = threshold_otsu(edge_scores)
    elif method == 'adaptive':
        sorted_scores = np.sort(edge_scores)[::-1]
        gaps = sorted_scores[:-1] - sorted_scores[1:]
        elbow_idx = np.argmax(gaps) + 1
        threshold = sorted_scores[elbow_idx]
    elif method == 'relative':
        sorted_scores = np.sort(edge_scores)[::-1]
        cumsum = np.cumsum(sorted_scores)
        threshold_idx = np.searchsorted(cumsum, 0.5 * cumsum[-1])
        threshold = sorted_scores[threshold_idx]
    else:
        # Fallback
        threshold = np.quantile(edge_scores, 0.8)

    writes_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'writes']
    cites_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'cites']
    all_edges = [(u, v, 'writes') for u, v in writes_edges] + [(u, v, 'cites') for u, v in cites_edges]

    predicted_edges = {all_edges[i] for i in range(len(edge_scores)) if edge_scores[i] > threshold}

    intersection = predicted_edges & true_edges
    precision = len(intersection) / len(predicted_edges) if predicted_edges else 0
    recall = len(intersection) / len(true_edges) if true_edges else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0

    return {
        "intersection": list(intersection),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": threshold,
        "num_pred_edges": len(predicted_edges),
        "num_true_edges": len(true_edges)
    }

def visualize_and_evaluate_explanations():
    model = train_model()
    with open("simple_dataset_with_noise.pkl", 'rb') as f:
        graphs, labels, _ = pickle.load(f)
    class0_graphs = [g for g, l in zip(graphs, labels) if l == 0]
    print(f"\n📦 Total class-0 graphs: {len(class0_graphs)}")
    methods = ['otsu', 'adaptive', 'relative']
    method_results = {method: [] for method in methods}
    
    sample_graphs = random.sample(class0_graphs, min(20, len(class0_graphs)))
    
    for method in methods:
        print(f"\n🔍 Evaluating with {method} thresholding:")
        for i, graph in enumerate(sample_graphs):
            result = evaluate_explanation(model, graph, method=method)
            method_results[method].append(result)
            # print(f"  Graph {i+1}/{len(sample_graphs)}: F1={result['f1']:.4f}, "
            #       f"P={result['precision']:.4f}, R={result['recall']:.4f}")
            if i < 3:  
                # Only visualize first 3
                visualize_explanation(model, graph, method=method)
    
    print("\n📊 Results Summary:")
    for method in methods:
        results = method_results[method]
        avg_prec = np.mean([r["precision"] for r in results])
        avg_rec = np.mean([r["recall"] for r in results])
        avg_f1 = np.mean([r["f1"] for r in results])
        thresh = np.mean([r["threshold"] for r in results])
        print(f"{method.upper()} Thresholding:")
        print(f"Threshold value: {thresh:.4f}")
        print(f"  Precision: {avg_prec:.4f}")
        print(f"  Recall:    {avg_rec:.4f}")
        print(f"  F1 Score:  {avg_f1:.4f}")
    
    best_method = max(methods, key=lambda m: np.mean([r["f1"] for r in method_results[m]]))
    print(f"\n🏆 Best method: {best_method}")
    return model, best_method

# if __name__ == "__main__":

#     # seed_values = [42, 123, 456, 789, 101112]
#     seed_values = [42]
#     for seed in seed_values:
#         print("SEED IS", seed)
#         set_seed(seed)
#         generate_simple_dataset_with_noise()
#         visualize_and_evaluate_explanations()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--method', type=str, choices=['otsu', 'adaptive', 'relative'], default='otsu')
    parser.add_argument('--alpha_sparse', type=float, default=0.005)
    parser.add_argument('--alpha_entropy', type=float, default=0.001)
    parser.add_argument('--alpha_mask', type=float, default=0.0)
    parser.add_argument('--outdir', type=str, default="results")

    args = parser.parse_args()

    set_seed(args.seed)

    tag = f"seed{args.seed}_method_{args.method}_s{args.alpha_sparse}_e{args.alpha_entropy}_m{args.alpha_mask}"
    img_dir = os.path.join(args.outdir, tag, "images")
    os.makedirs(img_dir, exist_ok=True)

    log_file = os.path.join(args.outdir, tag, f"output_log.txt")
    sys.stdout = open(log_file, 'w')

    print(f"📌 Experiment: {tag}")
    generate_simple_dataset_with_noise()
    model = train_model(args.alpha_sparse, args.alpha_entropy, args.alpha_mask)

    with open("simple_dataset_with_noise.pkl", 'rb') as f:
        graphs, labels, _ = pickle.load(f)

    class0_graphs = [g for g, l in zip(graphs, labels) if l == 0]
    sample_graphs = random.sample(class0_graphs, min(10, len(class0_graphs)))

    for i, graph in enumerate(sample_graphs[:3]):
        fig = visualize_explanation(model, graph, method=args.method)
        fig.savefig(os.path.join(img_dir, f"graph_{i+1}.png"))
        plt.close(fig)

    result = evaluate_explanation(model, sample_graphs[0], method=args.method)
    print("\n🔍 Final Evaluation:")
    for k, v in result.items():
        print(f"{k}: {v}")

    sys.stdout.close()

if __name__ == "__main__":
    main()


#################################


# def generate_trapezoid_dataset_with_noise(filename="trapezoid_dataset_with_noise.pkl", max_per_class=500):
#     graphs, labels, masks = [], [], []
#     label_counts = {0: 0, 1: 0}
#     for _ in tqdm(range(10000)):
#         if label_counts[0] < max_per_class:
#             G = generate_domain_expert_trapezoid_graph()
#             if add_random_noise(G, motif_checkers=[]):
#                 graphs.append(G)
#                 labels.append(0)
#                 masks.append(G.graph['motif_mask'])
#                 label_counts[0] += 1
#         elif label_counts[1] < max_per_class:
#             G = generate_random_non_motif_graph()
#             print(G.nodes(data=True))
#             if add_random_noise(G, motif_checkers=[contains_trapezoid_motif]):
#                 graphs.append(G)
#                 labels.append(1)
#                 masks.append(set())
#                 label_counts[1] += 1
#         else:
#             break
#     with open(filename, 'wb') as f:
#         pickle.dump((graphs, labels, masks), f)
#     return label_counts



# def visualize_explanation(model, graph):
#     """
#     Visualize the model's explanation by highlighting important edges.

#     Args:
#         model: Trained ExplainableHeteroGNN model
#         graph: NetworkX graph to explain
#     """
#     # Convert graph to HeteroData
#     data = convert_to_hetero_data(graph, graph.graph['motif_mask'])
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     data = data.to(device)

#     # Get edge importance scores and edge types
#     model.eval()
#     with torch.no_grad():
#         _, edge_scores, edge_types = model(data.x_dict, data.edge_index_dict, None)
#         print(f"Edge scores shape: {edge_scores}")
    
#     if edge_scores.numel() > 0:
#         edge_scores = edge_scores.cpu().numpy()
#         threshold = np.quantile(edge_scores, 0.5)

#         # Collect edges in order matching edge_scores
#         writes_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'writes']
#         cites_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'cites']
#         all_edges = writes_edges + cites_edges

#         # Important edge mask
#         important_edges = [edge for i, edge in enumerate(all_edges)
#                            if i < len(edge_scores) and edge_scores[i] > threshold]

#         # Draw
#         plt.figure(figsize=(10, 6))
#         pos = nx.spring_layout(graph, seed=42)
#         author_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'Author']
#         paper_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'Paper']
#         nx.draw_networkx_nodes(graph, pos, nodelist=author_nodes, node_color='lightblue', label='Authors')
#         nx.draw_networkx_nodes(graph, pos, nodelist=paper_nodes, node_color='lightgreen', label='Papers')
#         nx.draw_networkx_edges(graph, pos, alpha=0.2)

#         if important_edges:
#             nx.draw_networkx_edges(graph, pos, edgelist=important_edges,
#                                    edge_color='red', width=2, label='Important Edges')

#         # Ground truth motif edges
#         motif_edges = graph.graph.get('motif_mask', [])
#         if motif_edges:
#             nx.draw_networkx_edges(graph, pos, edgelist=motif_edges,
#                                    edge_color='black', width=2, style='dashed', label='Ground Truth')

#         nx.draw_networkx_labels(graph, pos)
#         plt.title("Explanation of Graph Classification")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No edge scores available for visualization")




# def train_explainable_model(a_writes = 0.0, a_cites = 0.0):
#     # with open("trapezoid_dataset_with_noise.pkl", 'rb') as f:
#     with open("simple_dataset_with_noise.pkl", 'rb') as f:
#         graphs, labels, masks = pickle.load(f)
#     hetero_data_list = []
#     for G, label, mask in zip(graphs, labels, masks):
#         data = convert_to_hetero_data(G, mask)
#         data.y = torch.tensor(label, dtype=torch.long)
#         hetero_data_list.append(data)

#     train_size = int(0.8 * len(hetero_data_list))
#     train_loader = DataLoader(hetero_data_list[:train_size], batch_size=8, shuffle=True)
#     test_loader = DataLoader(hetero_data_list[train_size:], batch_size=8)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = ExplainableHeteroGNN().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     for epoch in range(1, 10):
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             class_logits, edge_pred, edge_types = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
#             loss_cls = F.cross_entropy(class_logits, batch.y)

#             loss_mask_writes= 0
#             loss_mask_cites = 0
#             writes_len = batch['author', 'writes', 'paper'].edge_index.size(1)
#             cites_len = batch['paper', 'cites', 'paper'].edge_index.size(1)
#             edge_pred_writes = edge_pred[:writes_len]
#             edge_pred_cites = edge_pred[writes_len:writes_len + cites_len]

#             if hasattr(batch['author', 'writes', 'paper'], 'edge_mask'):
#                 loss_mask_writes += F.binary_cross_entropy(edge_pred_writes, batch['author', 'writes', 'paper'].edge_mask.to(device))
#             if hasattr(batch['paper', 'cites', 'paper'], 'edge_mask'):
#                 loss_mask_cites += F.binary_cross_entropy(edge_pred_cites, batch['paper', 'cites', 'paper'].edge_mask.to(device))
            
#             # a_writes = 0.0
#             # a_cites  = 0.0
#             loss = loss_cls + a_writes * loss_mask_writes + a_cites * loss_mask_cites
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         model.eval()
#         y_true, y_pred = [], []
#         subgraph_true, subgraph_pred = [], []
#         for batch in test_loader:
#             batch = batch.to(device)
#             with torch.no_grad():
#                 class_logits, edge_preds, _ = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
#                 preds = class_logits.argmax(dim=-1)
#                 y_true.extend(batch.y.cpu().numpy())
#                 y_pred.extend(preds.cpu().numpy())

#                 writes_len = batch['author', 'writes', 'paper'].edge_index.size(1)
#                 cites_len = batch['paper', 'cites', 'paper'].edge_index.size(1)
#                 edge_pred_writes = edge_preds[:writes_len]
#                 edge_pred_cites = edge_preds[writes_len:writes_len + cites_len]

#                 subgraph_true.extend(batch['author', 'writes', 'paper'].edge_mask.cpu().numpy())
#                 subgraph_pred.extend((edge_pred_writes > 0.5).cpu().numpy())
#                 subgraph_true.extend(batch['paper', 'cites', 'paper'].edge_mask.cpu().numpy())
#                 subgraph_pred.extend((edge_pred_cites > 0.5).cpu().numpy())
                
                


#         acc_cls = accuracy_score(np.array(y_true), np.array(y_pred))
#         acc_exp = accuracy_score(np.array(subgraph_true), np.array(subgraph_pred))
#         print(f"Epoch {epoch}: Loss={total_loss / len(train_loader):.4f}, Acc={acc_cls:.4f}, Acc_exp={acc_exp:.4f}")

#     return model



# class ExplainableHeteroGNN(nn.Module):
#     def __init__(self, hidden_dim=64, out_dim=2):
#         super().__init__()
#         self.conv1 = HeteroConv({
#             ('author', 'writes', 'paper'): SAGEConv(64, hidden_dim),
#             ('paper', 'cites', 'paper'): SAGEConv(64, hidden_dim),
#             ('author', 'self_loop', 'author'): SAGEConv(64, hidden_dim)
#         }, aggr='sum')
#         self.conv2 = HeteroConv({
#             ('author', 'writes', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('paper', 'cites', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('author', 'self_loop', 'author'): SAGEConv(hidden_dim, hidden_dim)
#         }, aggr='sum')
#         self.conv3 = HeteroConv({
#             ('author', 'writes', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('paper', 'cites', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('author', 'self_loop', 'author'): SAGEConv(hidden_dim, hidden_dim)
#         }, aggr='sum')
#         self.conv4 = HeteroConv({
#             ('author', 'writes', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('paper', 'cites', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('author', 'self_loop', 'author'): SAGEConv(hidden_dim, hidden_dim)
#         }, aggr='sum')
#         self.edge_mlp_writes = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
#         self.edge_mlp_cites = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

#     def forward(self, x_dict, edge_index_dict, batch_dict):
#         x_dict = {k: F.relu(v) for k, v in self.conv1(x_dict, edge_index_dict).items()}
#         x_dict = {k: F.relu(v) for k, v in self.conv2(x_dict, edge_index_dict).items()}
#         x_dict = {k: F.relu(v) for k, v in self.conv3(x_dict, edge_index_dict).items()}
#         x_dict = {k: F.relu(v) for k, v in self.conv4(x_dict, edge_index_dict).items()}
#         if batch_dict is not None:
#             x_author = global_mean_pool(x_dict['author'], batch_dict['author'])
#             x_paper = global_mean_pool(x_dict['paper'], batch_dict['paper'])
#         else: # this is the case for us, one graph at a time 
#             x_author = x_dict['author'].mean(dim=0, keepdim=True)
#             x_paper = x_dict['paper'].mean(dim=0, keepdim=True)
#         class_logits = self.fc(torch.cat([x_author, x_paper], dim=-1))

#         edge_scores = []
#         edge_tuples = []
#         if ('author', 'writes', 'paper') in edge_index_dict:
#             src, dst = edge_index_dict[('author', 'writes', 'paper')]
#             h_src = x_dict['author'][src]
#             h_dst = x_dict['paper'][dst]
#             # scores = self.edge_mlp_writes(torch.cat([h_src, h_dst], dim=-1)).squeeze()
#             scores = self.edge_mlp_writes(torch.cat([h_src, h_dst], dim=-1)).view(-1)
#             edge_scores.append(scores)
#             edge_tuples += [('author', 'writes', 'paper')] * scores.shape[0]
#         if ('paper', 'cites', 'paper') in edge_index_dict:
#             src, dst = edge_index_dict[('paper', 'cites', 'paper')]
#             h_src = x_dict['paper'][src]
#             h_dst = x_dict['paper'][dst]
#             # scores = self.edge_mlp_cites(torch.cat([h_src, h_dst], dim=-1)).squeeze()
#             scores = self.edge_mlp_cites(torch.cat([h_src, h_dst], dim=-1)).view(-1)
#             edge_scores.append(scores)
#             edge_tuples += [('paper', 'cites', 'paper')] * scores.shape[0]

#         final_scores = torch.cat(edge_scores, dim=0).sigmoid() if edge_scores else torch.tensor([])
#         return class_logits, final_scores, edge_tuples


# def generate_domain_expert_trapezoid_graph():
#     G = nx.MultiDiGraph()
#     authors, papers = add_nodes(G, 1, 4)
#     a = authors[0]
#     p1, p2, p3, p4 = papers
#     motif_edges = set([
#         (a, p1), (a, p4),
#         (p1, p2), (p2, p3), (p3, p4), (p1, p4)
#     ])
#     for u, v in motif_edges:
#         G.add_edge(u, v, type='writes' if u == a else 'cites')
#     G.graph['motif_mask'] = motif_edges
#     return G
# def contains_trapezoid_motif(G):
#     for a in [n for n, d in G.nodes(data=True) if d['type'] == 'Author']:
#         written = [v for u, v, d in G.out_edges(a, data=True) if d['type'] == 'writes']
#         if len(written) < 2:
#             continue
#         for p1 in written:
#             for p4 in written:
#                 if p1 == p4:
#                     continue
#                 p1_cites = [v for u, v, d in G.out_edges(p1, data=True) if d['type'] == 'cites']
#                 p4_cited_by = [u for u, v, d in G.in_edges(p4, data=True) if d['type'] == 'cites']
#                 if any(p != p4 and G.has_edge(p, p4) for p in G.nodes if G.nodes[p]['type'] == 'Paper') and p4 in p1_cites:
#                     return True
#     return False
# class ExplainableHeteroGNN(nn.Module):
#     def __init__(self, hidden_dim=64, out_dim=2):
#         super().__init__()
#         self.hidden_dim = hidden_dim

#         self.conv1 = HeteroConv({
#             ('author', 'writes', 'paper'): SAGEConv(64, hidden_dim),
#             ('paper', 'cites', 'paper'): SAGEConv(64, hidden_dim),
#             ('author', 'self_loop', 'author'): SAGEConv(64, hidden_dim)
#         }, aggr='sum')
#         self.conv2 = HeteroConv({
#             ('author', 'writes', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('paper', 'cites', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('author', 'self_loop', 'author'): SAGEConv(hidden_dim, hidden_dim)
#         }, aggr='sum')
#         self.conv3 = HeteroConv({
#             ('author', 'writes', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('paper', 'cites', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('author', 'self_loop', 'author'): SAGEConv(hidden_dim, hidden_dim)
#         }, aggr='sum')
#         self.conv4 = HeteroConv({
#             ('author', 'writes', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('paper', 'cites', 'paper'): SAGEConv(hidden_dim, hidden_dim),
#             ('author', 'self_loop', 'author'): SAGEConv(hidden_dim, hidden_dim)
#         }, aggr='sum')

#         # Attention vectors per edge type
#         self.att_writes = nn.Parameter(torch.randn(hidden_dim * 2))
#         self.att_cites  = nn.Parameter(torch.randn(hidden_dim * 2))

#         self.fc = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))

#     def compute_attention_scores(self, h_src, h_dst, att_vector):
#         h_cat = torch.cat([h_src, h_dst], dim=-1)  # [E, 2H]
#         scores = (h_cat * att_vector).sum(dim=-1)  # [E]
#         return scores

#     def forward(self, x_dict, edge_index_dict, batch_dict, return_attention=False):
#         x_dict = {k: F.relu(v) for k, v in self.conv1(x_dict, edge_index_dict).items()}
#         x_dict = {k: F.relu(v) for k, v in self.conv2(x_dict, edge_index_dict).items()}
#         x_dict = {k: F.relu(v) for k, v in self.conv3(x_dict, edge_index_dict).items()}
#         x_dict = {k: F.relu(v) for k, v in self.conv4(x_dict, edge_index_dict).items()}

#         if batch_dict is not None:
#             x_author = global_mean_pool(x_dict['author'], batch_dict['author'])
#             x_paper = global_mean_pool(x_dict['paper'], batch_dict['paper'])
#         else:
#             x_author = x_dict['author'].mean(dim=0, keepdim=True)
#             x_paper = x_dict['paper'].mean(dim=0, keepdim=True)
        
#         class_logits = self.fc(torch.cat([x_author, x_paper], dim=-1))

#         edge_scores = []
#         edge_tuples = []

#         if ('author', 'writes', 'paper') in edge_index_dict:
#             src, dst = edge_index_dict[('author', 'writes', 'paper')]
#             h_src = x_dict['author'][src]
#             h_dst = x_dict['paper'][dst]
#             scores = self.compute_attention_scores(h_src, h_dst, self.att_writes)
#             edge_scores.append(scores)
#             edge_tuples += [('author', 'writes', 'paper')] * scores.shape[0]

#         if ('paper', 'cites', 'paper') in edge_index_dict:
#             src, dst = edge_index_dict[('paper', 'cites', 'paper')]
#             h_src = x_dict['paper'][src]
#             h_dst = x_dict['paper'][dst]
#             scores = self.compute_attention_scores(h_src, h_dst, self.att_cites)
#             edge_scores.append(scores)
#             edge_tuples += [('paper', 'cites', 'paper')] * scores.shape[0]

#         final_scores = torch.cat(edge_scores, dim=0).sigmoid() if edge_scores else torch.tensor([])

#         if return_attention:
#             return class_logits, final_scores, edge_tuples
#         else:
#             return class_logits


# def evaluate_subgraph_explanation(model, graph, top_k_ratio=0.2):
#     motif_mask = graph.graph.get("motif_mask", set())
#     if not motif_mask:
#         print("⚠️ No ground truth motif_mask found in graph.")
#         return {}

#     # Convert motif edges to (u, v, type)
#     true_edges = {(u, v, d['type']) for u, v, d in graph.edges(data=True) if (u, v) in motif_mask}

#     data = convert_to_hetero_data(graph)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data = data.to(device)

#     model.eval()
#     with torch.no_grad():
#         _, edge_scores, edge_types = model(data.x_dict, data.edge_index_dict, None, return_attention=True)

#     edge_scores = edge_scores.cpu().numpy()

#     # Collect all edge (u, v, type) triples in same order as scores
#     writes_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'writes']
#     cites_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'cites']
#     all_edges = [(u, v, 'writes') for u, v in writes_edges] + [(u, v, 'cites') for u, v in cites_edges]

#     top_k = max(1, int(top_k_ratio * len(edge_scores)))
#     top_indices = np.argsort(edge_scores)[-top_k:]
#     predicted_edges = {all_edges[i] for i in top_indices}

#     intersection = predicted_edges & true_edges
#     precision = len(intersection) / len(predicted_edges) if predicted_edges else 0
#     recall = len(intersection) / len(true_edges) if true_edges else 0
#     f1 = 2 * precision * recall / (precision + recall + 1e-8) if (precision + recall) > 0 else 0

#     return {
#         "intersection": list(intersection),
#         "precision": precision,
#         "recall": recall,
#         "f1": f1
#     }

# ================== Explanation Visualization ==================
# from sklearn.metrics import accuracy_score, f1_score

# def train_explainable_model(alpha_sparse=0.00001, alpha_entropy=0.0001, alpha_mask = 0.00001):
#     with open("simple_dataset_with_noise.pkl", 'rb') as f:
#         graphs, labels, masks = pickle.load(f)

#     hetero_data_list = []
#     for G, label, mask in zip(graphs, labels, masks):
#         data = convert_to_hetero_data(G)  # Do NOT pass `mask` anymore
#         data.y = torch.tensor(label, dtype=torch.long)
#         hetero_data_list.append(data)

#     # Shuffle + split
#     from sklearn.model_selection import train_test_split
#     train_data, test_data = train_test_split(hetero_data_list, test_size=0.2, stratify=[d.y.item() for d in hetero_data_list])
#     train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
#     test_loader = DataLoader(test_data, batch_size=8)

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = ExplainableHeteroGNN().to(device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#     for epoch in range(1, 6):
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             batch = batch.to(device)
#             optimizer.zero_grad()
#             class_logits, edge_preds, _ = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, return_attention=True)

#             # --- Classification Loss ---
#             loss_cls = F.cross_entropy(class_logits, batch.y)

#             # --- GNNExplainer-style Loss ---
#             edge_preds = edge_preds.clamp(min=1e-6, max=1-1e-6)
#             loss_sparse = edge_preds.sum() / edge_preds.size(0)
#             entropy = - (edge_preds * edge_preds.log() + (1 - edge_preds) * (1 - edge_preds).log())
#             loss_entropy = entropy.mean()
#             loss_mask_writes = 0 
#             loss_mask_cites = 0

#             writes_len = batch['author', 'writes', 'paper'].edge_index.size(1)
#             cites_len = batch['paper', 'cites', 'paper'].edge_index.size(1)
#             edge_pred_writes = edge_preds[:writes_len]
#             edge_pred_cites = edge_preds[writes_len:writes_len + cites_len]
#             if hasattr(batch['author', 'writes', 'paper'], 'edge_mask'):
#                 loss_mask_writes += F.binary_cross_entropy(edge_pred_writes, batch['author', 'writes', 'paper'].edge_mask.to(device))
#             if hasattr(batch['paper', 'cites', 'paper'], 'edge_mask'):
#                 loss_mask_cites += F.binary_cross_entropy(edge_pred_cites, batch['paper', 'cites', 'paper'].edge_mask.to(device))

#             loss = loss_cls + alpha_sparse * loss_sparse + alpha_entropy * loss_entropy + alpha_mask * (loss_mask_writes + loss_mask_cites)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()

#         # -------- Evaluation --------
#         model.eval()
#         y_true, y_pred = [], []
#         for batch in test_loader:
#             batch = batch.to(device)
#             with torch.no_grad():
#                 class_logits = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)
#                 preds = class_logits.argmax(dim=-1)
#                 y_true.extend(batch.y.cpu().numpy())
#                 y_pred.extend(preds.cpu().numpy())

#         acc_cls = accuracy_score(y_true, y_pred)
#         print(f"Epoch {epoch}: Loss={total_loss / len(train_loader):.4f}, Acc={acc_cls:.4f}")

#     return model


# def visualize_explanation(model, graph, show_ground_truth=True):
#     """
#     Visualize the model's explanation by highlighting important edges via attention.

#     Args:
#         model: Trained ExplainableHeteroGNN model
#         graph: NetworkX graph to explain
#         show_ground_truth (bool): If True, show ground truth motif edges (if present)
#     """
#     # Convert to HeteroData (no mask needed anymore)
#     data = convert_to_hetero_data(graph)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     data = data.to(device)

#     # Get attention scores from the model
#     model.eval()
#     with torch.no_grad():
#         _, edge_scores, edge_types = model(data.x_dict, data.edge_index_dict, None, return_attention=True)

#     if edge_scores.numel() > 0:
#         edge_scores = edge_scores.cpu().numpy()
#         threshold = np.quantile(edge_scores, 0.0)  # Show top 20% edges as important

#         # Match score order to edge order
#         writes_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'writes']
#         cites_edges = [(u, v) for u, v, d in graph.edges(data=True) if d['type'] == 'cites']
#         all_edges = writes_edges + cites_edges

#         # Important edge mask (based on attention threshold)
#         important_edges = [edge for i, edge in enumerate(all_edges)
#                            if i < len(edge_scores) and edge_scores[i] > threshold]

#         # Draw
#         plt.figure(figsize=(10, 6))
#         pos = nx.spring_layout(graph, seed=42)
#         author_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'Author']
#         paper_nodes = [n for n, d in graph.nodes(data=True) if d['type'] == 'Paper']
#         nx.draw_networkx_nodes(graph, pos, nodelist=author_nodes, node_color='lightblue', label='Authors')
#         nx.draw_networkx_nodes(graph, pos, nodelist=paper_nodes, node_color='lightgreen', label='Papers')
#         nx.draw_networkx_edges(graph, pos, alpha=0.1)  # Light edges in background

#         if important_edges:
#             nx.draw_networkx_edges(graph, pos, edgelist=important_edges,
#                                    edge_color='red', width=2, label='Important Edges')

#         # Optionally show ground truth motif edges
#         motif_edges = graph.graph.get('motif_mask', [])
#         if show_ground_truth and motif_edges:
#             nx.draw_networkx_edges(graph, pos, edgelist=motif_edges,
#                                    edge_color='black', width=2, style='dashed', label='Ground Truth')

#         nx.draw_networkx_labels(graph, pos)
#         plt.title("Explanation of Graph Classification via Attention")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
#     else:
#         print("No edge scores available for visualization")


# # ================== Main Execution ==================
# def visualize_and_evaluate_5_class0_graphs():
#     # Load trained model
#     model = train_explainable_model()

#     # Load dataset
#     with open("simple_dataset_with_noise.pkl", 'rb') as f:
#         graphs, labels, _ = pickle.load(f)

#     # Collect all class-0 graphs
#     class0_graphs = [g for g, l in zip(graphs, labels) if l == 0]
#     print(f"\n📦 Total class-0 graphs: {len(class0_graphs)}")

#     # ---- 🎨 Visualization (only 5) ----
#     sample_graphs = random.sample(class0_graphs, 5)
#     for i, graph in enumerate(sample_graphs):
#         print(f"\n🔍 Visualizing Graph {i+1}/5")
#         visualize_explanation(model, graph)

#     # ---- 🧪 Evaluation (all class-0) ----
#     all_results = []
#     for i, graph in enumerate(class0_graphs):
#         result = evaluate_subgraph_explanation(model, graph)
#         all_results.append(result)

#     # ---- 📊 Aggregate Results ----
#     precisions = [r["precision"] for r in all_results]
#     recalls = [r["recall"] for r in all_results]
#     f1s = [r["f1"] for r in all_results]

#     print("\n📊 Average Subgraph Explanation Scores (across all class-0 graphs):")
#     print(f"Precision: {np.mean(precisions):.4f}")
#     print(f"Recall:    {np.mean(recalls):.4f}")
#     print(f"F1 Score:  {np.mean(f1s):.4f}")