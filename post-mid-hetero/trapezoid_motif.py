# same as simple_motif.py but testing on trapezoid and square motif 
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

def generate_trapezoid_motif_graph():
# def generate_domain_expert_trapezoid_graph():
    G = nx.MultiDiGraph()
    authors, papers = add_nodes(G, 1, 4)

    a = authors[0]
    p1, p2, p3, p4 = papers

    motif_edges = {
        (a, p1), (a, p4),
        (p1, p2), (p3, p4), (p1, p4)
    }

    for u, v in motif_edges:
        edge_type = 'writes' if G.nodes[u]['type'] == 'Author' else 'cites'
        G.add_edge(u, v, type=edge_type)

    G.graph['motif_mask'] = motif_edges
    return G
def contains_trapezoid_motif(G):
    authors = [n for n, d in G.nodes(data=True) if d['type'] == 'Author']
    for a in authors:
        written = [v for u, v, d in G.out_edges(a, data=True) if d['type'] == 'writes']
        if len(written) < 2:
            continue
        for p1 in written:
            for p4 in written:
                if p1 == p4:
                    continue
                if G.has_edge(p1, p4) and any(
                    G.has_edge(p1, p2) and G.has_edge(p3, p4)
                    for p2 in G.successors(p1)
                    for p3 in G.predecessors(p4)
                ):
                    return True
    return False


def add_random_noise(G, num_extra_authors=1, num_extra_papers=2, max_writes=2, max_cites=3, motif_checkers=[]):
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
        if not contains_trapezoid_motif(G):
            return G
def generate_simple_dataset_with_noise(filename="trapezoidmtf_dataset_with_noise.pkl", max_per_class=500):
    graphs, labels, masks = [], [], []
    label_counts = {0: 0, 1: 0}
    for _ in tqdm(range(10000)):
        if label_counts[0] < max_per_class:
            G = generate_trapezoid_motif_graph()
            if add_random_noise(G, motif_checkers=[]):
                graphs.append(G)
                labels.append(0)
                masks.append(G.graph['motif_mask'])
                label_counts[0] += 1
        elif label_counts[1] < max_per_class:
            G = generate_random_non_motif_graph()
            # print(G.nodes(data=True))
            if add_random_noise(G, motif_checkers=[contains_trapezoid_motif]):
                graphs.append(G)
                labels.append(1)
                masks.append(set())
                label_counts[1] += 1
        else:
            break
    with open(filename, 'wb') as f:
        pickle.dump((graphs, labels, masks), f)
    return label_counts


from simple_motif import convert_to_hetero_data, ExplainableHeteroGNN, visualize_explanation

# train model 

def train_model(alpha_sparse=0.005, alpha_entropy=0.001, alpha_mask=0.1):
    with open("trapezoidmtf_dataset_with_noise.pkl", 'rb') as f:
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
            torch.save(model.state_dict(), "best_model_trapezoid.pt")
        else:
            patience_counter += 1
            if patience_counter >= 5:  # 5 epochs without improvement
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model
    model.load_state_dict(torch.load("best_model_trapezoid.pt"))
    return model

from simple_motif import evaluate_explanation, visualize_and_evaluate_explanations

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

    with open("trapezoidmtf_dataset_with_noise.pkl", 'rb') as f:
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
