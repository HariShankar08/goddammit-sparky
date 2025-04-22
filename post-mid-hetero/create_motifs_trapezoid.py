import networkx as nx
import random
import pickle
from tqdm import tqdm

def add_nodes(G, num_authors, num_papers, offset=0):
    author_ids = list(range(offset, offset + num_authors))
    paper_ids = list(range(offset + num_authors, offset + num_authors + num_papers))
    for aid in author_ids:
        G.add_node(aid, type='Author')
    for pid in paper_ids:
        G.add_node(pid, type='Paper')
    return author_ids, paper_ids

# --- Motif: Domain Expert Trapezoid ---
def generate_domain_expert_trapezoid_graph():
    G = nx.MultiDiGraph()
    authors, papers = add_nodes(G, 1, 4)

    a = authors[0]
    p1, p2, p3, p4 = papers

    # a -> p1, a -> p4
    G.add_edge(a, p1, type='writes')
    G.add_edge(a, p4, type='writes')

    # p1 -> p2, p3 -> p4, p1 -> p4
    G.add_edge(p1, p2, type='cites')
    G.add_edge(p3, p4, type='cites')
    G.add_edge(p1, p4, type='cites')

    return G

# --- Motif-Free Graph ---
def generate_random_non_motif_graph():
    while True:
        G = nx.MultiDiGraph()
        authors, papers = add_nodes(G, 1, 6)

        # Random writes
        for a in authors:
            for p in random.sample(papers, random.randint(1, 3)):
                G.add_edge(a, p, type='writes')

        # Random citations (without forming motifs)
        for i in range(len(papers)):
            for j in range(len(papers)):
                if i != j and random.random() < 0.5:
                    G.add_edge(papers[i], papers[j], type='cites')

        if not contains_trapezoid_motif(G):
            return G

# --- Motif Checkers (for validation) ---

def contains_trapezoid_motif(G):
    for a in [n for n, d in G.nodes(data=True) if d['type'] == 'Author']:
        written = [v for u, v, d in G.out_edges(a, data=True) if d['type'] == 'writes']
        if len(written) < 2:
            continue
        for p1 in written:
            for p4 in written:
                if p1 == p4:
                    continue
                p1_cites = [v for u, v, d in G.out_edges(p1, data=True) if d['type'] == 'cites']
                p4_cited_by = [u for u, v, d in G.in_edges(p4, data=True) if d['type'] == 'cites']
                if any(p != p4 and G.has_edge(p, p4) for p in G.nodes if G.nodes[p]['type'] == 'Paper') and p4 in p1_cites:
                    return True
    return False

# ----- add noise -------- 
def add_random_noise(G, num_extra_authors=3, num_extra_papers=5, max_writes=3, max_cites=3, motif_checkers=[]):
    offset = max(G.nodes) + 1 if G.nodes else 0
    authors, papers = add_nodes(G, num_extra_authors, num_extra_papers, offset)

    # Add random writes
    for a in authors:
        for p in random.sample(papers, random.randint(1, max_writes)):
            G.add_edge(a, p, type='writes')

    # Add random cites
    all_papers = [n for n, d in G.nodes(data=True) if d['type'] == 'Paper']
    for i in range(len(all_papers)):
        for j in range(len(all_papers)):
            if i != j and random.random() < 0.4:
                G.add_edge(all_papers[i], all_papers[j], type='cites')

    # Validate that added noise didn’t break motif constraints
    for checker in motif_checkers:
        if checker(G):
            print("checker", checker.__name__, "found a motif after adding noise")
            return False

    return True

def generate_domain_expert_trapezoid_graph_with_noise():
    while True:
        G = generate_domain_expert_trapezoid_graph()
        success = add_random_noise(G, motif_checkers=[])
        if success:
            return G
def generate_random_non_motif_graph_with_noise():
    while True:
        G = generate_random_non_motif_graph()
        success = add_random_noise(G, motif_checkers=[contains_trapezoid_motif])
        if success:
            return G

# --- Dataset Generation ---

def generate_trapezoid_dataset_with_noise(filename="trapezoid_dataset_with_noise.pkl", max_per_class=500):
    graphs, labels = [], []
    label_counts = {0: 0, 
                    # 1: 0,
                      1: 0}

    for _ in tqdm(range(10000)):
        if label_counts[0] < max_per_class:
            G = generate_domain_expert_trapezoid_graph_with_noise()
            graphs.append(G)
            labels.append(0)
            label_counts[0] += 1
            continue

        if label_counts[1] < max_per_class:
            G = generate_random_non_motif_graph_with_noise()
            graphs.append(G)
            labels.append(1)
            label_counts[1] += 1
            continue

        if all(v == max_per_class for v in label_counts.values()):
            break

    with open(filename, 'wb') as f:
        pickle.dump((graphs, labels), f)

    return label_counts

# Run and store
if __name__ == "__main__":
    counts = generate_trapezoid_dataset_with_noise()
    print("Label counts:", counts)
