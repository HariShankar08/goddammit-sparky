# data.py
"""
Generate a mixed dataset with *mutually‑exclusive* motifs:
    0 – simple  (two authors → one paper)
    1 – trapezoid
    2 – square
    3 – no motif  (negative class)

$ python -m data  --out  mixed_ds.pkl  --per_class 500  --seed 42
"""
import random, pickle, argparse
from tqdm import tqdm
import networkx as nx

### ---- helpers --------------------------------------------------------- ###
def _add_nodes(G, n_auth, n_pap, off=0):
    auth  = list(range(off, off + n_auth))
    paper = list(range(off + n_auth, off + n_auth + n_pap))
    for a in auth:   G.add_node(a, type="Author")
    for p in paper:  G.add_node(p, type="Paper")
    return auth, paper

def _simple():                        # class‑0
    G = nx.MultiDiGraph()
    a, p = _add_nodes(G, 2, 1)
    G.add_edge(a[0], p[0], type="writes")
    G.add_edge(a[1], p[0], type="writes")
    G.graph["motif_mask"] = {(a[0], p[0]), (a[1], p[0])}
    return G

def _trapezoid():                     # class‑1
    G = nx.MultiDiGraph()
    a, p = _add_nodes(G, 1, 4)
    a0, (p1, p2, p3, p4) = a[0], p
    edges = {(a0,p1),(a0,p4),(p1,p2),(p3,p4),(p1,p4)}
    for u,v in edges:
        G.add_edge(u,v,type="writes" if u==a0 else "cites")
    G.graph["motif_mask"] = edges
    return G

def _square():                        # class‑2
    G = nx.MultiDiGraph()
    a, p = _add_nodes(G, 1, 3)
    a0, (p1,p2,p3)=a[0],p
    edges={(a0,p1),(a0,p2),(a0,p3),(p1,p3),(p2,p3)}
    for u,v in edges:
        G.add_edge(u,v,type="writes" if u==a0 else "cites")
    G.graph["motif_mask"] = edges
    return G

def _noise(G, extra_auth=1, extra_pap=2, seed=None):
    rnd = random.Random(seed)
    off = max(G.nodes)+1
    A,P=_add_nodes(G,extra_auth,extra_pap,off)
    papers=[n for n,d in G.nodes(data=True) if d["type"]=="Paper"]
    # random writes
    for a in A:
        for p in rnd.sample(papers, rnd.randint(1,3)):
            G.add_edge(a,p,type="writes")
    # random cites
    for i in papers:
        for j in papers:
            if i!=j and rnd.random()<0.3:
                G.add_edge(i,j,type="cites")

### ---- public API ------------------------------------------------------- ###
def generate_dataset(per_class=500, seed=42):
    rnd=random.Random(seed)
    gens=[_simple,_trapezoid,_square]
    graphs,labels,masks=[],[],[]
    counts=[0,0,0,0]
    for _ in tqdm(range(per_class*4)):
        cls= rnd.choice([c for c,cnt in enumerate(counts) if cnt<per_class])
        if cls<3:
            G=gens[cls]()
            _noise(G,seed=rnd.randint(0,1<<30))
        else:                          # negative class (no motif)
            while True:
                G=nx.MultiDiGraph()
                _add_nodes(G,2,6)
                _noise(G,seed=rnd.randint(0,1<<30))
                if not any(f(G) for f in (contains_simple,contains_trap,contains_square)):
                    G.graph["motif_mask"]=set()
                    break
        graphs.append(G); labels.append(cls); masks.append(G.graph["motif_mask"])
        counts[cls]+=1
    return graphs,labels,masks

# --- very cheap motif detectors for negative‑class filtering -------------
# ----------------------------------------------------------------------
#  Fast‑but‑sound motif detectors used only for *filtering negatives*.
#  They never rely on G.graph["motif_mask"] and run in O(|E|)‑to‑O(|V|³),
#  which is fine for our small synthetic graphs.
# ----------------------------------------------------------------------

def contains_simple(G):
    """Return True if ∃ two distinct authors a1,a2 and a paper p
       such that a1→p and a2→p are both ‘writes’ edges."""
    authors = [n for n,d in G.nodes(data=True) if d['type']=='Author']
    papers  = [n for n,d in G.nodes(data=True) if d['type']=='Paper']
    writes  = {(u,v) for u,v,attr in G.edges(data=True) if attr['type']=='writes'}
    for p in papers:
        writers = [a for a in authors if (a,p) in writes]
        if len(writers) >= 2:
            return True
    return False


def contains_trap(G):
    """
    Trapezoid (our class‑1 motif):
        author a writes p1 and p4
        p1 → p2   (cites)
        p3 → p4   (cites)
        p1 → p4   (cites)
    Any instantiation of those five directed edges is enough.
    """
    a_nodes = [n for n,d in G.nodes(data=True) if d['type']=='Author']
    p_nodes = [n for n,d in G.nodes(data=True) if d['type']=='Paper']

    writes = {(u,v) for u,v,attr in G.edges(data=True) if attr['type']=='writes'}
    cites  = {(u,v) for u,v,attr in G.edges(data=True) if attr['type']=='cites'}

    for a in a_nodes:
        # pick two distinct papers written by the same author
        pw = [p for p in p_nodes if (a,p) in writes]
        for i in range(len(pw)):
            for j in range(i+1, len(pw)):
                p1, p4 = pw[i], pw[j]

                # look for p1→p4 plus one extra “middle” citation p1→p2 and p3→p4
                if (p1, p4) not in cites:
                    continue
                for p2 in p_nodes:
                    if (p1, p2) not in cites or p2 in {p1, p4}:
                        continue
                    for p3 in p_nodes:
                        if (p3, p4) in cites and p3 not in {p1, p4, p2}:
                            return True
    return False


def contains_square(G):
    """
    Square motif (our class‑2):
        author a writes p1,p2,p3
        p1 → p3   and   p2 → p3  (both cites)
    """
    a_nodes = [n for n,d in G.nodes(data=True) if d['type']=='Author']
    p_nodes = [n for n,d in G.nodes(data=True) if d['type']=='Paper']
    writes  = {(u,v) for u,v,attr in G.edges(data=True) if attr['type']=='writes'}
    cites   = {(u,v) for u,v,attr in G.edges(data=True) if attr['type']=='cites'}

    for a in a_nodes:
        written = [p for p in p_nodes if (a,p) in writes]
        if len(written) < 3:
            continue
        # choose p1, p2 (writers) and p3 (common cite target)
        for i in range(len(written)):
            for j in range(i+1, len(written)):
                for k in range(len(written)):
                    if k in (i, j):
                        continue
                    p1, p2, p3 = written[i], written[j], written[k]
                    if (p1, p3) in cites and (p2, p3) in cites:
                        return True
    return False


### ---- CLI -------------------------------------------------------------- ###
if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="mixed_dataset.pkl")
    ap.add_argument("--per_class",type=int,default=500)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()
    g,l,m=generate_dataset(args.per_class,args.seed)
    pickle.dump((g,l,m),open(args.out,"wb"))
    print("Saved ->",args.out)
