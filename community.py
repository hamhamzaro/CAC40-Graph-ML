"""
community.py
------------
Community detection on CAC40 correlation graphs.

- Louvain algorithm: fast modularity-maximizing clustering
- Leiden algorithm: improved version with better modularity guarantees
- node2vec: graph embeddings for link prediction and structural similarity
- Temporal community tracking: community evolution across rolling windows

Usage:
    python src/community.py
"""

import networkx as nx
import numpy as np
import pandas as pd
import pickle
import os
import community as community_louvain   # python-louvain
import leidenalg
import igraph as ig
from node2vec import Node2Vec
from sklearn.metrics import adjusted_rand_score
from typing import Optional


# ─── Louvain ──────────────────────────────────────────────────────────────────

def detect_communities_louvain(
    G: nx.Graph,
    resolution: float = 1.0,
    random_state: int = 42
) -> dict[str, int]:
    """
    Detect communities using the Louvain algorithm.

    Louvain iteratively merges nodes to maximize modularity Q.
    Resolution > 1 → smaller communities. Resolution < 1 → larger ones.

    Args:
        G:            NetworkX graph (weighted).
        resolution:   Modularity resolution parameter.
        random_state: Seed for reproducibility.

    Returns:
        Dict mapping node (ticker) → community ID.
    """
    partition = community_louvain.best_partition(
        G,
        weight="weight",
        resolution=resolution,
        random_state=random_state
    )
    n_communities = len(set(partition.values()))
    modularity = community_louvain.modularity(partition, G, weight="weight")
    print(f"Louvain: {n_communities} communities | modularity={modularity:.4f}")
    return partition


# ─── Leiden ───────────────────────────────────────────────────────────────────

def detect_communities_leiden(
    G: nx.Graph,
    resolution: float = 1.0,
    random_state: int = 42
) -> dict[str, int]:
    """
    Detect communities using the Leiden algorithm.

    Leiden improves on Louvain by guaranteeing well-connected communities
    and avoiding the resolution limit problem.

    Args:
        G:          NetworkX graph (weighted).
        resolution: Resolution parameter for modularity.

    Returns:
        Dict mapping node (ticker) → community ID.
    """
    # Convert NetworkX → igraph
    nodes = list(G.nodes())
    node_idx = {n: i for i, n in enumerate(nodes)}
    edges = [(node_idx[u], node_idx[v]) for u, v in G.edges()]
    weights = [G[u][v].get("weight", 1.0) for u, v in G.edges()]

    ig_graph = ig.Graph(n=len(nodes), edges=edges)
    ig_graph.es["weight"] = weights

    np.random.seed(random_state)
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state
    )

    community_map = {nodes[i]: partition.membership[i] for i in range(len(nodes))}
    n_communities = len(set(partition.membership))
    modularity = partition.modularity
    print(f"Leiden:  {n_communities} communities | modularity={modularity:.4f}")
    return community_map


# ─── Algorithm Comparison ─────────────────────────────────────────────────────

def compare_algorithms(
    G: nx.Graph,
    louvain_partition: dict,
    leiden_partition: dict
) -> pd.DataFrame:
    """
    Compare Louvain vs Leiden community assignments.

    Args:
        G:                 The correlation graph.
        louvain_partition: Community assignments from Louvain.
        leiden_partition:  Community assignments from Leiden.

    Returns:
        DataFrame with per-node community assignments from both algorithms.
    """
    nodes = list(G.nodes())
    df = pd.DataFrame({
        "ticker": nodes,
        "louvain": [louvain_partition.get(n, -1) for n in nodes],
        "leiden":  [leiden_partition.get(n, -1) for n in nodes],
    })

    # Adjusted Rand Index — agreement between two clusterings
    ari = adjusted_rand_score(df["louvain"], df["leiden"])
    print(f"Algorithm agreement (ARI): {ari:.4f} (1.0 = identical)")
    df["agreement"] = df["louvain"] == df["leiden"]
    return df


# ─── node2vec Embeddings ──────────────────────────────────────────────────────

def train_node2vec(
    G: nx.Graph,
    dimensions: int = 64,
    walk_length: int = 30,
    num_walks: int = 200,
    p: float = 1.0,
    q: float = 0.5,
    window: int = 10,
    random_state: int = 42
) -> dict[str, np.ndarray]:
    """
    Train node2vec embeddings on the correlation graph.

    node2vec learns low-dimensional representations by running biased
    random walks and training a Word2Vec model on the resulting sequences.

    p (return parameter): controls likelihood of revisiting a node.
    q (in-out parameter): q < 1 → DFS-like (community structure);
                          q > 1 → BFS-like (structural equivalence).

    Args:
        G:           NetworkX graph.
        dimensions:  Embedding dimension.
        walk_length: Length of each random walk.
        num_walks:   Number of walks per node.
        p, q:        Biased walk parameters.
        window:      Word2Vec context window.

    Returns:
        Dict mapping ticker → embedding vector (np.ndarray of shape [dimensions]).
    """
    print(f"Training node2vec (dim={dimensions}, walks={num_walks}×{walk_length})...")
    node2vec = Node2Vec(
        G,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        weight_key="weight",
        workers=4,
        seed=random_state,
        quiet=True
    )
    model = node2vec.fit(window=window, min_count=1, batch_words=4)

    embeddings = {node: model.wv[str(node)] for node in G.nodes()}
    print(f"Embeddings trained: {len(embeddings)} nodes × {dimensions}d")
    return embeddings, model


# ─── Link Prediction ──────────────────────────────────────────────────────────

def predict_links(
    G: nx.Graph,
    embeddings: dict[str, np.ndarray],
    top_k: int = 20
) -> pd.DataFrame:
    """
    Predict missing edges using node2vec cosine similarity.

    Pairs of non-adjacent nodes with high embedding similarity
    are likely to become correlated in future windows.

    Args:
        G:          Current graph (existing edges excluded).
        embeddings: node2vec embeddings dict.
        top_k:      Number of top predicted links to return.

    Returns:
        DataFrame of top-k predicted links with similarity scores.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    nodes = list(G.nodes())
    emb_matrix = np.stack([embeddings[n] for n in nodes])
    sim_matrix = cosine_similarity(emb_matrix)

    candidates = []
    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if j <= i:
                continue
            if G.has_edge(n1, n2):
                continue  # existing edge, skip
            candidates.append({
                "source": n1,
                "target": n2,
                "similarity": sim_matrix[i, j]
            })

    df = pd.DataFrame(candidates).nlargest(top_k, "similarity").reset_index(drop=True)
    print(f"Top {top_k} predicted links (non-existing edges with highest embedding similarity):")
    print(df.head(10).to_string(index=False))
    return df


# ─── Temporal Community Tracking ──────────────────────────────────────────────

def track_communities_over_time(
    rolling_graphs: list[dict],
    algorithm: str = "louvain"
) -> pd.DataFrame:
    """
    Track community membership evolution across rolling graph windows.

    Args:
        rolling_graphs: List of rolling graph dicts (from graph_builder.py).
        algorithm:      'louvain' or 'leiden'.

    Returns:
        DataFrame (index=window end dates, columns=tickers, values=community IDs).
    """
    records = []
    fn = detect_communities_louvain if algorithm == "louvain" else detect_communities_leiden

    for g_dict in rolling_graphs:
        G = g_dict["graph"]
        if G.number_of_edges() < 5:
            continue  # skip sparse windows
        partition = fn(G, resolution=1.0)
        row = {"date": g_dict["end"], **partition}
        records.append(row)

    df = pd.DataFrame(records).set_index("date")
    print(f"Community tracking: {len(df)} windows | {df.shape[1]} stocks")
    return df


# ─── Community Summary ────────────────────────────────────────────────────────

def summarize_communities(
    G: nx.Graph,
    partition: dict,
    sector_map: Optional[dict] = None,
    name_map: Optional[dict] = None
) -> pd.DataFrame:
    """
    Build a summary table of detected communities.

    Args:
        G:          Correlation graph.
        partition:  Community assignment dict.
        sector_map: Optional ticker → GICS sector mapping.
        name_map:   Optional ticker → company name mapping.

    Returns:
        DataFrame with one row per node and community metadata.
    """
    rows = []
    for ticker, community_id in partition.items():
        degree = G.degree(ticker) if ticker in G else 0
        rows.append({
            "ticker": ticker,
            "community": community_id,
            "degree": degree,
            "name": name_map.get(ticker, ticker) if name_map else ticker,
            "sector": sector_map.get(ticker, "Unknown") if sector_map else "Unknown",
        })

    df = pd.DataFrame(rows).sort_values(["community", "degree"], ascending=[True, False])

    # Community-level stats
    print("\nCommunity summary:")
    summary = df.groupby("community").agg(
        n_stocks=("ticker", "count"),
        top_stock=("ticker", "first"),
        sectors=("sector", lambda x: ", ".join(x.unique()[:3]))
    )
    print(summary.to_string())
    return df


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_embeddings(embeddings: dict, path: str = "data/processed/embeddings.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Embeddings saved → {path}")


def load_embeddings(path: str = "data/processed/embeddings.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import CAC40_TICKERS, SECTOR_MAP
    from graph_builder import fetch_returns, build_correlation_graph, load_graphs

    # 1. Build static graph
    print("=" * 60)
    print("CAC40 Community Detection")
    print("=" * 60)

    returns = fetch_returns(list(CAC40_TICKERS.keys()), period="3y")
    G = build_correlation_graph(returns, threshold=0.5)

    # 2. Community detection — both algorithms
    print("\n[Louvain]")
    louvain_partition = detect_communities_louvain(G)

    print("\n[Leiden]")
    leiden_partition = detect_communities_leiden(G)

    # 3. Compare
    print("\n[Comparison]")
    comparison_df = compare_algorithms(G, louvain_partition, leiden_partition)
    os.makedirs("data/processed", exist_ok=True)
    comparison_df.to_csv("data/processed/community_comparison.csv", index=False)

    # 4. Community summary
    print("\n[Summary — Louvain]")
    summary_df = summarize_communities(G, louvain_partition, SECTOR_MAP, CAC40_TICKERS)
    summary_df.to_csv("data/processed/communities.csv", index=False)

    # 5. node2vec
    print("\n[node2vec]")
    embeddings, n2v_model = train_node2vec(G)
    save_embeddings(embeddings)

    # 6. Link prediction
    print("\n[Link Prediction]")
    predicted_links = predict_links(G, embeddings, top_k=20)
    predicted_links.to_csv("data/processed/predicted_links.csv", index=False)

    # 7. Temporal tracking
    print("\n[Temporal Community Tracking]")
    try:
        rolling_graphs = load_graphs()
        temporal_df = track_communities_over_time(rolling_graphs, algorithm="louvain")
        temporal_df.to_csv("data/processed/temporal_communities.csv")
        print(f"Temporal communities saved → data/processed/temporal_communities.csv")
    except FileNotFoundError:
        print("No rolling graphs found — run graph_builder.py first")

    print("\nDone. Next step: run dashboard.py")
