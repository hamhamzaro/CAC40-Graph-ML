"""
graph_builder.py
----------------
Builds a dynamic correlation graph from CAC40 stock returns.

- Computes Pearson correlation matrices over rolling windows
- Thresholds to retain binary edges (|r| > threshold)
- Exports graph objects for downstream community detection and anomaly scoring

Usage:
    python src/graph_builder.py
"""

import pandas as pd
import numpy as np
import networkx as nx
import yfinance as yf
import pickle
import os
from typing import Optional


# ─── CAC40 Tickers ────────────────────────────────────────────────────────────

CAC40_TICKERS = [
    "AI.PA", "AIR.PA", "ALO.PA", "MT.AS", "CS.PA", "BNP.PA", "EN.PA",
    "CAP.PA", "CA.PA", "ACA.PA", "BN.PA", "DSY.PA", "ENGI.PA", "EL.PA",
    "ERF.PA", "RMS.PA", "KER.PA", "OR.PA", "LR.PA", "MC.PA", "ML.PA",
    "ORA.PA", "RI.PA", "PUB.PA", "RNO.PA", "SAF.PA", "SGO.PA", "SAN.PA",
    "SU.PA", "GLE.PA", "STLAM.MI", "STM.PA", "TEP.PA", "HO.PA", "TTE.PA",
    "URW.AS", "VIE.PA", "DG.PA", "VIV.PA", "WLN.PA"
]


# ─── Data Loading ──────────────────────────────────────────────────────────────

def fetch_returns(tickers: list[str], period: str = "3y") -> pd.DataFrame:
    """
    Fetch adjusted closing prices and compute log-returns.

    Args:
        tickers: List of Yahoo Finance ticker symbols.
        period:  Time period string (e.g. '3y', '1y', '6mo').

    Returns:
        DataFrame of daily log-returns, shape (n_days, n_stocks).
    """
    print(f"Fetching data for {len(tickers)} tickers over {period}...")
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(how="all", axis=1)  # drop delisted / missing

    log_returns = np.log(raw / raw.shift(1)).dropna()
    print(f"  → {log_returns.shape[0]} trading days, {log_returns.shape[1]} stocks")
    return log_returns


# ─── Correlation Graph ─────────────────────────────────────────────────────────

def build_correlation_graph(
    returns: pd.DataFrame,
    threshold: float = 0.5,
    method: str = "pearson"
) -> nx.Graph:
    """
    Build a static correlation graph from a returns DataFrame.

    Nodes represent stocks. An edge is added between two stocks if their
    absolute pairwise correlation exceeds `threshold`.

    Args:
        returns:   DataFrame of log-returns (rows=days, cols=tickers).
        threshold: Minimum |correlation| to create an edge.
        method:    Correlation method ('pearson', 'spearman', 'kendall').

    Returns:
        NetworkX Graph with correlation weights on edges.
    """
    corr_matrix = returns.corr(method=method)
    tickers = list(corr_matrix.columns)

    G = nx.Graph()
    G.add_nodes_from(tickers)

    edge_count = 0
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if j <= i:
                continue
            r = corr_matrix.loc[t1, t2]
            if abs(r) >= threshold:
                G.add_edge(t1, t2, weight=round(float(r), 4))
                edge_count += 1

    print(f"  → Graph built: {G.number_of_nodes()} nodes, {edge_count} edges (threshold={threshold})")
    return G


# ─── Dynamic (Rolling) Graphs ─────────────────────────────────────────────────

def build_rolling_graphs(
    returns: pd.DataFrame,
    window: int = 60,
    step: int = 5,
    threshold: float = 0.5
) -> list[dict]:
    """
    Build a sequence of correlation graphs over rolling time windows.

    Args:
        returns:   DataFrame of log-returns.
        window:    Rolling window size in trading days.
        step:      Step size between consecutive windows.
        threshold: Minimum |correlation| to create an edge.

    Returns:
        List of dicts with keys: 'start', 'end', 'graph', 'density', 'avg_degree'.
    """
    graphs = []
    dates = returns.index

    for start_idx in range(0, len(dates) - window, step):
        end_idx = start_idx + window
        window_returns = returns.iloc[start_idx:end_idx]

        G = build_correlation_graph(window_returns, threshold=threshold)
        density = nx.density(G)
        avg_degree = (
            np.mean([d for _, d in G.degree()]) if G.number_of_nodes() > 0 else 0
        )

        graphs.append({
            "start": dates[start_idx],
            "end": dates[end_idx - 1],
            "graph": G,
            "density": density,
            "avg_degree": avg_degree,
            "n_edges": G.number_of_edges(),
        })

    print(f"Built {len(graphs)} rolling graphs (window={window}d, step={step}d)")
    return graphs


# ─── Graph Feature Extraction ──────────────────────────────────────────────────

def extract_graph_features(graphs: list[dict]) -> pd.DataFrame:
    """
    Extract time-series of graph-level structural features.

    Used downstream for anomaly detection (Isolation Forest).

    Args:
        graphs: List of rolling graph dicts from `build_rolling_graphs`.

    Returns:
        DataFrame with one row per window and columns for each feature.
    """
    records = []
    for g in graphs:
        G = g["graph"]
        clustering_values = list(nx.clustering(G, weight="weight").values())

        records.append({
            "date": g["end"],
            "density": g["density"],
            "avg_degree": g["avg_degree"],
            "n_edges": g["n_edges"],
            "avg_clustering": np.mean(clustering_values) if clustering_values else 0,
            "transitivity": nx.transitivity(G),
            "n_components": nx.number_connected_components(G),
        })

    df = pd.DataFrame(records).set_index("date")
    print(f"Extracted features: {df.shape[1]} features over {df.shape[0]} windows")
    return df


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_graphs(graphs: list[dict], path: str = "data/processed/rolling_graphs.pkl") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graphs, f)
    print(f"Saved {len(graphs)} graphs → {path}")


def load_graphs(path: str = "data/processed/rolling_graphs.pkl") -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Fetch returns
    returns = fetch_returns(CAC40_TICKERS, period="3y")

    # 2. Static graph (full period)
    print("\n[Static Graph]")
    G_static = build_correlation_graph(returns, threshold=0.5)

    # 3. Rolling graphs
    print("\n[Rolling Graphs]")
    rolling_graphs = build_rolling_graphs(returns, window=60, step=5, threshold=0.5)

    # 4. Feature extraction
    print("\n[Feature Extraction]")
    features_df = extract_graph_features(rolling_graphs)
    os.makedirs("data/processed", exist_ok=True)
    features_df.to_csv("data/processed/graph_features.csv")
    print(f"Features saved → data/processed/graph_features.csv")

    # 5. Save graphs
    save_graphs(rolling_graphs)

    print("\nDone. Next step: run community.py or anomaly.py")
