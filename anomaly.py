"""
anomaly.py
----------
Anomaly detection on CAC40 market structure using graph-level features.

Applies Isolation Forest on time-series of structural graph metrics
(density, clustering, degree, modularity) to detect market regime shifts
and structural anomalies (crashes, liquidity crises, correlation breakdowns).

Usage:
    python src/anomaly.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import json
from typing import Optional


# ─── Known Market Events (for backtesting / annotation) ───────────────────────

MARKET_EVENTS = {
    "2020-02-24": "COVID crash begins",
    "2020-03-12": "Black Thursday (COVID)",
    "2022-02-24": "Russia-Ukraine war",
    "2022-06-13": "ECB rate hike shock",
    "2023-03-10": "SVB collapse",
}


# ─── Feature Loading ───────────────────────────────────────────────────────────

def load_features(path: str = "data/processed/graph_features.csv") -> pd.DataFrame:
    """Load pre-computed graph features from CSV."""
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    print(f"Loaded features: {df.shape[0]} windows × {df.shape[1]} features")
    return df


# ─── Isolation Forest Pipeline ────────────────────────────────────────────────

def train_isolation_forest(
    features: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
    random_state: int = 42
) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    """
    Train an Isolation Forest model on graph structural features.

    Args:
        features:      DataFrame of graph-level features (time-indexed).
        contamination: Expected proportion of anomalies (default 5%).
        n_estimators:  Number of trees in the forest.
        random_state:  Reproducibility seed.

    Returns:
        Tuple of (fitted pipeline, predictions array, anomaly scores array).
        Predictions: -1 = anomaly, 1 = normal.
        Scores: lower = more anomalous.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    pipeline.fit(features)
    predictions = pipeline.predict(features)
    scores = pipeline.decision_function(features)  # higher = more normal

    n_anomalies = (predictions == -1).sum()
    print(f"Detected {n_anomalies} anomalous windows ({n_anomalies / len(predictions):.1%} of total)")
    return pipeline, predictions, scores


# ─── Results Assembly ─────────────────────────────────────────────────────────

def build_results_df(
    features: pd.DataFrame,
    predictions: np.ndarray,
    scores: np.ndarray
) -> pd.DataFrame:
    """Attach predictions and scores to the feature DataFrame."""
    df = features.copy()
    df["anomaly_score"] = scores
    df["is_anomaly"] = (predictions == -1)
    df["alert_level"] = pd.cut(
        -scores,  # invert: higher = worse
        bins=[-np.inf, 0.05, 0.15, np.inf],
        labels=["normal", "warning", "critical"]
    )
    return df


# ─── Backtesting ──────────────────────────────────────────────────────────────

def evaluate_against_events(
    results: pd.DataFrame,
    events: dict = MARKET_EVENTS,
    tolerance_days: int = 10
) -> dict:
    """
    Check if known market events were flagged as anomalies (within tolerance window).

    Args:
        results:        Results DataFrame with 'is_anomaly' column.
        events:         Dict of {date_str: event_name}.
        tolerance_days: Days around event date to consider a hit.

    Returns:
        Dict with hit/miss for each event.
    """
    anomaly_dates = results[results["is_anomaly"]].index
    evaluation = {}

    for date_str, event_name in events.items():
        event_date = pd.Timestamp(date_str)
        window_start = event_date - pd.Timedelta(days=tolerance_days)
        window_end = event_date + pd.Timedelta(days=tolerance_days)

        hit = any((anomaly_dates >= window_start) & (anomaly_dates <= window_end))
        evaluation[event_name] = {
            "date": date_str,
            "detected": hit,
            "status": "✅ HIT" if hit else "❌ MISS"
        }
        print(f"  {evaluation[event_name]['status']}  {event_name} ({date_str})")

    precision = sum(1 for v in evaluation.values() if v["detected"]) / len(evaluation)
    print(f"\nEvent detection rate: {precision:.0%}")
    return evaluation


# ─── Visualization ────────────────────────────────────────────────────────────

def plot_anomaly_timeline(
    results: pd.DataFrame,
    events: Optional[dict] = None,
    save_path: str = "outputs/anomaly_timeline.png"
) -> None:
    """
    Plot anomaly scores over time with flagged windows and known events.

    Args:
        results:   Results DataFrame.
        events:    Known market events to annotate.
        save_path: Output file path for the figure.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")

    # ── Top: anomaly score time series ──
    ax1 = axes[0]
    ax1.plot(results.index, results["anomaly_score"], color="#00d4ff", linewidth=1.2, label="Anomaly Score")
    ax1.axhline(0, color="#ff6b6b", linestyle="--", linewidth=0.8, alpha=0.7, label="Decision boundary")
    ax1.fill_between(
        results.index, results["anomaly_score"], 0,
        where=(results["anomaly_score"] < 0),
        alpha=0.3, color="#ff6b6b", label="Anomalous region"
    )
    ax1.set_ylabel("Anomaly Score", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax1.set_title("CAC40 Market Structure — Anomaly Detection (Isolation Forest)", color="white", fontsize=12)

    # ── Bottom: graph density ──
    ax2 = axes[1]
    ax2.fill_between(results.index, results["density"], alpha=0.6, color="#7b68ee", label="Graph Density")
    ax2.plot(results.index, results["density"], color="#9b88ff", linewidth=0.8)
    ax2.set_ylabel("Graph Density", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
    ax2.set_xlabel("Date", color="white")

    # ── Annotate known events ──
    if events:
        for date_str, label in events.items():
            event_date = pd.Timestamp(date_str)
            for ax in axes:
                ax.axvline(event_date, color="#ffd700", linestyle=":", linewidth=1, alpha=0.8)
            axes[0].annotate(
                label, xy=(event_date, axes[0].get_ylim()[1] * 0.85),
                fontsize=7, color="#ffd700", rotation=90, ha="right"
            )

    # ── Flag anomaly windows ──
    anomaly_dates = results[results["is_anomaly"]].index
    for ax in axes:
        for d in anomaly_dates:
            ax.axvspan(
                d - pd.Timedelta(days=2),
                d + pd.Timedelta(days=2),
                alpha=0.15, color="#ff6b6b"
            )

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, color="white")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {save_path}")
    plt.show()


# ─── Export ───────────────────────────────────────────────────────────────────

def export_alerts(results: pd.DataFrame, path: str = "outputs/alerts.json") -> None:
    """Export anomaly alerts to JSON for dashboard consumption."""
    alerts = results[results["is_anomaly"]][["anomaly_score", "alert_level"]].copy()
    alerts.index = alerts.index.strftime("%Y-%m-%d")
    alerts["alert_level"] = alerts["alert_level"].astype(str)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(alerts.to_dict(orient="index"), f, indent=2)
    print(f"Alerts exported → {path} ({len(alerts)} anomalies)")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("CAC40 Anomaly Detection — Isolation Forest on Graph Features")
    print("=" * 60)

    # 1. Load pre-computed graph features
    features = load_features("data/processed/graph_features.csv")

    # 2. Train Isolation Forest
    print("\n[Training]")
    pipeline, predictions, scores = train_isolation_forest(
        features, contamination=0.05, n_estimators=200
    )

    # 3. Build results
    results = build_results_df(features, predictions, scores)
    results.to_csv("outputs/anomaly_results.csv")
    print(f"Results saved → outputs/anomaly_results.csv")

    # 4. Backtest against known events
    print("\n[Backtesting]")
    evaluation = evaluate_against_events(results, MARKET_EVENTS)

    # 5. Visualize
    print("\n[Visualization]")
    plot_anomaly_timeline(results, MARKET_EVENTS)

    # 6. Export alerts
    export_alerts(results)

    print("\nDone. Run dashboard.py to explore results interactively.")
