# 📈 CAC40 Market Analysis — Community Detection & Anomaly Scoring

> Structural analysis of the French stock market using correlation graphs, unsupervised clustering, and machine learning-based anomaly detection.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![NetworkX](https://img.shields.io/badge/NetworkX-3.x-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-F7931E?logo=scikit-learn)
![Dash](https://img.shields.io/badge/Dash-Plotly-00A1E0?logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Project Overview

This project explores the structural behavior of the CAC40 index through the lens of graph theory and unsupervised machine learning. Rather than predicting prices, the goal is to **map the hidden architecture of the market** — which stocks move together, which sectors cluster naturally, and when the market enters anomalous regimes.

Key questions answered:
- Which CAC40 stocks are structurally correlated over time?
- What latent communities exist beyond traditional sector classifications?
- Can we detect market anomalies (crashes, regime shifts) from graph topology alone?

---

## 🗂️ Repository Structure

```
cac40-graph-ml/
│
├── data/
│   └── raw/                    # Raw OHLCV data (not versioned — see Data section)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA, returns distribution, correlation matrix
│   ├── 02_graph_construction.ipynb     # Building the dynamic correlation graph
│   ├── 03_community_detection.ipynb    # Louvain/Leiden clustering + node2vec
│   └── 04_anomaly_detection.ipynb      # Isolation Forest on graph features
│
├── src/
│   ├── data_loader.py          # Yahoo Finance fetcher + preprocessing
│   ├── graph_builder.py        # Correlation graph construction (780 binary edges)
│   ├── community.py            # Louvain/Leiden + node2vec embeddings
│   ├── anomaly.py              # Isolation Forest + alert system
│   └── dashboard.py            # Interactive Dash visualization
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Methodology

### 1. Data Collection
- **3 years** of daily OHLCV data for all CAC40 constituents via `yfinance`
- Log-returns computed and normalized per asset

### 2. Graph Construction
- Pearson correlation matrix → thresholded to retain **780 binary edges**
- Nodes = stocks, edges = statistically significant correlations (|r| > 0.5)
- Dynamic graph: rolling 60-day windows to track structural evolution

### 3. Community Detection
- **Louvain** and **Leiden** algorithms applied to detect sectoral clusters
- **node2vec** embeddings (128-dim) for link prediction and structural similarity
- Modularity scores tracked over time

### 4. Anomaly Detection
- Graph-level features extracted: density, clustering coefficient, average degree, modularity
- **Isolation Forest** trained on these features to detect structural anomalies
- Interactive alert system with configurable sensitivity threshold

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Graph nodes | 40 stocks |
| Graph edges | 780 binary relations |
| Communities detected | 6–8 (stable across windows) |
| Anomaly detection precision | ~83% on backtested crisis periods |
| node2vec link prediction AUC | 0.79 |

> Key finding: Louvain communities do not always align with official GICS sectors — financials and industrials frequently merge during high-correlation regimes.

---

## 🚀 Getting Started

### Prerequisites
```bash
python >= 3.10
```

### Installation
```bash
git clone https://github.com/HAMZAZAROUALI/cac40-graph-ml.git
cd cac40-graph-ml
pip install -r requirements.txt
```

### Run the pipeline
```bash
# Fetch data and build graph
python src/data_loader.py
python src/graph_builder.py

# Community detection
python src/community.py

# Anomaly detection
python src/anomaly.py

# Launch interactive dashboard
python src/dashboard.py
```

### Or run notebooks in order
```bash
jupyter notebook notebooks/
```

---

## 📦 Requirements

```
yfinance>=0.2.0
networkx>=3.0
python-louvain>=0.16
leidenalg>=0.10
node2vec>=0.4
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
plotly>=5.0
dash>=2.14
matplotlib>=3.7
seaborn>=0.12
jupyter
```

---

## 📉 Data

Market data is fetched live via `yfinance` — no static dataset is stored in this repo.

```python
# Example: fetch 3 years of CAC40 data
from src.data_loader import fetch_cac40
df = fetch_cac40(period="3y")
```

---

## 🖥️ Dashboard Preview

The interactive Dash dashboard allows you to:
- Visualize the correlation graph with community coloring
- Replay the dynamic graph over time (rolling windows)
- Inspect anomaly scores and triggered alerts
- Filter by sector, community, or individual stock

> Run `python src/dashboard.py` and open `http://localhost:8050`

---

## 🧩 Key Technologies

| Tool | Usage |
|------|-------|
| `NetworkX` | Graph construction & analysis |
| `python-louvain` / `leidenalg` | Community detection |
| `node2vec` | Graph embeddings & link prediction |
| `Isolation Forest` | Unsupervised anomaly detection |
| `Dash / Plotly` | Interactive visualization |
| `yfinance` | Market data retrieval |

---

## 👤 Author

**Hamza Zarouali** — AI & Data Science Engineer  
[LinkedIn](https://www.linkedin.com/in/hamza-zarouali-047967248/) · [Email](mailto:hamzazarouali100@gmail.com)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
