"""
dashboard.py
------------
Interactive Dash dashboard for CAC40 graph market analysis.

Features:
- Interactive correlation graph with community coloring
- Rolling window replay (dynamic graph evolution)
- Anomaly score timeline with known market events
- Community membership table with sector breakdown
- Link prediction viewer

Usage:
    python src/dashboard.py
    Open http://localhost:8050
"""

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import pandas as pd
import numpy as np
import pickle
import os
import json

# ─── App Init ─────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    title="CAC40 Graph Market Analysis",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# ─── Color Palette ────────────────────────────────────────────────────────────

COMMUNITY_COLORS = [
    "#00d4ff", "#ff6b9d", "#ffd700", "#7b68ee",
    "#00c864", "#ff6b35", "#a8ff78", "#ff8c42"
]

SECTOR_COLORS = {
    "Financials": "#1f77b4", "Technology": "#2ca02c", "Healthcare": "#d62728",
    "Industrials": "#ff7f0e", "Consumer Discretionary": "#9467bd",
    "Consumer Staples": "#8c564b", "Energy": "#e377c2", "Materials": "#7f7f7f",
    "Telecom": "#bcbd22", "Utilities": "#17becf", "Real Estate": "#aec7e8",
    "Communication": "#ffbb78", "Unknown": "#cccccc"
}

DARK_BG = "#0f0f0f"
CARD_BG = "#1a1a2e"
BORDER = "#2d2d44"
TEXT = "#e8e8e8"


# ─── Data Loaders ─────────────────────────────────────────────────────────────

def load_graph(path: str = "data/processed/rolling_graphs.pkl"):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_anomaly_results(path: str = "outputs/anomaly_results.csv") -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


# ─── Graph → Plotly Figure ───────────────────────────────────────────────────

def graph_to_plotly(
    G: nx.Graph,
    partition: dict | None = None,
    color_by: str = "community"
) -> go.Figure:
    """
    Convert a NetworkX graph to an interactive Plotly scatter figure.

    Args:
        G:         NetworkX graph.
        partition: Community dict {ticker: community_id}.
        color_by:  'community' or 'sector'.

    Returns:
        Plotly Figure.
    """
    pos = nx.spring_layout(G, seed=42, k=2.5 / np.sqrt(len(G.nodes()) + 1))

    # ── Edges ──
    edge_x, edge_y, edge_weights = [], [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(data.get("weight", 0.5))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.5, color="rgba(150,150,180,0.25)"),
        hoverinfo="none",
        showlegend=False
    )

    # ── Nodes ──
    node_traces = []
    communities = sorted(set(partition.values())) if partition else [0]

    for comm_id in communities:
        if partition:
            comm_nodes = [n for n, c in partition.items() if c == comm_id and n in pos]
        else:
            comm_nodes = list(G.nodes())

        if not comm_nodes:
            continue

        nx_arr = [pos[n][0] for n in comm_nodes]
        ny_arr = [pos[n][1] for n in comm_nodes]
        degrees = [G.degree(n) for n in comm_nodes]
        sizes = [8 + d * 2 for d in degrees]
        hover = [
            f"<b>{n}</b><br>Community: {comm_id}<br>Degree: {d}"
            for n, d in zip(comm_nodes, degrees)
        ]

        node_traces.append(go.Scatter(
            x=nx_arr, y=ny_arr,
            mode="markers+text",
            marker=dict(
                size=sizes,
                color=COMMUNITY_COLORS[comm_id % len(COMMUNITY_COLORS)],
                line=dict(width=1, color="white"),
                opacity=0.85
            ),
            text=comm_nodes,
            textposition="top center",
            textfont=dict(size=8, color=TEXT),
            hovertext=hover,
            hoverinfo="text",
            name=f"Community {comm_id}",
        ))

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=TEXT),
        showlegend=True,
        legend=dict(bgcolor=CARD_BG, bordercolor=BORDER, font=dict(color=TEXT)),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=10, b=10),
        height=500,
        hovermode="closest"
    )
    return fig


# ─── Anomaly Timeline Figure ──────────────────────────────────────────────────

MARKET_EVENTS = {
    "2020-02-24": "COVID crash",
    "2022-02-24": "Ukraine war",
    "2022-06-13": "ECB shock",
    "2023-03-10": "SVB collapse",
}


def anomaly_timeline_figure(results: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results.index, y=results["anomaly_score"],
        mode="lines", name="Anomaly Score",
        line=dict(color="#00d4ff", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.08)"
    ))

    # Decision boundary
    fig.add_hline(y=0, line_dash="dash", line_color="#ff6b6b", line_width=0.8,
                  annotation_text="Decision boundary", annotation_font_color="#ff6b6b")

    # Anomaly highlights
    anomalies = results[results["is_anomaly"]]
    if len(anomalies):
        fig.add_trace(go.Scatter(
            x=anomalies.index, y=anomalies["anomaly_score"],
            mode="markers", name="Anomaly",
            marker=dict(color="#ff4b4b", size=8, symbol="x")
        ))

    # Market event annotations
    for date_str, label in MARKET_EVENTS.items():
        try:
            event_date = pd.Timestamp(date_str)
            if results.index.min() <= event_date <= results.index.max():
                fig.add_vline(x=event_date, line_color="#ffd700",
                              line_dash="dot", line_width=1)
                fig.add_annotation(
                    x=event_date, y=results["anomaly_score"].max() * 0.85,
                    text=label, showarrow=False,
                    font=dict(color="#ffd700", size=9), textangle=-90
                )
        except Exception:
            pass

    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        xaxis=dict(showgrid=False, title="Date"),
        yaxis=dict(gridcolor=BORDER, title="Anomaly Score"),
        legend=dict(bgcolor=CARD_BG, bordercolor=BORDER),
        margin=dict(l=20, r=20, t=20, b=20),
        height=300
    )
    return fig


# ─── Layout ───────────────────────────────────────────────────────────────────

def header():
    return html.Div([
        html.H2("📈 CAC40 — Community Detection & Anomaly Analysis",
                style={"color": TEXT, "marginBottom": "4px"}),
        html.P("Structural market analysis via correlation graphs, Louvain/Leiden clustering, and Isolation Forest",
               style={"color": "#888", "fontSize": "13px", "margin": 0})
    ], style={"padding": "20px 24px 8px"})


def card(children, style=None):
    base = {"background": CARD_BG, "borderRadius": "10px", "padding": "16px",
            "border": f"1px solid {BORDER}", "marginBottom": "16px"}
    if style:
        base.update(style)
    return html.Div(children, style=base)


app.layout = html.Div(style={"background": DARK_BG, "minHeight": "100vh", "fontFamily": "Arial"}, children=[
    header(),

    html.Div(style={"padding": "0 24px"}, children=[

        # ── Controls ──
        card([
            html.Div(style={"display": "flex", "gap": "24px", "flexWrap": "wrap"}, children=[
                html.Div([
                    html.Label("Correlation Threshold", style={"color": TEXT, "fontSize": "12px"}),
                    dcc.Slider(
                        id="threshold-slider", min=0.3, max=0.8, step=0.05,
                        value=0.5, marks={i/10: str(i/10) for i in range(3, 9)},
                        tooltip={"placement": "bottom"}
                    )
                ], style={"flex": "1", "minWidth": "280px"}),

                html.Div([
                    html.Label("Community Algorithm", style={"color": TEXT, "fontSize": "12px"}),
                    dcc.Dropdown(
                        id="algo-dropdown",
                        options=[{"label": "Louvain", "value": "louvain"},
                                 {"label": "Leiden", "value": "leiden"}],
                        value="louvain",
                        style={"background": CARD_BG, "color": "#000"}
                    )
                ], style={"flex": "1", "minWidth": "200px"}),

                html.Div([
                    html.Label("Window (rolling graphs)", style={"color": TEXT, "fontSize": "12px"}),
                    dcc.Slider(
                        id="window-slider", min=0, max=50, step=1, value=0,
                        tooltip={"placement": "bottom"}
                    )
                ], style={"flex": "2", "minWidth": "300px"}),
            ])
        ]),

        # ── KPI Row ──
        html.Div(id="kpi-row", style={"display": "flex", "gap": "12px", "marginBottom": "16px"}),

        # ── Graph + Communities ──
        html.Div(style={"display": "flex", "gap": "16px"}, children=[
            card([
                html.H4("Correlation Graph", style={"color": TEXT, "margin": "0 0 12px"}),
                dcc.Graph(id="graph-figure", config={"displayModeBar": False})
            ], style={"flex": "2"}),

            card([
                html.H4("Community Members", style={"color": TEXT, "margin": "0 0 12px"}),
                html.Div(id="community-table")
            ], style={"flex": "1", "overflowY": "auto", "maxHeight": "560px"}),
        ]),

        # ── Anomaly Timeline ──
        card([
            html.H4("Anomaly Score Timeline", style={"color": TEXT, "margin": "0 0 12px"}),
            html.Div(id="anomaly-plot")
        ]),

        # ── Link Prediction ──
        card([
            html.H4("Top Predicted Links (node2vec)", style={"color": TEXT, "margin": "0 0 12px"}),
            html.Div(id="link-table")
        ]),
    ])
])


# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("kpi-row", "children"),
    Output("graph-figure", "figure"),
    Output("community-table", "children"),
    Output("anomaly-plot", "children"),
    Output("link-table", "children"),
    Input("threshold-slider", "value"),
    Input("algo-dropdown", "value"),
    Input("window-slider", "value"),
)
def update_dashboard(threshold, algorithm, window_idx):
    from graph_builder import fetch_returns, build_correlation_graph
    from community import detect_communities_louvain, detect_communities_leiden
    from data_loader import CAC40_TICKERS, SECTOR_MAP

    # ── Load or build graph ──
    rolling_graphs = load_graph()

    if rolling_graphs and window_idx < len(rolling_graphs):
        G = rolling_graphs[window_idx]["graph"]
        window_label = f"{rolling_graphs[window_idx]['start'].date()} → {rolling_graphs[window_idx]['end'].date()}"
    else:
        returns = fetch_returns(period="1y")
        G = build_correlation_graph(returns, threshold=threshold)
        window_label = "Full period (static)"

    # ── Community detection ──
    if G.number_of_edges() > 0:
        if algorithm == "louvain":
            partition = detect_communities_louvain(G)
        else:
            partition = detect_communities_leiden(G)
    else:
        partition = {n: 0 for n in G.nodes()}

    n_communities = len(set(partition.values()))

    # ── KPIs ──
    kpis = [
        ("Nodes", G.number_of_nodes()),
        ("Edges", G.number_of_edges()),
        ("Communities", n_communities),
        ("Density", f"{nx.density(G):.3f}"),
        ("Window", window_label),
    ]
    kpi_row = [
        html.Div([
            html.Div(str(val), style={"fontSize": "22px", "fontWeight": "bold", "color": "#00d4ff"}),
            html.Div(label, style={"fontSize": "11px", "color": "#888"})
        ], style={"background": CARD_BG, "borderRadius": "8px", "padding": "12px 16px",
                  "border": f"1px solid {BORDER}", "flex": "1", "textAlign": "center"})
        for label, val in kpis
    ]

    # ── Graph figure ──
    graph_fig = graph_to_plotly(G, partition)

    # ── Community table ──
    rows = []
    for ticker, comm in sorted(partition.items(), key=lambda x: x[1]):
        rows.append({"Ticker": ticker,
                     "Name": CAC40_TICKERS.get(ticker, ticker)[:18],
                     "Community": comm,
                     "Sector": SECTOR_MAP.get(ticker, "?")[:20],
                     "Degree": G.degree(ticker) if ticker in G else 0})
    comm_df = pd.DataFrame(rows)
    comm_table = dash_table.DataTable(
        data=comm_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in comm_df.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": DARK_BG, "color": TEXT, "fontWeight": "bold", "fontSize": "11px"},
        style_cell={"backgroundColor": CARD_BG, "color": TEXT, "fontSize": "11px",
                    "border": f"1px solid {BORDER}", "padding": "4px 8px"},
        style_data_conditional=[
            {"if": {"filter_query": f"{{Community}} = {i}"},
             "borderLeft": f"3px solid {COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]}"}
            for i in range(8)
        ],
        page_size=20,
        sort_action="native"
    )

    # ── Anomaly plot ──
    results = load_anomaly_results()
    if results is not None and "anomaly_score" in results.columns:
        anomaly_component = dcc.Graph(
            figure=anomaly_timeline_figure(results),
            config={"displayModeBar": False}
        )
    else:
        anomaly_component = html.P(
            "Run anomaly.py first to generate anomaly results.",
            style={"color": "#888", "fontSize": "13px"}
        )

    # ── Link prediction table ──
    links_df = load_csv("data/processed/predicted_links.csv")
    if links_df is not None:
        links_table = dash_table.DataTable(
            data=links_df.head(15).round(4).to_dict("records"),
            columns=[{"name": c, "id": c} for c in links_df.columns],
            style_header={"backgroundColor": DARK_BG, "color": TEXT, "fontWeight": "bold"},
            style_cell={"backgroundColor": CARD_BG, "color": TEXT, "fontSize": "12px",
                        "border": f"1px solid {BORDER}"},
            page_size=10
        )
    else:
        links_table = html.P("Run community.py first.", style={"color": "#888", "fontSize": "13px"})

    return kpi_row, graph_fig, comm_table, anomaly_component, links_table


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
