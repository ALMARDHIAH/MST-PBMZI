import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

# =========================
# Setup
# =========================
st.set_page_config(layout="wide")

# =========================
# Load and prepare the data
# =========================
@st.cache_data
def load_data():
    df = pd.read_excel("cleaned_PBMZI.xlsx")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

cleaned_PBMZI = load_data()
available_years = sorted(cleaned_PBMZI['Date'].dt.year.unique())

# =========================
# Sidebar Navigation & Filters
# =========================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["PBMZI (2018-2023)", "MST Overview"])

all_companies = cleaned_PBMZI.columns[1:]

if page == "PBMZI (2018-2023)":
    selected_companies = st.sidebar.multiselect(
        "Select companies:", all_companies, default=list(all_companies)
    )
    selected_years = st.sidebar.multiselect(
        "Select years:", available_years, default=available_years
    )
    if not selected_years:
        selected_years = list(range(2018, 2024))

elif page == "MST Overview":
    selected_years = st.sidebar.multiselect(
        "Select year(s):", available_years, default=available_years
    )
    if not selected_years:
        selected_years = list(range(2018, 2024))

# =========================
# PAGE 1 – PBMZI EDA
# =========================
if page == "PBMZI (2018-2023)":
    st.title("PBMZI (2018-2023)")

    filtered_data = cleaned_PBMZI[cleaned_PBMZI['Date'].dt.year.isin(selected_years)]

    # 1. Price Movement
    st.subheader("Price Movement Overview")
    fig1 = px.line(filtered_data, x="Date", y=selected_companies,
                   labels={"value": "Price (~M$)", "Date": "Date"},
                   title=f"Price Movement of PBMZI ({min(selected_years)}–{max(selected_years)})")
    st.plotly_chart(fig1, use_container_width=True)

    # 2. Return Movement
    st.subheader("Logarithmic Return Movement")
    log_return = filtered_data[selected_companies].apply(lambda col: np.log(col / col.shift(1)))
    import plotly.express as px

    # Prepare data in long form
    df_long = log_return[selected_companies].copy()
    df_long["Date"] = filtered_data["Date"]
    df_long = df_long.melt(id_vars="Date", var_name="Company", value_name="Log Return")

    # Plot
    fig2 = px.line(
        df_long,
        x="Date",
        y="Log Return",
        color="Company",  # separate line per company
        title=f"Logarithmic Return Movement of PLCs ({min(selected_years)}–{max(selected_years)})"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # 3. Volatility
    st.subheader("60-Day Rolling Volatility")
    volatility_return = log_return.rolling(window=60).std()
    fig3 = px.line(x=filtered_data['Date'], y=volatility_return[selected_companies],
                   labels={"x": "Date", "value": "60-Day Rolling Volatility"},
                   title="60-Day Rolling Volatility Based on PBMZI Logarithmic Return")
    st.plotly_chart(fig3, use_container_width=True)

    # 4. Correlation Matrix
    st.subheader("Correlation Matrix of Log Return")
    corr_matrix = log_return.corr(method='pearson')
    fig4 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1,
                     labels=dict(color="Correlation"),
                     title="Correlation Matrix of PBMZI Companies' Log Return")
    st.plotly_chart(fig4, use_container_width=True)

    # 5. Correlation Distribution
    st.subheader("Distribution of Pairwise Correlations (PBMZI Log Returns)")
    mask = np.tril(np.ones(corr_matrix.shape), k=-1).astype(bool)
    corr_values = corr_matrix.where(mask).stack().values
    bins = [-1.0, -0.7, -0.4, 0, 0.4, 0.7, 1.0]
    labels = [
        'Highly Strong Negative [-1.0, -0.7)',
        'Strong Negative [-0.7, -0.4)',
        'Weak Negative [-0.4, 0)',
        'Weak Positive [0, 0.4)',
        'Strong Positive [0.4, 0.7)',
        'Highly Strong Positive [0.7, 1.0]'
    ]
    categories = pd.cut(corr_values, bins=bins, labels=labels, right=False)
    dist = categories.value_counts().sort_index()
    fig5 = px.bar(x=dist.index, y=dist.values, text=dist.values,
                  labels={"x": "Correlation Category", "y": "Number of Pairs"},
                  title="Distribution of Pairwise Correlations")
    st.plotly_chart(fig5, use_container_width=True)


# =========================
# PAGE 2 – MST Overview
# =========================
elif page == "MST Overview":
    st.title("MST Overview")

    filtered_data = cleaned_PBMZI[cleaned_PBMZI['Date'].dt.year.isin(selected_years)]

    if filtered_data.shape[0] > 1:
        returns = np.log(filtered_data.iloc[1:, 1:].values /
                         filtered_data.iloc[:-1, 1:].values)
        returns_df = pd.DataFrame(returns, columns=filtered_data.columns[1:])

        corremat = np.corrcoef(returns_df.T)
        distmat = np.round(np.sqrt(2 * (1 - corremat)), 2)

        companies = returns_df.columns.tolist()
        G = nx.Graph()
        G.add_nodes_from(companies)
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                G.add_edge(companies[i], companies[j], weight=distmat[i, j])

        MST_PBMZI = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")

        # Pyvis interactive network
        net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="black")
        net.from_nx(MST_PBMZI)

        # Add edge weights to tooltip
        for u, v, d in MST_PBMZI.edges(data=True):
            net.add_edge(u, v, title=f"Distance: {d['weight']:.2f}")

        # Highlight neighbors when clicked
        net.toggle_physics(True)
        net.set_options("""
        const options = {
          nodes: { borderWidth: 2, size: 20, color: { border: '#222', background: '#97C2FC' } },
          edges: { color: { color:'#848484' }, smooth: false },
          interaction: { hover: true, multiselect: true, dragNodes:true, dragView: true, zoomView: true }
        }
        """)

        net.save_graph("mst_network.html")
        HtmlFile = open("mst_network.html", "r", encoding="utf-8")
        components.html(HtmlFile.read(), height=800)
    else:
        st.warning("Not enough data for the selected year to build MST.")
