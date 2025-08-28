import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from sample_data import generate_pbmzi_data

# Page configuration
st.set_page_config(
    page_title="PBMZI Stocks Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the PBMZI data"""
    return generate_pbmzi_data()

def calculate_log_returns(df):
    """Calculate logarithmic returns"""
    returns_df = df.copy()
    for col in df.columns[1:]:
        returns_df[col] = np.log(df[col] / df[col].shift(1))
    return returns_df

def calculate_volatility(df, window):
    """Calculate rolling volatility"""
    log_returns = calculate_log_returns(df)
    volatility = log_returns[df.columns[1:]].rolling(window=window).std()
    return volatility

def max_drawdown(series):
    """Calculate maximum drawdown"""
    peak = series.expanding(min_periods=1).max()
    drawdown = (series - peak) / peak
    return drawdown.min()

def max_return(log_series):
    """Calculate maximum return using log returns"""
    cumulative = log_series.cumsum().dropna()
    return np.exp(cumulative.max()) - 1

def create_mst_network(df, selected_years=None):
    """Create MST network from correlation data"""
    if selected_years:
        # Filter data for selected years
        df_filtered = df[df['Date'].dt.year.isin(selected_years)]
    else:
        df_filtered = df
    
    # Calculate log returns
    returns = np.log(df_filtered.iloc[1:, 1:].values / df_filtered.iloc[:-1, 1:].values)
    returns_df = pd.DataFrame(returns, columns=df.columns[1:])
    
    # Correlation and distance matrix
    corremat = np.corrcoef(returns_df.T)
    distmat = np.sqrt(2 * (1 - corremat))
    
    # Create graph and MST
    companies = returns_df.columns.tolist()
    G = nx.Graph()
    G.add_nodes_from(companies)
    
    for i in range(len(companies)):
        for j in range(i+1, len(companies)):
            G.add_edge(companies[i], companies[j], weight=distmat[i, j])
    
    MST = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    
    return MST, distmat, companies

def plot_mst_network(MST, companies, title="MST Network"):
    """Create interactive MST network plot using Plotly"""
    pos = nx.kamada_kawai_layout(MST)
    
    # Extract edges
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in MST.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(MST[edge[0]][edge[1]]['weight'])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='lightblue'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Extract nodes
    node_x = []
    node_y = []
    node_text = []
    node_sizes = []
    
    for node in MST.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_sizes.append(MST.degree(node) * 10 + 20)  # Size based on degree
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        marker=dict(
            size=node_sizes,
            color='lightcoral',
            line=dict(width=2, color='darkred')
        )
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title=title,
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[ dict(
                           text="Click on nodes to highlight connections",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0.005, y=-0.002,
                           xanchor="left", yanchor="bottom",
                           font=dict(color="gray", size=12)
                       )],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       height=500
                   ))
    
    return fig

# Load data
cleaned_PBMZI = load_data()

# Sidebar
st.sidebar.title("📊 Dashboard Controls")

# Page selection
page = st.sidebar.selectbox(
    "Select Page",
    ["PBMZI Stocks Overview", "Interaction of PBMZI Stocks"]
)

# Year filter
st.sidebar.subheader("📅 Year Filter")
available_years = sorted(cleaned_PBMZI['Date'].dt.year.unique())
selected_years = st.sidebar.multiselect(
    "Select Years",
    available_years,
    default=available_years
)

# Company filter
st.sidebar.subheader("🏢 Company Filter")
available_companies = cleaned_PBMZI.columns[1:].tolist()
selected_companies = st.sidebar.multiselect(
    "Select Companies",
    available_companies,
    default=available_companies
)

# Filter data based on selections
if selected_years and selected_companies:
    filtered_data = cleaned_PBMZI[
        (cleaned_PBMZI['Date'].dt.year.isin(selected_years))
    ][['Date'] + selected_companies]
else:
    filtered_data = cleaned_PBMZI

# Page 1: PBMZI Stocks Overview
if page == "PBMZI Stocks Overview":
    st.title("📈 PBMZI (2018-2023)")
    st.markdown("---")
    
    if filtered_data.empty or len(selected_companies) == 0:
        st.warning("Please select at least one year and one company to view the analysis.")
    else:
        # Row 1: Stock Prices and Returns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💹 Stock Prices Trend")
            fig_prices = go.Figure()
            for company in selected_companies:
                fig_prices.add_trace(go.Scatter(
                    x=filtered_data['Date'],
                    y=filtered_data[company],
                    mode='lines',
                    name=company,
                    line=dict(width=2)
                ))
            fig_prices.update_layout(
                title="Price Movement of PBMZI",
                xaxis_title="Date",
                yaxis_title="Price (~M$)",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_prices, use_container_width=True)
        
        with col2:
            st.subheader("📊 Returns Trend")
            log_returns = calculate_log_returns(filtered_data)
            fig_returns = go.Figure()
            for company in selected_companies:
                fig_returns.add_trace(go.Scatter(
                    x=filtered_data['Date'],
                    y=log_returns[company],
                    mode='lines',
                    name=company,
                    line=dict(width=2)
                ))
            fig_returns.update_layout(
                title="Logarithmic Return of PBMZI",
                xaxis_title="Date",
                yaxis_title="Logarithmic Return",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_returns, use_container_width=True)
        
        # Row 2: Volatility Charts
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("📈 30-Day Rolling Volatility")
            volatility_30 = calculate_volatility(filtered_data, 30)
            fig_vol30 = go.Figure()
            for company in selected_companies:
                fig_vol30.add_trace(go.Scatter(
                    x=filtered_data['Date'],
                    y=volatility_30[company],
                    mode='lines',
                    name=company,
                    line=dict(width=2)
                ))
            fig_vol30.update_layout(
                title="30-Day Rolling Volatility",
                xaxis_title="Date",
                yaxis_title="30-Day Rolling Volatility",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_vol30, use_container_width=True)
        
        with col4:
            st.subheader("📈 60-Day Rolling Volatility")
            volatility_60 = calculate_volatility(filtered_data, 60)
            fig_vol60 = go.Figure()
            for company in selected_companies:
                fig_vol60.add_trace(go.Scatter(
                    x=filtered_data['Date'],
                    y=volatility_60[company],
                    mode='lines',
                    name=company,
                    line=dict(width=2)
                ))
            fig_vol60.update_layout(
                title="60-Day Rolling Volatility",
                xaxis_title="Date",
                yaxis_title="60-Day Rolling Volatility",
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig_vol60, use_container_width=True)
        
        # Row 3: Correlation and Drawdown Analysis
        col5, col6 = st.columns(2)
        
        with col5:
            st.subheader("🔥 Correlation Matrix")
            log_returns_corr = calculate_log_returns(filtered_data)
            correlation_matrix = log_returns_corr[selected_companies].corr()
            
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
            ))
            fig_corr.update_layout(
                title="Correlation Matrix of PBMZI Companies' Log Return",
                height=400
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col6:
            st.subheader("💰 Max Return vs Max Drawdown")
            log_returns_dd = calculate_log_returns(filtered_data)
            
            drawdown_data = []
            for company in selected_companies:
                max_dd = max_drawdown(filtered_data[company])
                max_ret = max_return(log_returns_dd[company])
                drawdown_data.append({
                    'Company': company,
                    'Max Drawdown': max_dd,
                    'Max Return': max_ret
                })
            
            drawdown_df = pd.DataFrame(drawdown_data)
            
            fig_scatter = px.scatter(
                drawdown_df,
                x='Max Drawdown',
                y='Max Return',
                color='Company',
                title='Max Return vs Max Drawdown per Company',
                hover_data=['Company']
            )
            fig_scatter.update_layout(height=400)
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_scatter, use_container_width=True)

# Page 2: Interaction of PBMZI Stocks
elif page == "Interaction of PBMZI Stocks":
    st.title("🕸️ Interaction of PBMZI Stocks")
    st.markdown("---")
    
    if not selected_years:
        st.warning("Please select at least one year to view the MST analysis.")
    else:
        # Overall MST for 2018-2023
        st.subheader("🌐 MST Network (2018-2023)")
        mst_overall, distmat_overall, companies_overall = create_mst_network(cleaned_PBMZI)
        fig_mst_overall = plot_mst_network(mst_overall, companies_overall, "Minimum Spanning Tree for PBMZI Log Return (2018-2023)")
        st.plotly_chart(fig_mst_overall, use_container_width=True)
        
        # Additional MSTs for selected years if different from all years
        if len(selected_years) < len(available_years):
            st.subheader(f"🔍 MST Network for Selected Years: {', '.join(map(str, selected_years))}")
            mst_selected, distmat_selected, companies_selected = create_mst_network(cleaned_PBMZI, selected_years)
            fig_mst_selected = plot_mst_network(mst_selected, companies_selected, f"MST for Years {', '.join(map(str, selected_years))}")
            st.plotly_chart(fig_mst_selected, use_container_width=True)
        
        # Network Statistics
        st.subheader("📊 Network Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Nodes", len(mst_overall.nodes()))
        with col2:
            st.metric("Number of Edges", len(mst_overall.edges()))
        with col3:
            avg_distance = np.mean([mst_overall[u][v]['weight'] for u, v in mst_overall.edges()])
            st.metric("Average Distance", f"{avg_distance:.3f}")
        
        # Node Degree Analysis
        st.subheader("🎯 Node Connectivity Analysis")
        degrees = dict(mst_overall.degree())
        degree_df = pd.DataFrame(list(degrees.items()), columns=['Company', 'Connections'])
        degree_df = degree_df.sort_values('Connections', ascending=False)
        
        fig_degree = px.bar(
            degree_df,
            x='Company',
            y='Connections',
            title='Number of Connections per Company in MST',
            color='Connections',
            color_continuous_scale='viridis'
        )
        fig_degree.update_layout(height=400)
        st.plotly_chart(fig_degree, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        📈 PBMZI Stocks Dashboard | Built with Streamlit & Plotly
    </div>
    """,
    unsafe_allow_html=True
)
