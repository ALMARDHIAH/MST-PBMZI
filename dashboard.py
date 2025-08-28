import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import networkx as nx
from sample_data import get_cleaned_pbmzi

# Page configuration
st.set_page_config(
    page_title="PBMZI Stocks Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    return get_cleaned_pbmzi()

cleaned_PBMZI = load_data()

# Sidebar
st.sidebar.title("Filters")

# Year filter
years = list(range(2018, 2024))
selected_years = st.sidebar.multiselect(
    "Select Years",
    years,
    default=years,
    help="Select years to display in the analysis"
)

# Company filter
companies = cleaned_PBMZI.columns[1:].tolist()
selected_companies = st.sidebar.multiselect(
    "Select Companies",
    companies,
    default=companies,
    help="Select companies to display in the analysis"
)

# Filter data based on selections
if selected_years and selected_companies:
    # Filter by years
    cleaned_PBMZI['Year'] = pd.to_datetime(cleaned_PBMZI['Date']).dt.year
    filtered_data = cleaned_PBMZI[cleaned_PBMZI['Year'].isin(selected_years)].copy()
    
    # Filter by companies
    filtered_data = filtered_data[['Date', 'Year'] + selected_companies].copy()
else:
    st.warning("Please select at least one year and one company.")
    st.stop()

# Navigation
page = st.sidebar.selectbox("Select Page", ["PBMZI Stocks Overview", "Interaction of PBMZI Stocks"])

if page == "PBMZI Stocks Overview":
    st.title("PBMZI (2018-2023)")
    st.markdown("---")
    
    if len(filtered_data) == 0:
        st.warning("No data available for the selected filters.")
        st.stop()
    
    # Calculate log returns
    log_return_data = filtered_data.copy()
    for company in selected_companies:
        log_return_data[company] = np.log(filtered_data[company] / filtered_data[company].shift(1))
    
    # Layout with columns
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Stock Prices Trend
        st.subheader("Stock Prices Trend")
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
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig_prices, use_container_width=True)
        
        # 3. 30-Day Rolling Volatility
        st.subheader("30-Day Rolling Volatility")
        volatility_30 = pd.DataFrame()
        volatility_30['Date'] = filtered_data['Date']
        for company in selected_companies:
            volatility_30[company] = log_return_data[company].rolling(window=30).std()
        
        fig_vol30 = go.Figure()
        for company in selected_companies:
            fig_vol30.add_trace(go.Scatter(
                x=volatility_30['Date'],
                y=volatility_30[company],
                mode='lines',
                name=company,
                line=dict(width=2)
            ))
        fig_vol30.update_layout(
            title="30-Day Rolling Volatility Based on PBMZI Logarithmic Return",
            xaxis_title="Date",
            yaxis_title="30-Day Rolling Volatility",
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig_vol30, use_container_width=True)
        
        # 5. Correlation Matrix
        st.subheader("Correlation Matrix")
        returns_for_corr = log_return_data[selected_companies].dropna()
        corr_matrix = returns_for_corr.corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        fig_corr.update_layout(
            title="Correlation Matrix of PBMZI Companies' Log Return",
            height=500,
            xaxis=dict(side="bottom"),
            yaxis=dict(side="left")
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with col2:
        # 2. Returns Trend
        st.subheader("Logarithmic Returns Trend")
        fig_returns = go.Figure()
        for company in selected_companies:
            fig_returns.add_trace(go.Scatter(
                x=filtered_data['Date'],
                y=log_return_data[company],
                mode='lines',
                name=company,
                line=dict(width=2)
            ))
        fig_returns.update_layout(
            title="Logarithmic Return of PBMZI",
            xaxis_title="Date",
            yaxis_title="Logarithmic Return",
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig_returns, use_container_width=True)
        
        # 4. 60-Day Rolling Volatility
        st.subheader("60-Day Rolling Volatility")
        volatility_60 = pd.DataFrame()
        volatility_60['Date'] = filtered_data['Date']
        for company in selected_companies:
            volatility_60[company] = log_return_data[company].rolling(window=60).std()
        
        fig_vol60 = go.Figure()
        for company in selected_companies:
            fig_vol60.add_trace(go.Scatter(
                x=volatility_60['Date'],
                y=volatility_60[company],
                mode='lines',
                name=company,
                line=dict(width=2)
            ))
        fig_vol60.update_layout(
            title="60-Day Rolling Volatility Based on PBMZI Logarithmic Return",
            xaxis_title="Date",
            yaxis_title="60-Day Rolling Volatility",
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
        )
        st.plotly_chart(fig_vol60, use_container_width=True)
        
        # 6. Maximum Drawdown vs Maximum Return
        st.subheader("Maximum Drawdown vs Maximum Return")
        
        def max_drawdown(series):
            peak = series.expanding(min_periods=1).max()
            drawdown = (series - peak) / peak
            return drawdown.min()
        
        def max_return(log_series):
            cumulative = log_series.cumsum().dropna()
            return np.exp(cumulative.max()) - 1
        
        drawdown_data = []
        for company in selected_companies:
            max_dd = max_drawdown(filtered_data[company])
            max_ret = max_return(log_return_data[company].dropna())
            drawdown_data.append({
                'Company': company,
                'Max_Drawdown': max_dd,
                'Max_Return': max_ret
            })
        
        drawdown_df = pd.DataFrame(drawdown_data)
        
        fig_scatter = px.scatter(
            drawdown_df,
            x='Max_Drawdown',
            y='Max_Return',
            color='Company',
            title='Max Return vs Max Drawdown per Company',
            labels={'Max_Drawdown': 'Max Drawdown', 'Max_Return': 'Max Return'},
            height=500
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_scatter.update_layout(showlegend=True)
        st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "Interaction of PBMZI Stocks":
    st.title("Interaction of PBMZI Stocks")
    st.markdown("---")
    
    if len(filtered_data) == 0:
        st.warning("No data available for the selected filters.")
        st.stop()
    
    # Calculate returns for MST
    returns_data = []
    for company in selected_companies:
        company_data = filtered_data[company].dropna()
        if len(company_data) > 1:
            returns = np.log(company_data.values[1:] / company_data.values[:-1])
            returns_data.append(returns)
        else:
            returns_data.append(np.array([]))
    
    # Ensure all return series have the same length
    min_length = min(len(r) for r in returns_data if len(r) > 0)
    if min_length == 0:
        st.warning("Insufficient data to calculate returns.")
        st.stop()
    
    returns_matrix = np.array([r[:min_length] for r in returns_data if len(r) > 0]).T
    valid_companies = [comp for i, comp in enumerate(selected_companies) if len(returns_data[i]) > 0]
    
    # Calculate correlation and distance matrices
    if len(valid_companies) < 2:
        st.warning("Need at least 2 companies to create MST.")
        st.stop()
    
    corremat = np.corrcoef(returns_matrix.T)
    distmat = np.sqrt(2 * (1 - corremat))
    
    # Create MST
    G = nx.Graph()
    G.add_nodes_from(valid_companies)
    
    for i in range(len(valid_companies)):
        for j in range(i+1, len(valid_companies)):
            if not np.isnan(distmat[i, j]):
                G.add_edge(valid_companies[i], valid_companies[j], weight=distmat[i, j])
    
    if len(G.edges()) == 0:
        st.warning("Unable to create MST with current data.")
        st.stop()
    
    MST = nx.minimum_spanning_tree(G, weight="weight", algorithm="kruskal")
    
    # Create network visualization
    pos = nx.spring_layout(MST, k=3, iterations=50)
    
    # Extract edge information
    edge_x = []
    edge_y = []
    edge_info = []
    
    for edge in MST.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        weight = MST[edge[0]][edge[1]]['weight']
        edge_info.append(f"{edge[0]} - {edge[1]}: {weight:.3f}")
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='lightblue'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Extract node information
    node_x = []
    node_y = []
    node_text = []
    node_info = []
    
    for node in MST.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        adjacencies = list(MST.neighbors(node))
        node_info.append(f'{node}<br>Connections: {len(adjacencies)}<br>Connected to: {", ".join(adjacencies)}')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="middle center",
        hovertext=node_info,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            reversescale=True,
            color=[],
            size=30,
            colorbar=dict(
                thickness=15,
                len=0.5,
                x=1.02,
                title="Node Connections"
            ),
            line=dict(width=2, color='black')
        )
    )
    
    # Color nodes by degree
    node_adjacencies = []
    for node in MST.nodes():
        node_adjacencies.append(len(list(MST.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    # Create figure
    fig_mst = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=f'Minimum Spanning Tree for PBMZI Log Return ({min(selected_years)}-{max(selected_years)})',
                           titlefont_size=16,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20,l=5,r=5,t=40),
                           annotations=[ dict(
                               text="Click on nodes to see connections",
                               showarrow=False,
                               xref="paper", yref="paper",
                               x=0.005, y=-0.002,
                               xanchor='left', yanchor='bottom',
                               font=dict(color="gray", size=12)
                           )],
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=600
                       ))
    
    st.plotly_chart(fig_mst, use_container_width=True)
    
    # Display MST statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Nodes", len(MST.nodes()))
    with col2:
        st.metric("Total Edges", len(MST.edges()))
    with col3:
        avg_distance = np.mean([MST[edge[0]][edge[1]]['weight'] for edge in MST.edges()])
        st.metric("Average Distance", f"{avg_distance:.3f}")
    
    # Show edge weights table
    st.subheader("MST Edge Weights")
    edge_data = []
    for edge in MST.edges():
        weight = MST[edge[0]][edge[1]]['weight']
        edge_data.append({
            'Company 1': edge[0],
            'Company 2': edge[1],
            'Distance': round(weight, 3)
        })
    
    edge_df = pd.DataFrame(edge_data).sort_values('Distance')
    st.dataframe(edge_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**PBMZI Stocks Dashboard** - Interactive analysis of stock market data")
