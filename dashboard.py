import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from utils import (
    load_data, filter_data_by_year, filter_data_by_companies,
    calculate_log_returns, calculate_rolling_volatility,
    calculate_max_drawdown, calculate_max_return,
    create_mst_network, create_network_plot
)

# Page configuration
st.set_page_config(
    page_title="PBMZI Stocks Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def get_data():
    return load_data()

df = get_data()

# Sidebar
st.sidebar.title("📊 Dashboard Filters")

# Year filter
st.sidebar.subheader("Select Years")
available_years = sorted(df['Date'].dt.year.unique())
selected_years = []
for year in available_years:
    if st.sidebar.checkbox(f"{year}", value=True, key=f"year_{year}"):
        selected_years.append(year)

# Company filter
st.sidebar.subheader("Select Companies")
available_companies = list(df.columns[1:])  # Exclude Date column
selected_companies = []
for company in available_companies:
    if st.sidebar.checkbox(f"{company}", value=True, key=f"company_{company}"):
        selected_companies.append(company)

# Filter data
filtered_df = filter_data_by_year(df, selected_years)
filtered_df = filter_data_by_companies(filtered_df, selected_companies)

# Navigation
page = st.sidebar.selectbox("Select Page", ["PBMZI Stocks Overview", "Interaction of PBMZI Stocks"])

if page == "PBMZI Stocks Overview":
    st.title("📈 PBMZI (2018-2023)")
    
    if filtered_df.empty or len(selected_companies) == 0:
        st.warning("Please select at least one year and one company to display data.")
    else:
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Stock Prices Trend
            st.subheader("Stock Prices Trend")
            fig_prices = go.Figure()
            
            for company in selected_companies:
                fig_prices.add_trace(go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df[company],
                    mode='lines',
                    name=company,
                    line=dict(width=2)
                ))
            
            fig_prices.update_layout(
                title="Price Movement of PBMZI",
                xaxis_title="Date",
                yaxis_title="Price (~M$)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_prices, use_container_width=True)
            
            # 3. 30-Day Rolling Volatility
            st.subheader("30-Day Rolling Volatility")
            volatility_30 = calculate_rolling_volatility(filtered_df, 30)
            
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
                title="30-Day Rolling Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_vol30, use_container_width=True)
            
            # 5. Correlation Matrix
            st.subheader("Correlation Matrix")
            log_returns = calculate_log_returns(filtered_df)
            corr_matrix = log_returns[selected_companies].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                title="Correlation Matrix of Log Returns"
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            # 2. Returns Trend
            st.subheader("Logarithmic Returns Trend")
            log_returns = calculate_log_returns(filtered_df)
            
            fig_returns = go.Figure()
            for company in selected_companies:
                fig_returns.add_trace(go.Scatter(
                    x=filtered_df['Date'],
                    y=log_returns[company],
                    mode='lines',
                    name=company,
                    line=dict(width=2)
                ))
            
            fig_returns.update_layout(
                title="Logarithmic Return of PBMZI",
                xaxis_title="Date",
                yaxis_title="Logarithmic Return",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_returns, use_container_width=True)
            
            # 4. 60-Day Rolling Volatility
            st.subheader("60-Day Rolling Volatility")
            volatility_60 = calculate_rolling_volatility(filtered_df, 60)
            
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
                title="60-Day Rolling Volatility",
                xaxis_title="Date",
                yaxis_title="Volatility",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_vol60, use_container_width=True)
            
            # 6. Maximum Drawdown vs Maximum Return
            st.subheader("Max Return vs Max Drawdown")
            log_returns = calculate_log_returns(filtered_df)
            
            drawdown_data = []
            for company in selected_companies:
                max_dd = calculate_max_drawdown(filtered_df[company])
                max_ret = calculate_max_return(log_returns[company])
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
                title="Max Return vs Max Drawdown per Company",
                hover_data=['Company']
            )
            fig_scatter.update_layout(height=400)
            fig_scatter.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_scatter, use_container_width=True)

elif page == "Interaction of PBMZI Stocks":
    st.title("🔗 Interaction of PBMZI Stocks")
    
    if filtered_df.empty or len(selected_companies) < 2:
        st.warning("Please select at least two companies and one year to display network analysis.")
    else:
        # Create MST for overall period (2018-2023)
        st.subheader("Minimum Spanning Tree Network (2018-2023)")
        
        # Use full dataset for overall MST
        full_company_df = filter_data_by_companies(df, selected_companies)
        mst_full, dist_matrix_full = create_mst_network(full_company_df)
        fig_mst_full = create_network_plot(mst_full, dist_matrix_full)
        st.plotly_chart(fig_mst_full, use_container_width=True)
        
        # Show MST for selected years if different from full period
        if len(selected_years) < len(available_years) and len(selected_years) > 0:
            year_str = ", ".join(map(str, sorted(selected_years)))
            st.subheader(f"MST Network for Selected Years ({year_str})")
            
            mst_filtered, dist_matrix_filtered = create_mst_network(filtered_df)
            fig_mst_filtered = create_network_plot(mst_filtered, dist_matrix_filtered)
            st.plotly_chart(fig_mst_filtered, use_container_width=True)
        
        # Display network statistics
        st.subheader("Network Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Number of Nodes", len(mst_full.nodes()))
        
        with col2:
            st.metric("Number of Edges", len(mst_full.edges()))
        
        with col3:
            avg_degree = sum(dict(mst_full.degree()).values()) / len(mst_full.nodes())
            st.metric("Average Degree", f"{avg_degree:.2f}")
        
        # Show edge weights table
        st.subheader("Edge Weights (Distances)")
        edge_data = []
        for edge in mst_full.edges(data=True):
            edge_data.append({
                'Company 1': edge[0],
                'Company 2': edge[1],
                'Distance': round(edge[2]['weight'], 4)
            })
        
        edge_df = pd.DataFrame(edge_data)
        edge_df = edge_df.sort_values('Distance')
        st.dataframe(edge_df, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Dashboard Features:**")
st.sidebar.markdown("• Interactive filtering by year and company")
st.sidebar.markdown("• Real-time chart updates")
st.sidebar.markdown("• Professional visualizations")
st.sidebar.markdown("• Network analysis with MST")
