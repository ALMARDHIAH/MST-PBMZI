# =========================
# PAGE 1 – PBMZI EDA
# =========================
if page == "PBMZI (2018-2023)":
    st.title("PBMZI (2018-2023)")

    filtered_data = cleaned_PBMZI[cleaned_PBMZI['Date'].dt.year.isin(selected_years)]

    # 1. Price Movement
    fig1 = px.line(filtered_data, x="Date", y=selected_companies,
                   labels={"value": "Price (~M$)", "Date": "Date"},
                   title=f"Price Movement of PBMZI ({min(selected_years)}–{max(selected_years)})")

    # 2. Logarithmic Return Movement
    log_return = filtered_data[selected_companies].apply(lambda col: np.log(col / col.shift(1)))
    df_long = log_return[selected_companies].copy()
    df_long["Date"] = filtered_data["Date"]
    df_long = df_long.melt(id_vars="Date", var_name="Company", value_name="Log Return")
    fig2 = px.line(df_long, x="Date", y="Log Return", color="Company",
                   title=f"Logarithmic Return Movement of PLCs ({min(selected_years)}–{max(selected_years)})")

    # 3.1 Volatility 30-day rolling
    volatility_return_30 = log_return.rolling(window=30).std()
    df_vol_long = volatility_return_30[selected_companies].copy()
    df_vol_long["Date"] = filtered_data["Date"]
    df_vol_long = df_vol_long.melt(id_vars="Date", var_name="Company", value_name="30-Day Rolling Volatility")
    fig3_1 = px.line(df_vol_long, x="Date", y="30-Day Rolling Volatility", color="Company",
                     title="30-Day Rolling Volatility")

    # 3.2 Volatility 60-day rolling
    volatility_return_60 = log_return.rolling(window=60).std()
    df_vol_long = volatility_return_60[selected_companies].copy()
    df_vol_long["Date"] = filtered_data["Date"]
    df_vol_long = df_vol_long.melt(id_vars="Date", var_name="Company", value_name="60-Day Rolling Volatility")
    fig3_2 = px.line(df_vol_long, x="Date", y="60-Day Rolling Volatility", color="Company",
                     title="60-Day Rolling Volatility")

    # 4. Correlation Matrix
    corr_matrix = log_return.corr(method='pearson')
    fig4 = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu",
                     zmin=-1, zmax=1, labels=dict(color="Correlation"),
                     title="Correlation Matrix of PBMZI Companies' Log Return")

    # =========================
    # LAYOUT – fit all charts in one page
    # =========================
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig3_1, use_container_width=True)
    with col4:
        st.plotly_chart(fig3_2, use_container_width=True)

    # Full-width Correlation Matrix
    st.plotly_chart(fig4, use_container_width=True)
