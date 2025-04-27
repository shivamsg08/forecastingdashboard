# Install these if missing
# pip install streamlit plotly pandas ta

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import ta  # For technical analysis metrics

# --------- Load your data (Sample for now) ----------
np.random.seed(42)
weeks = pd.date_range('2023-01-01', periods=52, freq='W')
data = pd.DataFrame({
    'Store': np.random.choice(['Store A', 'Store B', 'Store C'], 52),
    'Item': np.random.choice(['Item X', 'Item Y', 'Item Z'], 52),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 52),
    'Week': weeks,
    'Forecast_Model1': np.random.randint(80, 120, 52),
    'Forecast_Model2': np.random.randint(85, 115, 52),
    'Forecast_Model3': np.random.randint(78, 125, 52),
    'Forecast_Model4': np.random.randint(82, 118, 52),
    'Forecast_Model5': np.random.randint(80, 130, 52),
    'Actual': np.random.randint(85, 125, 52),
    'Promo_Week': np.random.choice([0, 1], 52, p=[0.8, 0.2]),
    'Event_Week': np.random.choice([0, 1], 52, p=[0.9, 0.1])
})

# --------- Sidebar Filters ----------
st.sidebar.title("Filters")
selected_store = st.sidebar.selectbox("Select Store", sorted(data['Store'].unique()))
selected_item = st.sidebar.selectbox("Select Item", sorted(data['Item'].unique()))
selected_region = st.sidebar.selectbox("Select Region", sorted(data['Region'].unique()))
selected_models = st.sidebar.multiselect("Select Forecast Models", 
                                         ['Forecast_Model1', 'Forecast_Model2', 'Forecast_Model3', 'Forecast_Model4', 'Forecast_Model5'],
                                         default=['Forecast_Model1'])

# Filtered data
filtered_data = data[(data['Store'] == selected_store) & 
                     (data['Item'] == selected_item) & 
                     (data['Region'] == selected_region)].sort_values('Week')

# --------- Editable Table ----------
st.title("📈 Demand Forecast Dashboard")
st.subheader("Editable Forecast Table")
edited_data = st.data_editor(
    filtered_data[selected_models + ['Actual', 'Promo_Week', 'Event_Week', 'Week']],
    num_rows="dynamic",
    use_container_width=True
)

# --------- Line Chart ----------
st.subheader("Forecast vs Actual Trend")

fig = go.Figure()

# Plot all selected forecasts
for model in selected_models:
    fig.add_trace(go.Scatter(
        x=edited_data['Week'], 
        y=edited_data[model], 
        mode='lines+markers',
        name=model
    ))

# Actuals
fig.add_trace(go.Scatter(
    x=edited_data['Week'],
    y=edited_data['Actual'],
    mode='lines+markers',
    name='Actual',
    line=dict(color='black', dash='dash')
))

# Promo Weeks as bars
promo_weeks = edited_data[edited_data['Promo_Week'] == 1]
for pw in promo_weeks['Week']:
    fig.add_vrect(x0=pw, x1=pw + pd.Timedelta(days=7),
                  fillcolor="LightSkyBlue", opacity=0.3, line_width=0)

# Event Weeks as bars
event_weeks = edited_data[edited_data['Event_Week'] == 1]
for ew in event_weeks['Week']:
    fig.add_vrect(x0=ew, x1=ew + pd.Timedelta(days=7),
                  fillcolor="LightGreen", opacity=0.3, line_width=0)

fig.update_layout(
    title="Demand Trend with Promo & Event Weeks",
    xaxis_title="Week",
    yaxis_title="Demand",
    template="plotly_white",
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# --------- Historical Promotions ----------
st.subheader("📋 Historical Promotions Table")
promo_hist = edited_data[edited_data['Promo_Week'] == 1]
st.dataframe(promo_hist[['Week', 'Actual']])

# --------- Tabs for Advanced Metrics ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Wmape Comparison", 
    "Demand Metrics",
    "RSI",
    "MACD",
    "Bollinger Bands",
    "Sharpe Ratio"
])

# Tab 1 - WMAPE Comparison
with tab1:
    st.subheader("WMAPE (Weighted MAPE) Comparison")
    wmape_scores = {}
    for model in selected_models:
        forecast = edited_data[model]
        actual = edited_data['Actual']
        wmape = np.sum(np.abs(forecast - actual)) / np.sum(actual)
        wmape_scores[model] = wmape

    wmape_df = pd.DataFrame({
        'Model': list(wmape_scores.keys()),
        'WMAPE': list(wmape_scores.values())
    }).sort_values('WMAPE')

    bar_chart = go.Figure(go.Bar(
        x=wmape_df['Model'],
        y=wmape_df['WMAPE'],
        marker_color='indianred'
    ))
    bar_chart.update_layout(template="plotly_white", height=500)
    st.plotly_chart(bar_chart, use_container_width=True)

# Tab 2 - Demand Metrics
with tab2:
    st.subheader("📊 Demand Metrics")
    st.metric("Standard Deviation of Weekly Demand", round(edited_data['Actual'].std(),2))
    st.metric("Average Weekly Demand", round(edited_data['Actual'].mean(),2))
    st.metric("Demand Range (Max - Min)", round(edited_data['Actual'].max() - edited_data['Actual'].min(),2))

# Tab 3 - RSI
with tab3:
    st.subheader("📈 Relative Strength Index (RSI)")
    rsi = ta.momentum.RSIIndicator(edited_data['Actual'], window=14).rsi()
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=edited_data['Week'], y=rsi, mode='lines', name='RSI'))
    fig_rsi.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_rsi, use_container_width=True)

# Tab 4 - MACD
with tab4:
    st.subheader("📈 Moving Average Convergence Divergence (MACD)")
    macd = ta.trend.MACD(edited_data['Actual'])
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=edited_data['Week'], y=macd.macd(), mode='lines', name='MACD Line'))
    fig_macd.add_trace(go.Scatter(x=edited_data['Week'], y=macd.macd_signal(), mode='lines', name='Signal Line'))
    fig_macd.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_macd, use_container_width=True)

# Tab 5 - Bollinger Bands
with tab5:
    st.subheader("📈 Bollinger Bands")
    bollinger = ta.volatility.BollingerBands(edited_data['Actual'])
    fig_boll = go.Figure()
    fig_boll.add_trace(go.Scatter(x=edited_data['Week'], y=bollinger.bollinger_hband(), mode='lines', name='Upper Band'))
    fig_boll.add_trace(go.Scatter(x=edited_data['Week'], y=bollinger.bollinger_lband(), mode='lines', name='Lower Band'))
    fig_boll.add_trace(go.Scatter(x=edited_data['Week'], y=edited_data['Actual'], mode='lines', name='Actual'))
    fig_boll.update_layout(template="plotly_white", height=500)
    st.plotly_chart(fig_boll, use_container_width=True)

# Tab 6 - Sharpe Ratio
with tab6:
    st.subheader("📈 Sharpe Ratio (Demand Stability)")
    returns = edited_data['Actual'].pct_change().dropna()
    sharpe_ratio = returns.mean() / returns.std()
    st.metric("Sharpe Ratio", round(sharpe_ratio, 2))

