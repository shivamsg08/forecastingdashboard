import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import datetime
import random
import ta

# -----------------------------
# 1. Simulate data
# -----------------------------

@st.cache_data
def simulate_data():
    np.random.seed(42)
    random.seed(42)

    start_date = datetime.date(2021, 4, 1)
    weeks = pd.date_range(start=start_date, periods=156, freq='W')

    stores = [f"Store_{i}" for i in range(1, 21)]
    items = [f"Item_{i}" for i in range(1, 16)]
    regions = ['North', 'South', 'East', 'West']

    data = []
    for store in stores:
        for item in items:
            region = random.choice(regions)
            sales = np.random.poisson(lam=random.randint(80, 200), size=len(weeks))

            promo_weeks = random.sample(range(len(weeks)), k=20)
            event_weeks = random.sample(range(len(weeks)), k=15)

            for idx, week in enumerate(weeks):
                data.append({
                    "Store": store,
                    "Item": item,
                    "Region": region,
                    "Week": week,
                    "Actuals": sales[idx],
                    "Promo": 1 if idx in promo_weeks else 0,
                    "Event": 1 if idx in event_weeks else 0
                })

    df_actuals = pd.DataFrame(data)

    # Forecasts for next 6 weeks
    forecast_weeks = pd.date_range(start=weeks[-1] + datetime.timedelta(days=7), periods=6, freq='W')
    forecast_data = []
    for store in stores:
        for item in items:
            for week in forecast_weeks:
                for model in range(1, 6):
                    forecast_data.append({
                        "Store": store,
                        "Item": item,
                        "Region": random.choice(regions),
                        "Week": week,
                        f"Model_{model}_Forecast": np.random.randint(90, 200)
                    })

    df_forecasts = pd.DataFrame(forecast_data)
    df_forecasts = df_forecasts.groupby(["Store", "Item", "Region", "Week"]).sum().reset_index()

    return df_actuals, df_forecasts

df_actuals, df_forecasts = simulate_data()

# Save original forecasts for reset
original_forecasts = df_forecasts.copy()

# -----------------------------
# 2. UI - Sidebar Filters
# -----------------------------

st.set_page_config(page_title="Demand Forecasting Dashboard", layout="wide")

st.sidebar.title("🔎 Filters")
store_selected = st.sidebar.selectbox("Select Store", df_actuals['Store'].unique())
item_selected = st.sidebar.selectbox("Select Item", df_actuals['Item'].unique())
region_selected = st.sidebar.selectbox("Select Region", df_actuals['Region'].unique())
moving_avg_weeks = st.sidebar.multiselect("Select Moving Averages", [3,5,10])

# Reset Button
if st.sidebar.button("🔄 Reset Forecasts"):
    df_forecasts = original_forecasts.copy()
    st.success("Forecasts have been reset!")
    st.session_state.clear()  # Clear session state to reset the app

# Filter data
df_hist = df_actuals[(df_actuals['Store'] == store_selected) &
                     (df_actuals['Item'] == item_selected) &
                     (df_actuals['Region'] == region_selected)]

df_future = df_forecasts[(df_forecasts['Store'] == store_selected) &
                         (df_forecasts['Item'] == item_selected) &
                         (df_forecasts['Region'] == region_selected)]

# -----------------------------
# 3. Tabs Layout
# -----------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Forecast Adjustment",
    "📊 WMAPE Comparison",
    "📉 Metrics - RSI & MACD",
    "🎯 Metrics - Std Dev, Bollinger",
    "🚀 Metrics - Sharpe Ratio",
    "🧠 Metrics - Volume Impact",
    "📝 Promo History"
])

# -----------------------------
# 4. Tab 1 - Forecast Adjustment
# -----------------------------

with tab1:
    st.title("Demand Forecast Analysis")

    # ➡️ Use df_future to fill the forecast table dynamically
    forecast_columns = ['Week'] + [col for col in df_future.columns if 'Model_' in col]
    forecast_df = df_future[forecast_columns].copy()

    st.subheader("Editable Forecast Table for Next 6 Weeks")
    edited_forecast_df = st.data_editor(
        forecast_df,
        num_rows="dynamic",
        use_container_width=True
    )

    # ➡️ Plot the LINE CHART using edited_forecast_df
    st.subheader("Forecast Line Chart")

    import altair as alt
    chart = alt.Chart(edited_forecast_df.melt('Week', var_name='Model', value_name='Forecast')).mark_line(point=True).encode(
        x='Week:T',
        y='Forecast:Q',
        color='Model:N'
    ).properties(
        width=800,
        height=400
    )

    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# 5. Tab 2 - WMAPE Comparison
# -----------------------------

with tab2:
    st.title("📊 WMAPE Across Models")

    wmape_scores = {}
    for model in range(1, 6):
        pred = df_future[f"Model_{model}_Forecast"].values
        actual = df_hist['Actuals'].values[-len(pred):] if len(df_hist) >= len(pred) else df_hist['Actuals'].values
        wmape = np.sum(np.abs(actual - pred)) / np.sum(actual) if np.sum(actual) != 0 else 0
        wmape_scores[f'Model {model}'] = wmape

    fig_wmape = go.Figure([go.Bar(x=list(wmape_scores.keys()), y=list(wmape_scores.values()))])
    fig_wmape.update_layout(title="WMAPE by Model", template="plotly_white")

    st.plotly_chart(fig_wmape, use_container_width=True)

# -----------------------------
# 6. Tab 3-7: Advanced Metrics
# -----------------------------

with tab3:
    st.title("📉 RSI and MACD")
    df_metrics = df_hist.copy()
    df_metrics['RSI'] = ta.momentum.RSIIndicator(df_metrics['Actuals']).rsi()
    macd = ta.trend.MACD(df_metrics['Actuals'])
    df_metrics['MACD'] = macd.macd()
    df_metrics['MACD_Signal'] = macd.macd_signal()

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_metrics['Week'], y=df_metrics['RSI'], mode='lines', name='RSI'))
    fig_rsi.update_layout(title="RSI", template="plotly_white")
    st.plotly_chart(fig_rsi, use_container_width=True)

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df_metrics['Week'], y=df_metrics['MACD'], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df_metrics['Week'], y=df_metrics['MACD_Signal'], name="Signal"))
    fig_macd.update_layout(title="MACD", template="plotly_white")
    st.plotly_chart(fig_macd, use_container_width=True)

with tab4:
    st.title("🎯 Std Dev and Bollinger Bands")
    df_metrics['StdDev'] = df_metrics['Actuals'].rolling(window=10).std()
    boll = ta.volatility.BollingerBands(df_metrics['Actuals'])
    df_metrics['Boll_Upper'] = boll.bollinger_hband()
    df_metrics['Boll_Lower'] = boll.bollinger_lband()

    fig_std = go.Figure()
    fig_std.add_trace(go.Scatter(x=df_metrics['Week'], y=df_metrics['StdDev'], mode='lines', name='Std Dev'))
    fig_std.update_layout(title="Standard Deviation", template="plotly_white")
    st.plotly_chart(fig_std, use_container_width=True)

    fig_boll = go.Figure()
    fig_boll.add_trace(go.Scatter(x=df_metrics['Week'], y=df_metrics['Actuals'], name="Actuals"))
    fig_boll.add_trace(go.Scatter(x=df_metrics['Week'], y=df_metrics['Boll_Upper'], name="Upper Band"))
    fig_boll.add_trace(go.Scatter(x=df_metrics['Week'], y=df_metrics['Boll_Lower'], name="Lower Band"))
    fig_boll.update_layout(title="Bollinger Bands", template="plotly_white")
    st.plotly_chart(fig_boll, use_container_width=True)

with tab5:
    st.title("🚀 Sharpe Ratio Analysis")
    returns = df_hist['Actuals'].pct_change()
    sharpe_ratio = returns.mean() / returns.std() if returns.std() != 0 else 0
    sharpe_ratio_annualized = sharpe_ratio * np.sqrt(52)  # Annualized Sharpe Ratio for weekly data
    st.metric("Annualized Sharpe Ratio", f"{sharpe_ratio_annualized:.2f}")

with tab6:
    st.title("🧠 Volume Impact of Promotions")
    promo_impact = df_hist[df_hist['Promo'] == 1]['Actuals'].mean() - df_hist[df_hist['Promo'] == 0]['Actuals'].mean()
    st.metric("Promo Volume Impact", f"{promo_impact:.1f} Units")

with tab7:
    st.title("📝 Historical Promo/Event Weeks")
    promo_table = df_hist[(df_hist['Promo'] == 1)]
