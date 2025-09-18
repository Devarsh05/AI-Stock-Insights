# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_fetch import get_stock_data
from analysis import add_technical_indicators
from ml_model import prepare_dataset, train_and_evaluate, predict_with_uncertainty, DEFAULT_FEATURES

st.set_page_config(page_title="AI Stock Insights (Phase 2)", layout="wide")
st.title("📈 AI Stock Insights — Phase 2")

# --- Sidebar controls
with st.sidebar:
    st.header("Controls")
    ticker = st.text_input("Ticker", value="AAPL")
    period = st.selectbox("Data period", options=["6mo","1y","2y","5y"], index=3)
    horizon = st.selectbox("Prediction horizon (days ahead)", options=[1,3,5], index=0)
    model_choice = st.selectbox("Model", options=["Random Forest","Gradient Boosting","Linear Regression"], index=0)
    test_pct = st.slider("Test set %", min_value=10, max_value=50, value=20, step=5)
    # features: default to DEFAULT_FEATURES (existing columns will be filtered inside)
    features_to_use = st.multiselect("Choose features (defaults shown)", options=DEFAULT_FEATURES, default=DEFAULT_FEATURES)
    run_button = st.button("Fetch, Train & Predict")

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # flatten MultiIndex columns if present
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in col if x and str(x)!='']).strip('_') for col in df.columns]
    # attempt to ensure 'Close' column exists
    if 'Close' not in df.columns:
        close_candidates = [c for c in df.columns if 'close' in str(c).lower()]
        if close_candidates:
            df['Close'] = df[close_candidates[0]]
    return df

if run_button:
    try:
        raw = get_stock_data(ticker, period=period)
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        st.stop()

    if raw.empty:
        st.error("No data returned. Try a longer period or different ticker.")
        st.stop()

    df = normalize_dataframe(raw)
    # normalize & debug rows
df = normalize_dataframe(raw)

# Debugging: show how many rows were fetched and the date range
st.write(f"Rows fetched (raw): {len(raw)}")
try:
    st.write(f"Date range: {raw.index.min().date()} to {raw.index.max().date()}")
except Exception:
    pass

# Add technical indicators early so you can see how many rows remain AFTER indicators
from analysis import add_technical_indicators
df_with_ind = add_technical_indicators(df)
st.write(f"Rows after adding indicators (rows with indicator values may be fewer due to rolling windows): {len(df_with_ind)}")

    st.write("Columns detected:", df.columns.tolist())

    # Add indicators
    df = add_technical_indicators(df)
    st.subheader(f"{ticker} — last rows")
    st.dataframe(df.tail()[['Close','MA10','MA20','MA50','RSI14']].dropna(how='all'))

    # ---- Price chart with MA & Bollinger (plotly)
    st.subheader("Price chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(color='black')))
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', line=dict(color='blue')))
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(color='orange')))
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        # plot upper & lower as semi-transparent band
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper', line=dict(color='rgba(0,0,0,0.0)'), showlegend=False))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower', fill='tonexty', fillcolor='rgba(173,216,230,0.2)', line=dict(color='rgba(0,0,0,0.0)'), showlegend=False))
    fig.update_layout(height=400, margin=dict(l=20,r=20,t=30,b=20))
    st.plotly_chart(fig, use_container_width=True)

    # RSI chart
    st.subheader("RSI (14)")
    if 'RSI14' in df.columns:
        rfig = go.Figure()
        rfig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], name='RSI14'))
        rfig.update_layout(height=200, margin=dict(l=20,r=20,t=10,b=10), yaxis=dict(range=[0,100]))
        st.plotly_chart(rfig, use_container_width=True)
    else:
        st.info("RSI not available (needs >= 14 observations).")

    # ---- Prepare dataset for ML
    try:
        X, y = prepare_dataset(df, horizon=horizon, features=features_to_use)
    except Exception as e:
        st.error(f"Failed to prepare dataset: {e}")
        st.stop()

    if len(X) < 30:
        st.warning("Not enough rows to train with these features/horizon. Try a longer period or fewer features.")
    else:
        model_map = {'Random Forest':'rf', 'Gradient Boosting':'gbr', 'Linear Regression':'lr'}
        model_type = model_map.get(model_choice, 'rf')

        with st.spinner("Training model..."):
            results = train_and_evaluate(X, y, model_type=model_type, test_size=test_pct/100.0)
        model = results['model']

        # metrics
        st.subheader("Evaluation (Test set)")
        st.write(f"- RMSE: **{results['rmse']:.4f}**")
        st.write(f"- MAE: **{results['mae']:.4f}**")
        st.write(f"- R²: **{results['r2']:.4f}**")
        if not np.isnan(results['directional_accuracy']):
            st.write(f"- Directional accuracy: **{results['directional_accuracy']*100:.2f}%**")

        # plot actual vs predicted
        res_df = pd.DataFrame({
            'Actual': results['y_test'].values,
            'Predicted': results['preds']
        }, index=results['y_test'].index)
        st.subheader("Actual vs Predicted (test)")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=res_df.index, y=res_df['Actual'], name='Actual'))
        fig2.add_trace(go.Scatter(x=res_df.index, y=res_df['Predicted'], name='Predicted'))
        fig2.update_layout(height=350, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig2, use_container_width=True)

        # Feature importance (if available)
        st.subheader("Feature importance")
        fi = results.get('feature_importances', {})
        if fi:
            fi_items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            fi_df = pd.DataFrame(fi_items, columns=['feature','importance'])
            st.bar_chart(fi_df.set_index('feature'))
        else:
            st.info("No feature importance available for this model.")

        # Next-day (or horizon) prediction + uncertainty (if supported)
        last_row = X.iloc[-1].values.reshape(1, -1)
        pred_mean, pred_std = predict_with_uncertainty(results['model'], last_row)
        st.subheader(f"Prediction (horizon = {horizon} day(s))")
        if np.isfinite(pred_mean):
            st.metric(f"Predicted close in {horizon} day(s)", f"${pred_mean:.2f}")
            if np.isfinite(pred_std):
                st.write(f"Estimated std (approx.): {pred_std:.4f} → 68% interval: [{pred_mean - pred_std:.2f}, {pred_mean + pred_std:.2f}]")
            else:
                st.info("Model does not provide uncertainty estimate (std unavailable).")

        # allow downloading test predictions
        @st.cache_data
        def build_download_csv(df_in):
            return df_in.to_csv(index=True).encode('utf-8')

        download_csv = build_download_csv(res_df)
        st.download_button("Download test predictions CSV", download_csv, file_name=f"{ticker}_predictions.csv", mime="text/csv")

    st.success("Done.")
