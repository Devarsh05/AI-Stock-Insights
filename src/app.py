# src/app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from data_fetch import get_stock_data
from analysis import add_technical_indicators
from ml_model import prepare_dataset, train_and_evaluate, predict_with_uncertainty, DEFAULT_FEATURES

# --- Page setup
st.set_page_config(page_title="AI Stock Insights", layout="wide")
st.title("📊 AI Stock Insights — Machine Learning Powered Predictions")
st.caption("Built with Streamlit • scikit-learn • yfinance")

# --- Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")

    with st.expander("Data Settings", expanded=True):
        tickers = st.multiselect(
            "Tickers (select 1 or more)", 
            options=["AAPL","MSFT","GOOG","AMZN","TSLA","NVDA","FB"],
            default=["AAPL"]
        )
        period = st.selectbox("Data period", options=["6mo","1y","2y","5y"], index=3)

    with st.expander("Model Settings", expanded=True):
        horizon = st.selectbox("Prediction horizon (days ahead)", options=[1,3,5], index=0)
        model_choice = st.selectbox("Model", options=["Random Forest","Gradient Boosting","Linear Regression"], index=0)
        test_pct = st.slider("Test set %", min_value=10, max_value=50, value=20, step=5)
        features_to_use = st.multiselect("Features to use", options=DEFAULT_FEATURES, default=DEFAULT_FEATURES)

    with st.expander("Indicators to Plot", expanded=False):
        indicators_to_plot = st.multiselect(
            "Select indicators", 
            options=["MA20","MA50","BB_Upper","BB_Lower","RSI14"], 
            default=["MA20","MA50","BB_Upper","BB_Lower"]
        )

    run_button = st.button("🚀 Run Analysis")

# --- Helper functions
def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(x) for x in col if x]).strip('_') for col in df.columns]
    if 'Close' not in df.columns:
        close_candidates = [c for c in df.columns if 'close' in str(c).lower()]
        if close_candidates:
            df['Close'] = df[close_candidates[0]]
    return df

# --- Main run
if run_button:
    all_data = {}
    with st.spinner("Fetching & processing data..."):
        for ticker in tickers:
            try:
                raw = get_stock_data(ticker, period=period)
                if raw.empty:
                    st.warning(f"No data for {ticker}")
                    continue
                df = normalize_dataframe(raw)
                df = add_technical_indicators(df)
                all_data[ticker] = df
            except Exception as e:
                st.error(f"Error fetching {ticker}: {e}")

    if not all_data:
        st.error("No valid data fetched. Try a longer period or different tickers.")
        st.stop()

    # --- Price chart
    st.subheader("📈 Stock Price & Indicators")
    fig = go.Figure()
    colors = ['black', 'blue', 'red', 'green', 'orange', 'purple']
    for i, (ticker, df) in enumerate(all_data.items()):
        fig.add_trace(go.Scatter(
            x=df.index, y=df['Close'], 
            name=f"{ticker} Close", line=dict(color=colors[i % len(colors)], width=2)
        ))
        if 'MA20' in indicators_to_plot and 'MA20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MA20'], 
                name=f"{ticker} MA20", line=dict(color=colors[i % len(colors)], dash='dot')
            ))
        if 'MA50' in indicators_to_plot and 'MA50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MA50'], 
                name=f"{ticker} MA50", line=dict(color=colors[i % len(colors)], dash='dash')
            ))
        if 'BB_Upper' in indicators_to_plot and 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Upper'], line=dict(color='rgba(0,0,0,0)'), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Lower'], fill='tonexty',
                fillcolor='rgba(173,216,230,0.2)', line=dict(color='rgba(0,0,0,0)'), showlegend=False
            ))
    fig.update_layout(
        template="plotly_white", 
        height=400, 
        margin=dict(l=20,r=20,t=30,b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- RSI chart
    st.subheader("📊 Relative Strength Index (RSI)")
    for ticker, df in all_data.items():
        if 'RSI14' in df.columns:
            rfig = go.Figure()
            rfig.add_trace(go.Scatter(x=df.index, y=df['RSI14'], name=f"{ticker} RSI14"))
            rfig.update_layout(
                template="plotly_white", height=200, 
                margin=dict(l=20,r=20,t=10,b=10), yaxis=dict(range=[0,100])
            )
            st.plotly_chart(rfig, use_container_width=True)

    # --- ML Backtesting & Predictions
    st.subheader("🧪 Model Training & Backtesting")
    model_map = {'Random Forest':'rf', 'Gradient Boosting':'gbr', 'Linear Regression':'lr'}
    model_type = model_map.get(model_choice, 'rf')
    all_results = {}

    for ticker, df in all_data.items():
        st.write(f"### {ticker}")
        try:
            X, y = prepare_dataset(df, horizon=horizon, features=features_to_use)
            results = train_and_evaluate(X, y, model_type=model_type, test_size=test_pct/100.0)
            all_results[ticker] = results

            # Metrics
            st.markdown(f"""
            - **RMSE:** {results['rmse']:.4f}  
            - **MAE:** {results['mae']:.4f}  
            - **R²:** {results['r2']:.4f}  
            - **Directional accuracy:** {results['directional_accuracy']*100:.2f}%  
            """)

            # Actual vs Predicted
            res_df = pd.DataFrame({
                'Actual': results['y_test'].values,
                'Predicted': results['preds']
            }, index=results['y_test'].index)
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=res_df.index, y=res_df['Actual'], name=f"{ticker} Actual"))
            fig2.add_trace(go.Scatter(x=res_df.index, y=res_df['Predicted'], name=f"{ticker} Predicted"))
            fig2.update_layout(template="plotly_white", height=350, margin=dict(l=20,r=20,t=20,b=20))
            st.plotly_chart(fig2, use_container_width=True)

            # Feature importance
            st.subheader("🔍 Feature Importance")
            fi = results.get('feature_importances', {})
            if fi:
                fi_items = sorted(fi.items(), key=lambda x: x[1], reverse=True)
                fi_df = pd.DataFrame(fi_items, columns=['feature','importance'])
                st.bar_chart(fi_df.set_index('feature'))
            else:
                st.info("No feature importance available for this model.")

            # Next-day prediction + uncertainty
            st.subheader("🎯 Future Price Prediction")
            last_row = X.iloc[-1].values.reshape(1, -1)
            pred_mean, pred_std = predict_with_uncertainty(results['model'], last_row)
            st.metric(f"Predicted close in {horizon} day(s)", f"${pred_mean:.2f}")
            if np.isfinite(pred_std):
                st.write(f"68% confidence interval: [{pred_mean - pred_std:.2f}, {pred_mean + pred_std:.2f}]")

            # Download CSV
            @st.cache_data
            def build_download_csv(df_in):
                return df_in.to_csv(index=True).encode('utf-8')
            download_csv = build_download_csv(res_df)
            st.download_button(
                f"📥 Download {ticker} predictions CSV", 
                download_csv, file_name=f"{ticker}_predictions.csv", mime="text/csv"
            )

        except Exception as e:
            st.warning(f"Could not process {ticker}: {e}")

    # --- Footer
    st.markdown("---")
    st.caption("© 2025 AI Stock Insights • Built with ❤️ using Streamlit")
