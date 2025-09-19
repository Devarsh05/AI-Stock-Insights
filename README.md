# 📈 AI Stock Insights

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/Devarsh05/AI-Stock-Insights/main/src/app.py)

**AI Stock Insights** is an interactive web application for exploring stock market data, analyzing technical indicators, and making short-term predictions using machine learning models. Built with **Python**, **Streamlit**, and **scikit-learn**, this app is designed for investors, analysts, and enthusiasts who want AI-driven insights on stock prices.

---

## 🌟 Features

1. **Stock Data Fetching**
   - Fetch historical OHLCV (Open, High, Low, Close, Volume) data from Yahoo Finance.
   - Supports multiple tickers at once (`AAPL`, `MSFT`, `GOOG`, etc.).
   - Adjustable data periods: 6 months, 1 year, 2 years, 5 years.

2. **Technical Indicators**
   - Simple Moving Averages: `MA10`, `MA20`, `MA50`
   - Moving Average Difference: `MA_diff`
   - Returns: `Return1`, `Return3`
   - Momentum: `Momentum5`
   - Volatility: `Volatility20`
   - Bollinger Bands: `BB_Upper`, `BB_Lower`
   - RSI: `RSI14`
   - ATR: `ATR14`
   - Volume Change: `Volume_Change`

3. **Interactive Charts**
   - Price charts with moving averages & Bollinger Bands.
   - RSI charts for trend analysis.
   - Supports multiple tickers on the same chart.

4. **Machine Learning Predictions**
   - Predict next-day or multi-day ahead stock closing prices.
   - Models: Random Forest, Gradient Boosting, Linear Regression.
   - Metrics: RMSE, MAE, R², Directional Accuracy.
   - Feature importance visualization.
   - Downloadable CSV of predictions.

5. **Uncertainty Estimation**
   - For Random Forest: approximate standard deviation of predicted prices.

---

## 🚀 Live Demo

[Open AI Stock Insights in Streamlit](https://share.streamlit.io/Devarsh05/AI-Stock-Insights/main/src/app.py)

---

## 🛠 Installation & Local Setup

1. **Clone the repository:**

```bash
git clone https://github.com/Devarsh05/AI-Stock-Insights.git
cd AI-Stock-Insights/src
