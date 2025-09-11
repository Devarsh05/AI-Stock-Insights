import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 AI Stock Insights - Prototype")

# Input
ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, TSLA, MSFT):", "AAPL")

if ticker:
    # Fetch data
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    st.subheader(f"{ticker} - Last 6 Months")
    st.line_chart(hist["Close"])

    st.subheader("Raw Data")
    st.write(hist.tail())
