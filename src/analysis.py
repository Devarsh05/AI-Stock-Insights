# src/analysis.py
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators (consistent column names):
    - MA10, MA20, MA50
    - MA_diff (MA20 - MA50)
    - Return1, Return3
    - Momentum5
    - Volatility20 (rolling std of Close)
    - BB_Upper, BB_Lower (Bollinger bands from MA20 & Volatility20)
    - RSI14 (classic)
    - ATR14 (average true range)
    - Volume_Change
    """
    df = df.copy()
    if 'Close' not in df.columns:
        candidates = [c for c in df.columns if 'close' in str(c).lower()]
        if candidates:
            df['Close'] = df[candidates[0]]
        else:
            raise KeyError("No 'Close' column found in data")

    # Simple moving averages
    df['MA10'] = df['Close'].rolling(window=10, min_periods=1).mean()
    df['MA20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['MA50'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['MA_diff'] = df['MA20'] - df['MA50']

    # Returns & momentum
    df['Return1'] = df['Close'].pct_change().fillna(0)
    df['Return3'] = df['Close'].pct_change(3).fillna(0)
    df['Momentum5'] = (df['Close'] - df['Close'].shift(5)).fillna(0)

    # Volatility and Bollinger Bands
    df['Volatility20'] = df['Close'].rolling(window=20, min_periods=1).std().fillna(0)
    df['BB_Mid'] = df['MA20']
    df['BB_Upper'] = df['BB_Mid'] + 2 * df['Volatility20']
    df['BB_Lower'] = df['BB_Mid'] - 2 * df['Volatility20']

    # RSI14 (classic)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(window=14, min_periods=1).mean()
    roll_down = loss.rolling(window=14, min_periods=1).mean().replace(0, 1e-10)
    rs = roll_up / roll_down
    df['RSI14'] = 100 - (100 / (1 + rs))

    # ATR14
    if all(col in df.columns for col in ['High', 'Low']):
        prev_close = df['Close'].shift(1)
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - prev_close).abs()
        tr3 = (df['Low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR14'] = tr.rolling(window=14, min_periods=1).mean()
    else:
        df['ATR14'] = 0  # instead of NaN

    # Volume change
    if 'Volume' in df.columns:
        df['Volume_Change'] = df['Volume'].pct_change().fillna(0)
    else:
        df['Volume_Change'] = 0  # instead of NaN

    # 🔎 DEBUG: Check NaN counts per column
    logger.info("DEBUG: NaN counts after indicators:\n", df.isna().sum())

    return df