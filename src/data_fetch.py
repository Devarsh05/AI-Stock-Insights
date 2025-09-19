# src/data_fetch.py
import logging
import yfinance as yf
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_stock_data(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV historical data from Yahoo Finance.
    - period default is '5y' to ensure enough rows for indicators.
    - auto_adjust=True to prevent the FutureWarning and to adjust for splits/dividends.
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True
    )
    logger.info(f"DEBUG: Fetched {len(df)} rows for {ticker} with period={period}, interval={interval}")
    if df is None or df.empty:
        raise ValueError(f"No data found for ticker '{ticker}' with period='{period}'")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df
