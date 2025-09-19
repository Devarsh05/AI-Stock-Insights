# src/ml_model.py
import numpy as np
import logging
import pandas as pd
from typing import Tuple, List, Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

from analysis import add_technical_indicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_FEATURES = [
    'Close','MA10','MA20','MA50','MA_diff',
    'Return1','Return3','Momentum5','Volatility20',
    'ATR14','Volume_Change','RSI14'
]

def prepare_dataset(df: pd.DataFrame, horizon: int = 1, features: List[str] = None
                   ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns X, y for training.
    - df: DataFrame with OHLCV
    - horizon: days ahead to predict (1 = next day)
    - features: list of column names to use (default uses DEFAULT_FEATURES filtered by availability)
    """
    df = add_technical_indicators(df)

    if features is None:
        features = DEFAULT_FEATURES

    # keep only features that exist
    features = [f for f in features if f in df.columns]

    if 'Close' not in df.columns:
        raise KeyError("prepare_dataset requires 'Close' in dataframe")

    df = df.copy()
    df['Target'] = df['Close'].shift(-horizon)

    # Drop missing values
    df = df.dropna(subset=features + ['Target'])

    # 🔎 DEBUGGING PRINTS
    logger.info("DEBUG: Total rows after dropna:", len(df))
    logger.info("DEBUG: Using horizon:", horizon)
    logger.info("DEBUG: Features used:", features)

    # Safety check
    if len(df) < 100:  
        raise ValueError(
            f"Not enough rows to train (have {len(df)} rows, need >= 100). "
            "Try a longer period in data_fetch.py (e.g., '2y' or '5y')."
        )

    X = df[features].copy()
    y = df['Target'].copy()

    return X, y


def time_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    n = len(X)
    split_at = int((1 - test_size) * n)
    X_train = X.iloc[:split_at]
    X_test = X.iloc[split_at:]
    y_train = y.iloc[:split_at]
    y_test = y.iloc[split_at:]
    return X_train, X_test, y_train, y_test

def build_model(model_type: str = 'rf', random_state: int = 42):
    model_type = model_type.lower()
    if model_type == 'rf':
        model = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    elif model_type == 'gbr':
        model = GradientBoostingRegressor(n_estimators=200, random_state=random_state)
    elif model_type == 'lr':
        # linear regression with scaling
        model = make_pipeline(StandardScaler(), LinearRegression())
    else:
        raise ValueError("model_type must be one of 'rf','gbr','lr'")
    return model

def train_and_evaluate(X: pd.DataFrame, y: pd.Series, model_type: str = 'rf', test_size: float = 0.2
                       ) -> Dict[str, Any]:
    """
    Trains model and returns dict with:
    - model
    - metrics: rmse, mae, r2, directional_accuracy
    - preds, y_test, X_test
    - feature_importances (dict), if available or via permutation
    """
    if len(X) < 200:
        raise ValueError("Not enough rows to train (need >= 10).")
    X_train, X_test, y_train, y_test = time_train_test_split(X, y, test_size=test_size)
    model = build_model(model_type)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # directional accuracy
    prev_close = X_test['Close'].values if 'Close' in X_test.columns else None
    if prev_close is not None:
        actual_change = y_test.values - prev_close
        pred_change = preds - prev_close
        directional_accuracy = float(((actual_change * pred_change) > 0).mean())
    else:
        directional_accuracy = float(np.nan)

    # feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fi = dict(sorted(zip(X.columns.tolist(), importances), key=lambda x: x[1], reverse=True))
    else:
        # fallback: permutation importance (can be slower)
        try:
            perm = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, n_jobs=-1)
            fi = dict(sorted(zip(X.columns.tolist(), perm.importances_mean), key=lambda x: x[1], reverse=True))
        except Exception:
            fi = {c: 0.0 for c in X.columns.tolist()}

    return {
        'model': model,
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'directional_accuracy': float(directional_accuracy),
        'preds': preds,
        'y_test': y_test,
        'X_test': X_test,
        'feature_importances': fi
    }

def predict_with_uncertainty(model: Any, X_latest: np.ndarray) -> Tuple[float, float]:
    """
    Returns (mean_prediction, std_estimate) where std_estimate is None if not available.
    For RandomForestRegressor: use per-estimator predictions to estimate std.
    X_latest: 2D array shaped (1, n_features)
    """
    # RandomForest has estimators_
    if hasattr(model, 'estimators_'):
        all_preds = np.array([est.predict(X_latest.reshape(1, -1))[0] for est in model.estimators_])
        return float(all_preds.mean()), float(all_preds.std(ddof=0))
    else:
        # other models: no per-tree uncertainty; return model.predict and std = nan
        pred = model.predict(X_latest.reshape(1, -1))[0]
        return float(pred), float(np.nan)
