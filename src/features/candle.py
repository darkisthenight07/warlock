from __future__ import annotations
import pandas as pd
import numpy as np
from .base import safe_divide
from src.utils import config

def candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single-bar candle geometry features: where the close sits within the
    bar's range, how much of the range is "body" vs wick, and whether the
    bar has an abnormally large wick relative to recent typical range.
    """
    out = df.copy()
    bar_range = out["high"] - out["low"]

    # Close Location Value: 0 = closed at the low, 1 = closed at the high
    out["close_location"] = safe_divide(out["close"] - out["low"], bar_range)

    # Body size as a % of the full high-low range
    out["body_pct"] = safe_divide((out["close"] - out["open"]).abs(), bar_range) * 100
    upper_wick = out["high"] - np.maximum(out["open"], out["close"])
    lower_wick = np.minimum(out["open"], out["close"]) - out["low"]

    out["upper_wick_pct"] = safe_divide(upper_wick, bar_range) * 100
    out["lower_wick_pct"] = safe_divide(lower_wick, bar_range) * 100
    
    rolling_upper = out["upper_wick_pct"].rolling(
    window=20,
    min_periods=20,
     ).median()

    rolling_lower = out["lower_wick_pct"].rolling(
    window=20,
    min_periods=20,
    ).median()

    out["wick_anomaly_flag"] = (
    (out["upper_wick_pct"] > 2.0 * rolling_upper)
    | (out["lower_wick_pct"] > 2.0 * rolling_lower)
).astype(int)

    return out
