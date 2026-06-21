from __future__ import annotations
import pandas as pd
import numpy as np
from .base import safe_divide
from src.utils import config

_CFG = config["features"]["candle"]

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

    return out