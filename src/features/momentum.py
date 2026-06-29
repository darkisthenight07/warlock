from __future__ import annotations
import pandas as pd
import numpy as np
import talib
from .base import consecutive_streak
from src.utils import config

_CFG = config["features"]["momentum"]

def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["RSI"] = talib.RSI(
        out["close"],
        timeperiod=_CFG["rsi_window"],
    )

    out["ADX"] = talib.ADX(
        out["high"],
        out["low"],
        out["close"],
        timeperiod=_CFG["adx_window"],
    )

    out["ROC"] = talib.ROC(
        out["close"],
        timeperiod=10,
    )

    macd, signal, hist = talib.MACD(
        out["close"],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9,
    )

    out["MACD_hist"] = hist

    out["up_streak"] = consecutive_streak(
        out["close"].diff() > 0
    )

    return out
