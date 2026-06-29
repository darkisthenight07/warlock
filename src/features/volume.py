from __future__ import annotations
import pandas as pd
import numpy as np
from .base import rolling_zscore
from src.utils import config

_CFG = config["features"]["volume"]

def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["volume_zscore"] = rolling_zscore(out["volume"], window=_CFG["zscore_window"])
    volume_ma = out["volume"].rolling(
    window=_CFG["relative_volume_window"],
    min_periods=_CFG["relative_volume_window"],
     ).mean()

    out["relative_volume"] = out["volume"] / volume_ma

    direction = np.where(
        out["close"] > out["close"].shift(), 1,
        np.where(out["close"] < out["close"].shift(), -1, 0)
    )
    out["OBV"] = (direction * out["volume"]).cumsum()

    return out
