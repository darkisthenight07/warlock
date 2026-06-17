from __future__ import annotations
import pandas as pd
import numpy as np
from .base import safe_divide

def volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    vol_mean = out["volume_norm"].rolling(20, min_periods=20).mean()
    vol_std = out["volume_norm"].rolling(20, min_periods=20).std()
    out["vol_zscore"] = safe_divide(out["volume_norm"] - vol_mean, vol_std)

    direction = np.where(
        out["close"] > out["close"].shift(), 1,
        np.where(out["close"] < out["close"].shift(), -1, 0)
    )
    out["obv"] = (direction * out["volume_norm"]).cumsum()

    obv_mean = out["obv"].rolling(20, min_periods=20).mean()
    obv_std = out["obv"].rolling(20, min_periods=20).std()
    out["obv_zscore"] = safe_divide(out["obv"] - obv_mean, obv_std)
    out["vol_vs_ma20"] = safe_divide(out["volume_norm"], vol_mean)

    return out
