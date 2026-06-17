from __future__ import annotations
import pandas as pd
import numpy as np
from .base import safe_divide
import talib

def price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for p in [1, 4, 24]:
        out[f"return_{p}h"] = out["close"].pct_change(p)
        out[f"logret_{p}h"] = np.log1p(out[f"return_{p}h"])

    for w in [20, 50, 200]:
        ma = out["close"].rolling(w, min_periods=w).mean()
        out[f"price_vs_ma{w}"] = safe_divide(out["close"], ma)

    upper, middle, lower = talib.BBANDS(
        out["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    return out