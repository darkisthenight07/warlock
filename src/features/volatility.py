from __future__ import annotations
import pandas as pd
import numpy as np


def volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    high_low = out["high"] - out["low"]
    high_close = np.abs(out["high"] - out["close"].shift())
    low_close = np.abs(out["low"] - out["close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    out["atr_14"] = true_range.ewm(alpha=1 / 14, adjust=False).mean()

    if "logret_1h" not in out.columns:
        out["logret_1h"] = np.log1p(out["close"].pct_change())

    out["realvol_24h"] = out["logret_1h"].rolling(24, min_periods=24).std()
    out["realvol_72h"] = out["logret_1h"].rolling(72, min_periods=72).std()

    return out
