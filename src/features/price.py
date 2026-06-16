from __future__ import annotations
import pandas as pd
import numpy as np
from .base import safe_divide

try:
    import talib
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for p in [1, 4, 24]:
        out[f"return_{p}h"] = out["close"].pct_change(p)
        out[f"logret_{p}h"] = np.log1p(out[f"return_{p}h"])

    for w in [20, 50, 200]:
        ma = out["close"].rolling(w, min_periods=w).mean()
        out[f"price_vs_ma{w}"] = safe_divide(out["close"], ma)

    if _HAS_TALIB:
        upper, middle, lower = talib.BBANDS(
            out["close"], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
    else:
        window = 20
        rolling_std = out["close"].rolling(window, min_periods=window).std()
        middle = out["close"].rolling(window, min_periods=window).mean()
        upper = middle + 2 * rolling_std
        lower = middle - 2 * rolling_std

    band_width = upper - lower
    out["bb_zscore"] = safe_divide(out["close"] - middle, band_width)

    return out
