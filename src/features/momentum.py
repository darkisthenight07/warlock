from __future__ import annotations
import pandas as pd
import numpy as np
import talib

def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rsi_14"] = talib.RSI(out["close"], timeperiod=14)

    macd, macd_signal, _ = talib.MACD(
        out["close"], fastperiod=12, slowperiod=26, signalperiod=9
    )
    out["macd"] = macd
    out["macd_signal"] = macd_signal

    out["adx_14"] = talib.ADX(
        out["high"], out["low"], out["close"], timeperiod=14
    )

    return out
