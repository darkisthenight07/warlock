from __future__ import annotations
import pandas as pd
import numpy as np

try:
    import talib
    _HAS_TALIB = True
except Exception:
    _HAS_TALIB = False


def momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if _HAS_TALIB:
        out["rsi_14"] = talib.RSI(out["close"], timeperiod=14)

        macd, macd_signal, _ = talib.MACD(
            out["close"], fastperiod=12, slowperiod=26, signalperiod=9
        )
        out["macd"] = macd
        out["macd_signal"] = macd_signal

        out["adx_14"] = talib.ADX(
            out["high"], out["low"], out["close"], timeperiod=14
        )
    else:
        delta = out["close"].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)

        roll_up = up.ewm(alpha=1 / 14, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / 14, adjust=False).mean()
        rs = roll_up / roll_down
        out["rsi_14"] = 100 - (100 / (1 + rs))

        fast = out["close"].ewm(span=12, adjust=False).mean()
        slow = out["close"].ewm(span=26, adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=9, adjust=False).mean()
        out["macd"] = macd
        out["macd_signal"] = signal

        high_low = out["high"] - out["low"]
        high_close = np.abs(out["high"] - out["close"].shift())
        low_close = np.abs(out["low"] - out["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        plus_dm = out["high"].diff()
        minus_dm = -out["low"].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr14 = tr.rolling(14).sum()
        plus14 = plus_dm.rolling(14).sum()
        minus14 = minus_dm.rolling(14).sum()

        plus_di = 100 * (plus14 / tr14)
        minus_di = 100 * (minus14 / tr14)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)

        out["adx_14"] = dx.rolling(14).mean()

    return out
