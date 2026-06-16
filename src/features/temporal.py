from __future__ import annotations
import pandas as pd
from .base import cyclical_encode


def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["hour"] = out["timestamp"].dt.hour
    out["dow"]  = out["timestamp"].dt.dayofweek

    hour_enc = cyclical_encode(out["hour"], period=24)
    dow_enc  = cyclical_encode(out["dow"], period=7)

    out = pd.concat([out, hour_enc, dow_enc], axis=1)
    out.drop(columns=["hour", "dow"], inplace=True)

    return out
