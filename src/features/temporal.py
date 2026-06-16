from __future__ import annotations
import pandas as pd
from .base import cyclical_encode


def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Ensure timestamp is timezone‑aware (UTC). The cleaning step already set this.
    out["hour"] = out["timestamp"].dt.hour
    out["dow"]  = out["timestamp"].dt.dayofweek   # Monday=0

    hour_enc = cyclical_encode(out["hour"], period=24)
    dow_enc  = cyclical_encode(out["dow"], period=7)

    out = pd.concat([out, hour_enc, dow_enc], axis=1)

    # Drop the raw integer columns – they are no longer needed
    out.drop(columns=["hour", "dow"], inplace=True)

    return out