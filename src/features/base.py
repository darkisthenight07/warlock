from __future__ import annotations
import pandas as pd
import numpy as np


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Element‑wise division that yields NaN where denominator is 0."""
    return np.where(den == 0, np.nan, num / den)


def cyclical_encode(series: pd.Series, period: int) -> pd.DataFrame:
    """
    Encode an integer series (e.g., hour of day) as two cyclic features.
    Returns a DataFrame with columns `<name>_sin` and `<name>_cos`.
    """
    radians = 2 * np.pi * series / period
    return pd.DataFrame(
        {
            f"{series.name}_sin": np.sin(radians),
            f"{series.name}_cos": np.cos(radians),
        }
    )