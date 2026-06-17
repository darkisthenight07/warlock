import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.utils import root, config

def last_n_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise KeyError("Dataframe must contain a 'timestamp' column.")
    df = df.sort_values("timestamp")
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    return df[df["timestamp"] >= cutoff].copy()


def plot_features(df: pd.DataFrame, out_dir: str = config['paths']['feature_graphs_dir']) -> None:
    graph_path = root(out_dir)
    graph_path.mkdir(parents=True, exist_ok=True)

    raw_cols = {"timestamp", "open", "high", "low", "close", "volume"}
    feature_cols = [
        c for c in df.select_dtypes(include=["float", "int"]).columns
        if c not in raw_cols
    ]

    sns.set_style("whitegrid")
    sns.set_context("talk", font_scale=0.9)

    for col in feature_cols:
        plot_df = df

        if col in {"hour_sin", "hour_cos"}:
            plot_df = last_n_days(df, days=3)

        elif col in {"dow_sin", "dow_cos"}:
            plot_df = last_n_days(df, days=21)   # 3 weeks

            if col == "dow_sin":
                circle_path = graph_path / "dow_sin_vs_cos_circle.png"
                plt.figure(figsize=(5, 5))
                plt.scatter(df["dow_sin"], df["dow_cos"], s=10, alpha=0.6,
                            color="steelblue", edgecolor="none")
                theta = np.linspace(0, 2 * np.pi, 200)
                plt.plot(np.cos(theta), np.sin(theta), "r--", lw=1)
                plt.title("dow_sin vs. dow_cos – perfect unit circle")
                plt.xlabel("dow_sin")
                plt.ylabel("dow_cos")
                plt.axis("equal")
                plt.tight_layout()
                plt.savefig(circle_path, dpi=150)
                plt.close()

        elif col == "atr_14":
            if "close" not in df.columns:
                raise KeyError("Column 'close' required to compute NATR.")
            natr = (df["atr_14"] / df["close"]).replace([np.inf, -np.inf], np.nan) * 100
            plot_df = df.copy()
            plot_df["natr_14"] = natr
            col = "natr_14"

        elif col == "bb_zscore":
            plot_df = last_n_days(df, days=30)

            fig, (ax_price, ax_z) = plt.subplots(
                nrows=2, ncols=1, sharex=True, figsize=(12, 6),
                gridspec_kw={"height_ratios": [2, 1]}
            )
            ax_price.plot(plot_df["timestamp"], plot_df["close"],
                        color="steelblue", linewidth=0.9)
            ax_price.set_ylabel("BTC Close")
            ax_price.set_title("BTC Price (30-day window)")

            ax_z.plot(plot_df["timestamp"], plot_df["bb_zscore"],
                    color="darkorange", linewidth=0.9)
            ax_z.axhline(0, color="black", lw=0.7, ls="--")
            ax_z.axhline(1, color="red",   lw=0.7, ls="--")
            ax_z.axhline(-1, color="red",  lw=0.7, ls="--")
            ax_z.set_ylabel("BB Z-score")
            ax_z.set_xlabel("Timestamp")

            plt.tight_layout()
            safe_name = f"{col}_dual"
            plt.savefig(graph_path / f"{safe_name}.png", dpi=150)
            plt.close()
            continue

        elif col == "adx_14":
            roll_mean = df["adx_14"].rolling(14, min_periods=1).mean()
            plot_df = df.copy()
            plot_df["adx_14_roll14"] = roll_mean

        elif col in {"return_1h", "return_4h", "return_24h",
                    "logret_1h", "logret_4h", "logret_24h"}:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), bins=100, kde=True,
                        color="steelblue", edgecolor="white")
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            safe_name = f"{col}_hist"
            plt.savefig(graph_path / f"{safe_name}.png", dpi=150)
            plt.close()
            continue

        elif col in {"macd", "macd_signal"}:
            if "close" not in df.columns:
                raise KeyError("Column 'close' required for MACD normalisation.")
            macd_norm = (df["macd"] / df["close"]).replace([np.inf, -np.inf], np.nan) * 100
            macd_sig_norm = (df["macd_signal"] / df["close"]).replace([np.inf, -np.inf], np.nan) * 100
            macd_df = pd.DataFrame({
                "timestamp": df["timestamp"],
                "macd_norm": macd_norm,
                "macd_signal_norm": macd_sig_norm,
            })
            macd_df = last_n_days(macd_df, days=30)

            plt.figure(figsize=(12, 4))
            plt.plot(macd_df["timestamp"], macd_df["macd_norm"],
                    label="MACD (norm %)", color="steelblue", linewidth=0.9)
            plt.plot(macd_df["timestamp"], macd_df["macd_signal_norm"],
                    label="Signal (norm %)", color="orange", linewidth=0.9)
            plt.title("Normalised MACD & Signal (30-day window)")
            plt.xlabel("Timestamp")
            plt.ylabel("MACD (% of price)")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(graph_path / "macd_normalised.png", dpi=150)
            plt.close()
            continue

        elif col == "obv":
            obv_diff = df["obv"].diff().fillna(0)
            obv_z = (obv_diff - obv_diff.mean()) / obv_diff.std()
            obv_df = pd.DataFrame({
                "timestamp": df["timestamp"],
                "obv_rate": obv_diff,
                "obv_rate_z": obv_z,
            })
            obv_df = last_n_days(obv_df, days=14)

            plt.figure(figsize=(12, 4))
            plt.plot(obv_df["timestamp"], obv_df["obv_rate_z"],
                    label="OBV rate Z-score", color="steelblue", linewidth=0.9)
            plt.title("OBV Rate-of-Change Z-score (14-day window)")
            plt.xlabel("Timestamp")
            plt.ylabel("Z-score")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(graph_path / "obv_rate_zscore.png", dpi=150)
            plt.close()
            continue

        elif col == "obv_zscore":
            obv_sub = last_n_days(df, days=14)
            fig, (ax_price, ax_obv) = plt.subplots(
                nrows=2, ncols=1, sharex=True, figsize=(12, 6),
                gridspec_kw={"height_ratios": [2, 1]}
            )
            ax_price.plot(obv_sub["timestamp"], obv_sub["close"],
                        color="steelblue", linewidth=0.9)
            ax_price.set_ylabel("BTC Close")
            ax_price.set_title("BTC Price (14-day window)")

            ax_obv.plot(obv_sub["timestamp"], obv_sub["obv_zscore"],
                        color="darkorange", linewidth=0.9)
            ax_obv.set_ylabel("OBV Z-score")
            ax_obv.set_xlabel("Timestamp")
            plt.tight_layout()
            plt.savefig(graph_path / "obv_zscore_dual.png", dpi=150)
            plt.close()
            continue

        elif col.startswith("price_vs_ma"):
            pct_dist = (df[col] - 1) * 100
            tmp_df = df.copy()
            tmp_df[col + "_pct"] = pct_dist
            plot_df = last_n_days(tmp_df, days=90)
            plt.figure(figsize=(12, 4))
            plt.plot(plot_df["timestamp"], plot_df[col + "_pct"],
                    label=f"{col} (% diff)", color="steelblue", linewidth=0.9)
            plt.axhline(0, color="black", lw=0.7, ls="--")
            plt.title(f"{col} expressed as % distance from MA (90-day window)")
            plt.xlabel("Timestamp")
            plt.ylabel("% diff")    
            plt.legend(loc="upper right")
            plt.tight_layout()
            safe_name = f"{col}_pct"
            plt.savefig(graph_path / f"{safe_name}.png", dpi=150)
            plt.close()
            continue

        elif col in {"realvol_24h", "realvol_72h"}:
            smooth = df[col].rolling(7, min_periods=1).mean()
            plt.figure(figsize=(12, 4))
            plt.plot(df["timestamp"], smooth,
                    label=f"{col} (7-day MA)", color="steelblue", linewidth=0.9)
            plt.title(f"{col} – 7-day rolling average")
            plt.xlabel("Timestamp")
            plt.ylabel(col)
            plt.legend(loc="upper right")
            plt.tight_layout()
            safe_name = f"{col}_smooth"
            plt.savefig(graph_path / f"{safe_name}.png", dpi=150)
            plt.close()
            continue

        elif col == "return_1h":
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), bins=100, kde=True,
                        color="steelblue", edgecolor="white")
            plt.title("Histogram of return_1h")
            plt.xlabel("return_1h")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(graph_path / "return_1h_hist.png", dpi=150)
            plt.close()
            continue

        elif col == "rsi_14":
            plt.figure(figsize=(12, 4))
            plt.plot(df["timestamp"], df["rsi_14"],
                    label="RSI (14)", color="steelblue", linewidth=0.9)
            plt.axhline(70, color="red",   lw=0.7, ls="--", label="Over-bought (70)")
            plt.axhline(30, color="green", lw=0.7, ls="--", label="Over-sold (30)")
            plt.title("RSI (14) – over-bought / over-sold zones")
            plt.xlabel("Timestamp")
            plt.ylabel("RSI")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(graph_path / "rsi_14.png", dpi=150)
            plt.close()
            continue

        elif col in {"return_4h", "return_24h"}:
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), bins=100, kde=True,
                        color="steelblue", edgecolor="white")
            plt.title(f"Histogram of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.tight_layout()
            plt.savefig(graph_path / f"{col}_hist.png", dpi=150)
            plt.close()
            continue

        elif col in {"volume_norm", "vol_zscore"}:
            pass

        elif col == "vol_vs_ma20":
            price_change = df["close"].pct_change().fillna(0)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=price_change, y=df["vol_vs_ma20"],
                            alpha=0.5, edgecolor=None, color="steelblue")
            plt.title("Scatter: Price %-change vs Volume/MA20 ratio")
            plt.xlabel("Price %-change (Δclose)")
            plt.ylabel("vol_vs_ma20")
            plt.tight_layout()
            plt.savefig(graph_path / "vol_vs_ma20_scatter.png", dpi=150)
            plt.close()
            continue

        plt.figure(figsize=(12, 4))
        plt.plot(plot_df["timestamp"], plot_df[col],
                linewidth=0.8, label=col, color="steelblue")

        if col == "adx_14":
            plt.plot(plot_df["timestamp"], plot_df["adx_14_roll14"],
                    linewidth=1.2, label="ADX (14-day MA)", color="gray", alpha=0.8)

        plt.title(f"{col} over time")
        plt.xlabel("Timestamp")
        plt.ylabel(col)
        plt.legend(loc="upper right")
        plt.tight_layout()

        # Save the figure
        safe_name = col.replace("/", "_").replace("\\", "_")
        plt.savefig(graph_path / f"{safe_name}.png", dpi=150)
        plt.close()

    logger.success(f"\nAll feature graphs written to: {graph_path}\n")
