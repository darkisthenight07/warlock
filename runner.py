import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from loguru import logger
from utils import config
from data_manager.data_cleaning import clean_ohlcv


def main():
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/cleaning_{time}.log",
        rotation="1 day",
        retention="7 days",
        level="Debug",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    )

    symbols       = config["data"]["symbols"]
    timeframe     = config["data"]["timeframe"]
    raw_dir       = config["paths"]["raw_dir"]
    processed_dir = config["paths"]["processed_dir"]

    logger.info(f"Symbols: {symbols} | Timeframe: {timeframe}")
    logger.info(f"Raw → {raw_dir} | Processed → {processed_dir}")

    for symbol in symbols:
        try:
            clean_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
            )
        except FileNotFoundError as e:
            logger.error(f"Skipping {symbol} — {e}")
        except Exception as e:
            logger.exception(f"Failed on {symbol} — {e}")
            raise


if __name__ == "__main__":
    main()
