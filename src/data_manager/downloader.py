import ccxt
import time
import pandas as pd
from src.utils import config

def download():
    symbols = config['data']['symbols']
    data_dir = config['paths']['raw_dir']
    exchange = getattr(ccxt, config['data']['exchange'])()
    
    for symbol in symbols:
        print(f'Downloading {symbol}...')
        all_candles = []
        cursor = exchange.parse8601(f"{config['data']['start_date']}T00:00:00Z")
        end = exchange.parse8601(f"{config['data']['end_date']}T23:59:59Z")

        while cursor < end:
            try:
                batch = exchange.fetch_ohlcv(symbol, config['data']['timeframe'], cursor)
                if not batch:
                    break
                all_candles.extend(batch)
                cursor = batch[-1][0] + 1
                time.sleep(exchange.rateLimit / 1000)
            except ccxt.NetworkError as e:
                print(f"Network error: {e}. Retrying...")
                continue
            except ccxt.ExchangeError as e:
                print(f"Exchange error: {e}")
        
        all_candles = [candle for candle in all_candles if candle[0] <= end]
        data = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        symbol = symbol.split('/')[0]
        file_path = data_dir + '/'+ symbol + "_raw.parquet"
        data.to_parquet(file_path)
        print(f'Saved {symbol} data to {file_path}')

if __name__ == "__main__":
    download()