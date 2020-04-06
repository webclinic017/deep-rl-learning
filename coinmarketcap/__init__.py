from binance.client import Client
import pandas as pd

api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
client = Client(api_key, api_secret)


def convert_data(obj):
    open_time = obj[0]/1000
    open = obj[1]
    high = obj[2]
    low = obj[3]
    close = obj[4]
    volume = obj[5]
    num_of_trade = obj[8]
    return_data = {
        # 'open': float(open),
        # 'high': float(high),
        # 'low': float(low),
        'open_time': float(open_time),
        'close': float(close),
        'volume': float(volume),
        'num_of_trade': float(num_of_trade),
    }
    return return_data


# fetch train data
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, "1 Jan, 2019", "1 Feb, 2020")
train_df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close',
                                         'volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                                         'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])
train_df.drop(columns=['open_time', 'close_time', 'quote_asset_volume', 'number_of_trades',
                       'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'], axis=1)
train_df.to_csv('../data/train_5m.csv', sep=',')


# fetch test data
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, "1 Feb, 2020")
df_test = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close',
                                        'volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                                        'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])
df_test.drop(columns=['open_time', 'close_time', 'quote_asset_volume', 'number_of_trades',
                      'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'], axis=1)
df_test.to_csv('../data/test_5m.csv', sep=',')

# test_data = list(map(convert_data, klines))
# print(test_data[0])

# [
#   [
#     1499040000000,      // Open time
#     "0.01634790",       // Open
#     "0.80000000",       // High
#     "0.01575800",       // Low
#     "0.01577100",       // Close
#     "148976.11427815",  // Volume
#     1499644799999,      // Close time
#     "2434.19055334",    // Quote asset volume
#     308,                // Number of trades
#     "1756.87402397",    // Taker buy base asset volume
#     "28.46694368",      // Taker buy quote asset volume
#     "17928899.62484339" // Ignore.
#   ]
# ]
