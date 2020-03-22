import pandas as pd
import talib
from binance.enums import KLINE_INTERVAL_1MINUTE
from talib import MA_Type
from talib._ta_lib import RSI, BBANDS, MA, MOM, MACD
from binance.client import Client
from binance.websockets import BinanceSocketManager
import matplotlib.pyplot as plt


class RegressionMA:
    def __init__(self):
        self.train_data = self.get_data()
        self.budget = 0
        self.order = 0
        self.prev_histogram = 0

    def get_data(self):
        api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        binaci_client = Client(api_key, api_secret)
        klines = binaci_client.get_historical_klines("BTCUSDT", KLINE_INTERVAL_1MINUTE, "22 Mar, 2020")
        df = pd.DataFrame(klines, columns=['open_time', 'Open', 'High', 'Low', 'Close',
                                                'Volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                                                'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])

        df['MA'] = MA(df.Close, timeperiod=14)
        df['MACD'], df['Signal'], df['Histogram'] = MACD(df.Close, 12, 26, 9)
        # plt.plot(df.MA, label='MA')
        # plt.plot(df.MACD, label='MACD', color='green')
        # plt.plot(df.Signal, label='Signal', color="red")
        # plt.plot(df.Histogram, label='Histogram', color="blue")
        # plt.axhline(0)
        # plt.legend()
        return df

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        binaci_client = Client(api_key, api_secret)
        bm = BinanceSocketManager(binaci_client)
        conn_key = bm.start_kline_socket('BTCUSDT', self.process_message, interval=KLINE_INTERVAL_1MINUTE)
        # then start the socket manager
        bm.start()

    def process_message(self, msg):
        if self.train_data.open_time.values[-1] == msg['k']['t']:
            print("change value")
            self.train_data.at[len(self.train_data) - 1, 'Close'] = msg['k']['c']
        if self.train_data.open_time.values[-1] != msg['k']['t']:
            _open_time = msg['k']['t']
            _open = msg['k']['o']
            _high = msg['k']['h']
            _low = msg['k']['l']
            _close = msg['k']['c']
            _volume = msg['k']['v']
            _close_time = msg['k']['T']
            _quote_asset_volume = msg['k']['Q']
            _number_of_trades = msg['k']['n']
            _buy_base_asset_volume = msg['k']['V']
            _buy_quote_asset_volume = msg['k']['q']
            _ignore = msg['k']['B']
            df = pd.DataFrame([[_open_time, _open, _high, _low, _close, _volume, _close_time,
                                _quote_asset_volume, _number_of_trades, _buy_base_asset_volume,
                                _buy_quote_asset_volume, _ignore]],
                              columns=['open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])
            self.train_data = self.train_data.append(df, ignore_index=True, sort=False)
            print("insert value")
        self.trading(self.train_data)

    def trading(self, df):
        df['MA'] = MA(df.Close, timeperiod=14)
        df['MACD'], df['Signal'], df['Histogram'] = MACD(df.Close, 12, 26, 9)
        data = self.train_data.dropna()
        price = data.Close.astype('float64').values[-1]
        ma = data.MA.values[-1]
        histogram = data.Signal.values[-1]

        if not self.order and price > ma and histogram > self.prev_histogram:
            # buy signal
            self.order = price
        elif self.order and self.prev_histogram > histogram:
            diff = price - self.order
            self.budget += diff
            self.order = 0
            print(self.budget)

        if self.order and price - self.order < -5:
            diff = price - self.order
            self.budget += diff
            self.order = 0
            print(self.budget)

        self.prev_histogram = histogram

    def test_trading(self, df):
        df['MA'] = MA(df.Close, timeperiod=14)
        df['MACD'], df['Signal'], df['Histogram'] = MACD(df.Close, 12, 26, 9)
        data = df.dropna()
        close_price = data.Close.astype('float64').values
        ma_signal = data.MA.values
        histogram_signal = data.Signal.values

        prev_histogram = histogram_signal[0]
        for price, ma, histogram in zip(close_price[1:], ma_signal[1:], histogram_signal[1:]):
            if not self.order and price > ma and histogram > prev_histogram:
                # buy signal
                self.order = price
            elif self.order and prev_histogram > histogram:
                diff = price - self.order
                self.budget += diff
                self.order = 0
                print(self.budget)

            if self.order and price - self.order < -5:
                diff = price - self.order
                self.budget += diff
                self.order = 0
                print(self.budget)

            prev_histogram = histogram


if __name__ == '__main__':
    bottrading = RegressionMA()
    bottrading.get_data()
    bottrading.start_socket()
