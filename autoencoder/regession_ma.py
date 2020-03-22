import time
import logging
import pandas as pd
import talib
from binance.enums import KLINE_INTERVAL_1MINUTE, KLINE_INTERVAL_1HOUR, KLINE_INTERVAL_5MINUTE
from pymongo import MongoClient
from talib import MA_Type
from talib._ta_lib import RSI, BBANDS, MA, MOM, MACD, DX, MINUS_DM, PLUS_DM, MINUS_DI, PLUS_DI, ADX
from binance.client import Client
from binance.websockets import BinanceSocketManager
import matplotlib.pyplot as plt

logging.basicConfig(filename='../log/autotrade.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class RegressionMA:
    def __init__(self):
        self.train_data = self.get_data()
        self.budget = 0
        self.order = 0
        self.prev_histogram = 0
        self.max_diff = 0
        self.client = MongoClient()
        self.db = self.client.crypto

    def get_data(self):
        api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        binaci_client = Client(api_key, api_secret)
        klines = binaci_client.get_historical_klines("BTCUSDT", KLINE_INTERVAL_1HOUR, "1 Jan, 2019", "30 Jan, 2019")
        df = pd.DataFrame(klines, columns=['open_time', 'Open', 'High', 'Low', 'Close',
                                           'Volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])

        # df['MA'] = MA(df.Close, timeperiod=14)
        # df['MACD'], df['Signal'], df['Histogram'] = MACD(df.Close, 12, 26, 9)
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
        conn_key = bm.start_kline_socket('BTCUSDT', self.process_message, interval=KLINE_INTERVAL_1HOUR)
        # then start the socket manager
        bm.start()

    def getStockDataVec(self, key='train1hour'):
        indexes = []
        lines = open("../data/" + key + ".csv", "r").read().splitlines()
        prices = []
        delimiter = ','
        for _index, line in enumerate(lines[1:]):

            msg = {
                "e": "kline",					# event type
                "E": 1499404907056,				# event time
                "s": "BTCUSDT",					# symbol
                "k": {
                    "t": line[1], 		# start time of this bar
                    "T": line[7], 		# end time of this bar
                    "s": "ETHBTC",				# symbol
                    "i": "1m",					# interval
                    "f": 77462,					# first trade id
                    "L": 77465,					# last trade id
                    "o": line[2],			# open
                    "c": line[5],			# close
                    "h": line[3],			# high
                    "l": line[4],			# low
                    "v": line[6],			# volume
                    "n": 4,						# number of trades
                    "x": False,					# whether this bar is final
                    "q": "1.79662878",			# quote volume
                    "V": "2.34879839",			# volume of active buy
                    "Q": "0.24142166",			# quote volume of active buy
                    "B": "13279784.01349473"	# can be ignored
                    }
            }

            self.process_message(msg)

    def process_message(self, msg):
        msg['k']['timestamp'] = time.time()
        insert = self.db.btc_data.insert_one(msg).inserted_id
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

        if msg['k']['x']:
            df = pd.DataFrame([[_open_time, _open, _high, _low, _close, _volume, _close_time,
                                _quote_asset_volume, _number_of_trades, _buy_base_asset_volume,
                                _buy_quote_asset_volume, _ignore]],
                              columns=['open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])
            self.train_data = self.train_data.append(df, ignore_index=True, sort=False)

        else:
            # if self.train_data.open_time.values[-1] == msg['k']['t']:
            self.train_data.at[len(self.train_data) - 1, 'Close'] = _close
            self.train_data.at[len(self.train_data) - 1, 'High'] = _high
            self.train_data.at[len(self.train_data) - 1, 'Low'] = _low
        self.trading(self.train_data)

    def trading(self, df):
        df['MA'] = MA(df.Close, timeperiod=14)
        df['MACD'], df['Signal'], df['Histogram'] = MACD(df.Close, 12, 26, 9)
        df['MINUS_DI'] = MINUS_DI(df.High, df.Low, df.Close)
        df['PLUS_DI'] = PLUS_DI(df.High, df.Low, df.Close)
        df['ADX'] = ADX(df.High, df.Low, df.Close)
        data = self.train_data.dropna()
        close_price = data.Close.astype('float64').values
        price = close_price[-1]
        ma = data.MA.values[-1]
        macd = data.MACD.values[-1]
        signal = data.Signal.values[-1]
        histogram = data.Histogram.values[-1]
        minus_di = data.MINUS_DM.values[-1]
        plus_di = data.PLUS_DM.values[-1]
        adx_di = data.ADX.values[-1]

        prev_histogram = histogram[-2]
        if not self.order and histogram > prev_histogram and plus_di > minus_di and macd > signal > 0:
            # buy signal
            self.order = price
            # logging.warning("Buy Order: {}".format(price))

        if self.order and histogram < prev_histogram:
            diff = price - self.order
            self.budget += diff
            self.order = 0
            self.max_diff = 0
            logging.warning("Sell Order: Budget {} Diff: {}".format(self.budget, diff))

    def test_trading(self):
        df = self.train_data
        df['MA'] = MA(df.Close, timeperiod=14)
        df['MACD'], df['Signal'], df['Histogram'] = MACD(df.Close, 12, 26, 9)
        df['MINUS_DM'] = MINUS_DI(df.High, df.Low, df.Close)
        df['PLUS_DM'] = PLUS_DI(df.High, df.Low, df.Close)
        df['ADX'] = ADX(df.High, df.Low, df.Close)
        data = df.dropna()
        close_price = data.Close.astype('float64').values
        ma_signal = data.MA.values
        macd_data = data.MACD.values
        signal_data = data.Signal.values
        histogram_data = data.Histogram.values
        minus_dm = data.MINUS_DM.values
        plus_dm = data.PLUS_DM.values
        adx_dm = data.ADX.values
        print("Start Price: {} Close Price: {}".format(close_price[0], close_price[-1]))
        idx = 0
        max_diff = 0
        for price, ma, macd, signal, histogram, plus_di, minus_di, adx in zip(close_price, ma_signal, macd_data,
                                                                              signal_data, histogram_data, plus_dm,
                                                                              minus_dm, adx_dm):
            if idx > 2:
                prev_histogram = histogram_data[idx - 1]
                if not self.order and histogram > prev_histogram and plus_di > minus_di and macd > signal > 0:
                    # buy signal
                    self.order = price
                    # logging.warning("Buy Order: {}".format(price))

                if self.order:
                    diff = price - self.order
                    if diff > max_diff:
                        max_diff = diff
                    elif price < max_diff - 5 or diff < -5:
                        diff = price - self.order
                        self.budget += diff
                        self.order = 0
                        max_diff = 0
                        logging.warning("Sell Order: Budget {} Diff: {}".format(self.budget, diff))

                # elif self.order and price - self.order < -3:
                #     diff = price - self.order
                #     self.budget += diff
                #     self.order = 0
                #     logging.warning("Loss: Budget {} Diff: {}".format(self.budget, diff))
                # elif self.order and price - self.order > 50:
                #     diff = price - self.order
                #     self.budget += diff
                #     self.order = 0
                #     logging.warning("Negative Histogram: Budget {} Diff: {}".format(self.budget, diff))

            idx += 1

        print(self.budget)


if __name__ == '__main__':
    bottrading = RegressionMA()
    bottrading.get_data()
    # bottrading.test_trading()
    bottrading.start_socket()
