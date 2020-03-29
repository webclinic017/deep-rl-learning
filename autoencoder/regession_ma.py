import datetime
import json
import time
import logging
import pandas as pd
import talib
from binance.enums import KLINE_INTERVAL_1MINUTE, KLINE_INTERVAL_1HOUR, KLINE_INTERVAL_5MINUTE, SIDE_SELL, \
    ORDER_TYPE_MARKET, SIDE_BUY
from pymongo import MongoClient
from talib import MA_Type
from talib._ta_lib import RSI, BBANDS, MA, MOM, MACD, DX, MINUS_DM, PLUS_DM, MINUS_DI, PLUS_DI, ADX
from binance.client import Client
from binance.websockets import BinanceSocketManager
import matplotlib.pyplot as plt
from mailer import SendMail

logging.basicConfig(filename='../log/autotrade.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class RegressionMA:
    def __init__(self):
        self.train_data = self.get_data()
        self.global_step = 0
        self.budget = 0
        self.order = 0
        self.prev_histogram = 0
        self.max_diff = 0
        self.take_profit, self.stop_loss = 0, 0
        self.buy_mount = 0
        self.is_latest = False
        self.trade_amount = 0.1  # 10% currency you owned
        self.client = MongoClient()
        self.db = self.client.crypto
        self.api_key = "9Hj6HLNNMGgkqj6ngouMZD1kjIbUb6RZmIpW5HLiZjtDT5gwhXAzc20szOKyQ3HW"
        self.api_secret = "ioD0XICp0cFE99VVql5nuxiCJEb6GK8mh08NYnSYdIUfkiotd1SZqLTQsjFvrXwk"
        self.binace_client = Client(self.api_key, self.api_secret)
        self.bm = BinanceSocketManager(self.binace_client)

        # matplotlib
        self.exp4 = []
        self.exp5 = []
        self.exp6 = []
        self.exp7 = []
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211, label='ax1')
        self.ax2 = self.fig.add_subplot(212, label='ax2')

    def get_data(self):
        api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        binaci_client = Client(api_key, api_secret)
        klines = binaci_client.get_historical_klines("BTCUSDT", KLINE_INTERVAL_5MINUTE, "28 Mar, 2020")
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

    @classmethod
    def fibonacci(cls, price_max, price_min):
        diff = price_max - price_min
        take_profit = price_max + 0.618 * diff
        stop_loss = price_max - 0.382 * diff
        return take_profit, stop_loss

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        conn_key = self.bm.start_kline_socket('BTCUSDT', self.process_message, interval=KLINE_INTERVAL_5MINUTE)
        # then start the socket manager
        self.bm.start()

    def fake_socket(self):
        lines = open("../data/new/btc.csv", "r").read().splitlines()
        delimiter = ','

        for _index, line in enumerate(lines[20000:]):
            line = line.split(delimiter)
            msg = {
                "e": "kline",  # event type
                "E": 1499404907056,  # event time
                "s": "BTCUSDT",  # symbol
                "k": {
                    "t": "",  # start time of this bar
                    "T": "",  # end time of this bar
                    "s": "ETHBTC",  # symbol
                    "i": "1m",  # interval
                    "f": "",  # first trade id
                    "L": "",  # last trade id
                    "o": float(line[0]),  # open
                    "c": float(line[3]),  # close
                    "h": float(line[1]),  # high
                    "l": float(line[2]),  # low
                    "v": "",  # volume
                    "n": 4,  # number of trades
                    "x": True if line[4] == 'true' else False,  # whether this bar is final
                    "q": "1.79662878",  # quote volume
                    "V": "2.34879839",  # volume of active buy
                    "Q": "0.24142166",  # quote volume of active buy
                    "B": "13279784.01349473"  # can be ignored
                }
            }
            self.global_step += 1
            self.process_message(msg)

    def process_message(self, msg):
        msg['k']['timestamp'] = time.time()
        insert = self.db.btc_5minute_realtime.insert_one(msg).inserted_id
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

        if self.is_latest:
            df = pd.DataFrame([[_open_time, _open, _high, _low, _close, _volume, _close_time,
                                _quote_asset_volume, _number_of_trades, _buy_base_asset_volume,
                                _buy_quote_asset_volume, _ignore]],
                              columns=['open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])
            self.train_data = self.train_data.append(df, ignore_index=True, sort=False)
        elif len(self.train_data) > 1:
            # if self.train_data.open_time.values[-1] == msg['k']['t']:
            self.train_data.at[len(self.train_data) - 1, 'Close'] = _close
            self.train_data.at[len(self.train_data) - 1, 'High'] = _high
            self.train_data.at[len(self.train_data) - 1, 'Low'] = _low
        self.is_latest = msg['k']['x']
        self.trading()

    def trading(self):
        df = self.train_data.copy()
        df['MA'] = MA(df.Close, timeperiod=14)
        df['MACD'], df['Signal'], df['Histogram'] = MACD(df.Close, 12, 26, 9)
        df['MINUS_DI'] = MINUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['PLUS_DI'] = PLUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['ADX'] = ADX(df.High, df.Low, df.Close, timeperiod=14)
        df['MA_High'] = MA(df.High, timeperiod=10)
        df['MA_Low'] = MA(df.Low, timeperiod=10)
        data = df.dropna()
        close_price = data.Close.astype('float64').values
        close_p = close_price[-1]
        # ma_c = data.MA_Close.values[-1]
        # macd = data.MACD.values[-1]
        # signal = data.Signal.values[-1]
        histogram_data = data.Histogram.values
        adx = data.ADX.values[-1]
        minus_di = data.MINUS_DI.values[-1]
        plus_di = data.PLUS_DI.values[-1]
        ma_h = data.MA_High.values[-1]
        histogram = histogram_data[-1]
        prev_histogram = histogram_data[-2]
        timestamp = int(time.time())
        low_price = data.Low.values
        high_price = data.High.values
        open_time = data.open_time.values[-1]
        open_time_readable = datetime.datetime.fromtimestamp(open_time/1000).strftime('%Y-%m-%d %H:%M:%S')
        current_time_readable =  datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print("{} | Price: {} | MA: {} | DI-: {} | DI+: {} | Histogram: {}".format(current_time_readable, round(close_p, 2),
                                                                                   round(ma_h, 2), round(minus_di, 2),
                                                                                   round(plus_di, 2), round(histogram, 2)))

        if not self.order and close_p > ma_h and plus_di > minus_di and histogram > prev_histogram and plus_di > 25:
            # buy signal
            self.order = ma_h
            min_price = min(low_price[-10:])
            max_price = max(high_price[-10:])
            self.take_profit, self.stop_loss = self.fibonacci(close_p, min_price)
            logging.warning("{} | Buy Order: MA {} | Price {}".format(open_time_readable, ma_h, close_p))

        elif self.order and close_p >= self.take_profit:
            diff = close_p - self.order
            self.budget += diff
            self.reset()
            logging.warning("{} | Take Profit At {} | Budget {} | Diff {}".format(open_time_readable, close_p, self.budget, diff))

        elif self.order and close_p <= self.stop_loss:
            diff = close_p - self.order
            self.budget += diff
            self.reset()
            logging.warning("{} | Stop loss At {} | Budget {} | Diff {}".format(open_time_readable, close_p, self.budget, diff))

        elif self.order and histogram < 0.5:
            diff = close_p - self.order
            self.budget += diff
            self.reset()
            logging.warning("{} | Histogram close to zero {} | Budget {} | Diff {}".format(open_time_readable, close_p, self.budget, diff))

        elif self.order and prev_histogram - histogram > 0.5:
            diff = close_p - self.order
            self.budget += diff
            self.reset()
            logging.warning("{} | Histogram Down Trend {} | Budget {} | Diff {}".format(open_time_readable, close_p, self.budget, diff))

    def reset(self):
        self.order = 0
        self.max_diff = 0
        self.take_profit = 0
        self.stop_loss = 0

    def test_trading(self):
        df = self.train_data
        df['MA'] = MA(df.Close, timeperiod=14)
        df['MACD'], df['Signal'], df['Histogram'] = MACD(df.Close, 12, 26, 9)
        df['MINUS_DI'] = MINUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['PLUS_DI'] = PLUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['ADX'] = ADX(df.High, df.Low, df.Close, timeperiod=14)
        df['MA_High'] = MA(df.High, timeperiod=10)
        df['MA_Low'] = MA(df.Low, timeperiod=10)
        data = df.dropna()
        close_price = data.Close.astype('float64').values
        high_price = data.High.astype('float64').values
        low_price = data.Low.astype('float64').values
        open_price = data.Open.astype('float64').values
        ma_signal = data.MA.values
        macd_data = data.MACD.values
        signal_data = data.Signal.values
        histogram_data = data.Histogram.values
        minus_dm = data.MINUS_DI.values
        plus_dm = data.PLUS_DI.values
        adx_dm = data.ADX.values
        ma_high = data.MA_High.values
        ma_low = data.MA_Low.values
        open_time_data = data.open_time.values

        print("Start Price: {} Close Price: {}".format(close_price[0], close_price[-1]))
        idx = 0
        for open_p, close_p, ma, macd, signal, histogram, \
            plus_di, minus_di, adx, ma_h, ma_l, open_time in \
                zip(open_price, close_price, ma_signal, macd_data,
                    signal_data, histogram_data, plus_dm,
                    minus_dm, adx_dm, ma_high, ma_low, open_time_data):

            readable = datetime.datetime.fromtimestamp(open_time/1000).strftime('%Y-%m-%d %H:%M:%S')
            if 10 < idx < len(ma_low) - 2:
                prev_histogram = histogram_data[idx-1]
                if not self.order and close_p > ma_h and plus_di > minus_di and histogram > prev_histogram and plus_di > 25:
                    # buy signal
                    self.order = ma_h
                    min_price = min(low_price[idx-10:idx])
                    max_price = max(high_price[idx-10:idx])
                    self.take_profit, self.stop_loss = self.fibonacci(ma_h, min_price)
                    logging.warning("{} | Buy Order: {} Open Price {} Close Price {}".format(readable, ma_h, open_p, close_p))

                elif self.order and close_p >= self.take_profit >= open_p:
                    diff = close_p - self.order
                    self.budget += diff
                    self.reset()
                    logging.warning("{} | Take Profit At {} | Budget {} | Diff {}".format(readable, close_p, self.budget, diff))

                elif self.order and close_p >= self.stop_loss >= open_p:
                    diff = close_p - self.order
                    self.budget += diff
                    self.reset()
                    logging.warning("{} | Stop loss At {} | Budget {} | Diff {}".format(readable, close_p, self.budget, diff))

                elif self.order and histogram < 0.5:
                    diff = close_p - self.order
                    self.budget += diff
                    self.reset()
                    logging.warning("{} | Stop loss At {} | Budget {} | Diff {}".format(readable, close_p, self.budget, diff))

                elif self.order and prev_histogram - histogram > 0.5:
                    diff = close_p - self.order
                    self.budget += diff
                    self.reset()
                    logging.warning("{} | Stop loss At {} | Budget {} | Diff {}".format(readable, close_p, self.budget, diff))

            idx += 1

        print(self.budget)

    def test_order(self):
        info = self.binace_client.get_margin_account()
        get_margin_asset = self.binace_client.get_margin_asset(asset='BTC')

        usdt_amount = info['userAssets'][2]['free']
        details = self.binace_client.get_max_margin_transfer(asset='BTC')
        print("Margin Lever: {}".format(info['marginLevel']))
        print("Amount in BTC: {}".format(info['totalAssetOfBtc']))
        print("Account Status: {}".format(info['tradeEnabled']))

        symbol = 'BTCUSDT'
        self.buy_margin()
        time.sleep(5)
        self.sell_margin()

        print("Test Done!!")

    def buy_margin(self):
        mailer = SendMail()
        try:
            symbol = 'BTCUSDT'
            info = self.binace_client.get_margin_account()
            usdt_amount = info['userAssets'][2]['free']
            price_index = self.binace_client.get_margin_price_index(symbol=symbol)
            amount = int(float(usdt_amount))/float(price_index['price'])
            precision = 5
            amt_str = "{:0.0{}f}".format(amount*self.trade_amount, precision)
            txt = "Buy successfully: Amount: {} Price: {}".format(amt_str, price_index['price'])
            print(txt)
            buy_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str)
            mailer.notification(txt)
            logging.warning(txt)
            self.buy_mount = amt_str
            return True
        except Exception as ex:
            print(ex)
            mailer.notification(str(ex))
            logging.warning(ex)
            return False

    def sell_margin(self):
        mailer = SendMail()
        try:
            info = self.binace_client.get_margin_account()
            symbol = 'BTCUSDT'
            amount = info['totalAssetOfBtc']
            precision = 5
            # amount = round(float(amount)*self.trade_amount, 3)
            # amt_str = "{:0.0{}f}".format(amount, precision)
            amt_str = self.buy_mount
            price_index = self.binace_client.get_margin_price_index(symbol=symbol)
            sell_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str)

            info = self.binace_client.get_margin_account()
            current_btc = info['totalAssetOfBtc']
            usdt = info['userAssets'][2]['free']
            txt = "Sell successfully: Balance: {} Sell Amount: {} In {} Owned In BTC: {}".format(usdt, amt_str,
                                                                                                 price_index['price'],
                                                                                                 current_btc)
            logging.warning(txt)
            print(txt)
            mailer.notification(txt)
            return True
        except Exception as ex:
            print(ex)
            mailer.notification(str(ex))
            logging.warning(ex)
            return False

    def plot_data(self):
        df = pd.read_csv("../data/new/btc.csv")
        df = df[20000:]
        plt.plot(df.Close, label='MA')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    bottrading = RegressionMA()
    # bottrading.plot_data()
    bottrading.get_data()
    bottrading.test_trading()
    # bottrading.start_socket()
    # bottrading.test_order()
    # bottrading.getStockDataVec()
