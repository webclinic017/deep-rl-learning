import datetime
import time
import json
import logging
import pandas as pd
from binance.enums import KLINE_INTERVAL_1HOUR, SIDE_SELL, ORDER_TYPE_MARKET, SIDE_BUY
# from binance.websockets import BinanceSocketManager
from pymongo import MongoClient
from talib._ta_lib import MA, MACD, MINUS_DI, PLUS_DI, ADX
from binance.client import Client
import matplotlib.pyplot as plt
from mailer import SendMail

logging.basicConfig(filename='log/autotrade.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class RegressionMA:
    def __init__(self):
        self.global_step = 0
        self.order = 0
        self.buy_mount = 0
        self.side = None  # can order
        with open("config.txt", "r") as file:
            order_data = file.readline()
            order_data = order_data.split(',')
            if float(order_data[0]) != 0:
                self.order = float(order_data[0])
                self.buy_mount = float(order_data[1])
        self.budget = 0
        self.prev_histogram = 0
        self.max_profit = 0
        self.take_profit, self.stop_loss = 0, 0
        self.is_latest = False
        self.trade_amount = 0.5  # 50% currency you owned
        self.client = MongoClient(host='149.28.129.64', username='crypto', password='amyAFpNK', authSource='crypto')
        self.db = self.client.crypto
        # self.db.btc_5minute_realtime.drop()
        self.api_key = "9Hj6HLNNMGgkqj6ngouMZD1kjIbUb6RZmIpW5HLiZjtDT5gwhXAzc20szOKyQ3HW"
        self.api_secret = "ioD0XICp0cFE99VVql5nuxiCJEb6GK8mh08NYnSYdIUfkiotd1SZqLTQsjFvrXwk"
        self.binace_client = Client(self.api_key, self.api_secret)
        # self.bm = BinanceSocketManager(self.binace_client)
        self.interval = KLINE_INTERVAL_1HOUR
        self.train_data = self.get_data()

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
        klines = binaci_client.get_historical_klines("BTCUSDT", self.interval, "1 April, 2020")
        df = pd.DataFrame(klines, columns=['open_time', 'Open', 'High', 'Low', 'Close',
                                           'Volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])
        return df

    @classmethod
    def fibonacci(cls, price_max, price_min):
        diff = price_max - price_min
        take_profit = price_max + 0.618 * diff
        stop_loss = price_max - 0.382 * diff
        return take_profit, stop_loss

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        conn_key = self.bm.start_kline_socket('BTCUSDT', self.process_message, interval=self.interval)
        # then start the socket manager
        self.bm.start()

    def fake_socket(self):
        # data = self.db.btc_1h.find({}, {'_id': 0})
        # data = list(data)
        # with open('btc_1h.json', 'w') as outfile:
        #     json.dump(data, outfile, indent=4)
        with open('btc_1h.json') as json_file:
            data = json.load(json_file)
            for msg in data:
                self.global_step += 1
                self.process_message(msg)
        json_file.close()

    def process_message(self, msg):
        if 'timestamp' not in msg['k']:
            msg['k']['timestamp'] = time.time()

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
        _timestamp = msg['k']['timestamp']

        if self.is_latest:
            df = pd.DataFrame([[_open_time, _open, _high, _low, _close, _volume, _close_time,
                                _quote_asset_volume, _number_of_trades, _buy_base_asset_volume,
                                _buy_quote_asset_volume, _ignore]],
                              columns=['open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])
            self.train_data = self.train_data.append(df, ignore_index=True, sort=False)

        elif len(self.train_data) > 1:
            self.train_data.at[len(self.train_data) - 1, 'Close'] = _close
            self.train_data.at[len(self.train_data) - 1, 'High'] = _high
            self.train_data.at[len(self.train_data) - 1, 'Low'] = _low
        self.is_latest = msg['k']['x']

        if len(self.train_data) > 50:
            self.trading(float(_close), _timestamp)

    def trading(self, close_p, _timestamp):
        df = self.train_data.copy()
        df['MA'] = MA(df.Close, timeperiod=14)
        df['MACD'], df['Signal'], df['Histogram'] = MACD(df.Close, 12, 26, 9)
        df['MINUS_DI'] = MINUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['PLUS_DI'] = PLUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['ADX'] = ADX(df.High, df.Low, df.Close, timeperiod=14)
        df['MA_High'] = MA(df.High, timeperiod=10)
        df['MA_Low'] = MA(df.Low, timeperiod=10)
        data = df.dropna()
        # close_price = data.Close.astype('float64').values
        # close_p = close_price[-1]
        # ma_c = data.MA_Close.values[-1]
        # macd = data.MACD.values[-1]
        # signal = data.Signal.values[-1]
        histogram_data = data.Histogram.values
        plus_di_data = data.PLUS_DI.values

        adx_data = data.ADX.values
        adx = adx_data[-1]
        minus_di = data.MINUS_DI.values[-1]
        plus_di = plus_di_data[-1]
        ma_h = data.MA_High.values[-1]
        histogram = histogram_data[-1]
        # prev_histogram = histogram_data[-2]
        prev_adx = adx_data[-2]
        prev_prev_adx = adx_data[-3]
        # prev_plus_di = plus_di_data[-2]
        low_price = data.Low.astype('float64').values
        high_price = data.High.astype('float64').values
        open_time = data.open_time.values[-1]
        open_time_readable = datetime.datetime.fromtimestamp(open_time/1000).strftime('%Y-%m-%d %H:%M:%S')
        current_time_readable = datetime.datetime.fromtimestamp(_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print("{} | Price {} | DI- {} | DI+ {} | ADX {}".format(current_time_readable, round(close_p, 2),
                                                                round(minus_di, 2),
                                                                round(plus_di, 2), round(adx, 2)))

        # if not self.order and \
        #         close_p > ma_h and \
        #         plus_di > minus_di and \
        #         histogram > prev_histogram and \
        #         plus_di > 25 and \
        #         adx > prev_adx and \
        #         histogram > 0 and \
        #         adx > 25 and \
        #         plus_di > prev_plus_di:

        if not self.order and \
                plus_di > minus_di and \
                adx > 25:
            # buy signal
            self.buy_margin()
            self.side = 'buy'
            self.order = close_p
            min_price = min(low_price[-10:])
            max_price = max(high_price[-10:])
            self.take_profit, self.stop_loss = self.fibonacci(max_price, min_price)
            logging.warning("{} | Buy Order | Price {} | ADX {} | Stop Loss {}".format(current_time_readable,
                                                                                      round(close_p, 2),
                                                                                      round(adx, 2),
                                                                                      round(self.stop_loss, 2)))

        # elif not self.order and \
        #         plus_di < minus_di and \
        #         adx > 25:
        #     # buy signal
        #     # self.buy_margin()
        #     self.side = 'sell'
        #     self.order = close_p
        #     min_price = min(low_price[-10:])
        #     max_price = max(high_price[-10:])
        #     self.take_profit, self.stop_loss = self.fibonacci(max_price, min_price)
        #     logging.warning("{} | Sell Order | Price {} | ADX {} | Stop Loss {}".format(current_time_readable,
        #                                                                               round(close_p, 2),
        #                                                                               round(adx, 2),
        #                                                                               round(self.stop_loss, 2)))

        elif self.side == 'buy' and self.order and (adx < 25 or plus_di < minus_di):
            # take profit
            self.sell_margin()
            diff = close_p - self.order
            self.budget += diff
            self.side = None
            self.reset()
            logging.warning("{} | Close Order At {} | Budget {} | Diff {}".format(current_time_readable, round(close_p, 2),
                                                                                  round(self.budget, 2), round(diff, 2)))

        # elif self.side == 'sell' and self.order and (adx < 25 or plus_di > minus_di):
        #     # take profit
        #     # self.sell_margin()
        #     diff = self.order - close_p
        #     self.budget += diff
        #     self.side = None
        #     self.reset()
        #     logging.warning("{} | Close Order At {} | Budget {} | Diff {}".format(current_time_readable, round(close_p, 2),
        #                                                                           round(self.budget, 2), round(diff, 2)))

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
                prev_adx = adx_dm[idx-1]
                prev_plus_di = plus_dm[idx-1]
                if not self.order and \
                        plus_di > minus_di and \
                        adx > 25:

                    # buy signal
                    self.order = close_p
                    min_price = min(low_price[idx-10:idx])
                    max_price = max(high_price[idx-10:idx])
                    self.take_profit, _ = self.fibonacci(max_price, min_price)
                    self.stop_loss = close_p - 50
                    logging.warning("{} | Buy Order: {} | Close Price {}".format(readable, ma_h, close_p))

                elif self.order and (plus_di < minus_di or adx < 25):
                    # take profit
                    diff = close_p - self.order
                    self.budget += diff
                    self.reset()
                    logging.warning(
                        "{} | Close Order At {} | Budget {} | Diff {}".format(readable, round(close_p, 2),
                                                                              round(self.budget, 2), round(diff, 2)))

                elif self.order and close_p <= self.stop_loss:
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
        # self.buy_margin()
        # time.sleep(5)
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
            txt = "Buy successfully | Amount {} | Price {}".format(amt_str, price_index['price'])
            print(txt)
            buy_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str)
            mailer.notification(txt)
            logging.warning(txt)
            self.buy_mount = amt_str
            with open("config.txt", "w") as file:
                file.write("{},{}".format(price_index['price'], amt_str))
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
            txt = "Sell successfully | Balance {} | Sell Amount {} | At Price {} | Owned In BTC {}".format(usdt, amt_str,
                                                                                                           price_index['price'],
                                                                                                           current_btc)
            logging.warning(txt)
            print(txt)
            mailer.notification(txt)
            with open("config.txt", "w") as file:
                file.write("{},{}".format(0, 0))
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
    # bottrading.test_trading()
    # bottrading.fake_socket()
    bottrading.start_socket()
    # bottrading.test_order()
    # bottrading.getStockDataVec()
