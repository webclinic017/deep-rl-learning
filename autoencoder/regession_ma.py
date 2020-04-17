import datetime
import time
import json
import logging
import pandas as pd
import numpy as np
from binance.enums import KLINE_INTERVAL_1HOUR, SIDE_SELL, ORDER_TYPE_MARKET, SIDE_BUY
from binance.websockets import BinanceSocketManager
# from pymongo import MongoClient
from talib._ta_lib import MA, MACD, MINUS_DI, PLUS_DI, ADX, BBANDS, MA_Type, STOCH
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
        self.trade_amount = 0.5  # 75% currency you owned
        self.api_key = "9Hj6HLNNMGgkqj6ngouMZD1kjIbUb6RZmIpW5HLiZjtDT5gwhXAzc20szOKyQ3HW"
        self.api_secret = "ioD0XICp0cFE99VVql5nuxiCJEb6GK8mh08NYnSYdIUfkiotd1SZqLTQsjFvrXwk"
        self.binace_client = Client(self.api_key, self.api_secret)
        self.bm = BinanceSocketManager(self.binace_client)
        self.interval = KLINE_INTERVAL_1HOUR
        self.train_data = self.get_data()

        # Matplotlib
        self.exp4 = []
        self.exp5 = []
        self.exp6 = []
        self.exp7 = []
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211, label='ax1')
        self.ax2 = self.fig.add_subplot(212, label='ax2')

        # Global Config
        self.bbw_threshold = 0.03
        self.adx_threshold = 25

        # Email Services
        self.mailer = SendMail()

    def get_data(self):
        api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        binaci_client = Client(api_key, api_secret)
        klines = binaci_client.get_historical_klines("BTCUSDT", self.interval, "15 April, 2020")
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
            df = pd.DataFrame(
                [
                    [
                        _open_time, _open, _high, _low, _close, _volume, _close_time,
                        _quote_asset_volume, _number_of_trades, _buy_base_asset_volume,
                        _buy_quote_asset_volume, _ignore
                    ]
                ],
                columns=[
                    'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'
                ]
            )
            self.train_data = self.train_data.append(df, ignore_index=True, sort=False)

        elif len(self.train_data) > 1:
            self.train_data.at[len(self.train_data) - 1, 'Close'] = _close
            self.train_data.at[len(self.train_data) - 1, 'High'] = _high
            self.train_data.at[len(self.train_data) - 1, 'Low'] = _low

        self.is_latest = msg['k']['x']
        if len(self.train_data) > 1:
            self.trading(float(_close), _timestamp)

    def trading(self, close_p, _timestamp):
        df = self.train_data.copy()
        df['MINUS_DI'] = MINUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['PLUS_DI'] = PLUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['ADX'] = ADX(df.High, df.Low, df.Close, timeperiod=14)
        df['MA_High'] = MA(df.High, timeperiod=9)
        df['MA_Low'] = MA(df.Low, timeperiod=9)
        df['BAND_UPPER'], df['BAND_MIDDLE'], df['BAND_LOWER'] = BBANDS(df.Close, 20, 2, 2)
        df['slowk'], df['slowd'] = STOCH(df.High, df.Low, df.Close)
        data = df.dropna()

        # Bollinger band
        middle_band = data.BAND_MIDDLE.values[-1]
        lower_band = data.BAND_LOWER.values[-1]
        upper_band = data.BAND_UPPER.values[-1]

        # Bollinger band width
        bb_w = (upper_band - lower_band) / middle_band

        # Stochastic
        slowk = df.slowk.values[-1]
        slowd = df.slowd.values[-1]

        # ADX DMI
        plus_di_data = data.PLUS_DI.values
        minus_di_data = data.MINUS_DI.values
        adx = data.ADX.values[-1]
        minus_di = minus_di_data[-1]
        prev_minus_di = minus_di_data[-2]
        plus_di = plus_di_data[-1]
        prev_plus_di = plus_di_data[-2]
        current_time_readable = datetime.datetime.fromtimestamp(_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print("{} | Price {} | DI- {} | DI+ {} | ADX {} | BBW: {}".format(
            current_time_readable,
            round(close_p, 2),
            round(minus_di, 2),
            round(plus_di, 2),
            round(adx, 2),
            round(bb_w, 4)
        )
        )

        # Place Buy Order
        if not self.order and \
                plus_di > minus_di and \
                adx > self.adx_threshold and \
                plus_di > prev_plus_di and \
                close_p > middle_band and \
                slowk > slowd and \
                bb_w > self.bbw_threshold:
            # buy signal
            self.buy_margin()
            self.side = 'buy'
            self.order = close_p
            logging.warning("{} | Buy Order | Price {} | ADX {} | %K: {} | %D {}".format(current_time_readable,
                                                                                         round(close_p, 2),
                                                                                         round(adx, 2),
                                                                                         round(slowk, 2),
                                                                                         round(slowd, 2)))
        # CLose Buy Order
        elif self.side == 'buy' and self.order and \
                (close_p < middle_band or (slowd > slowk > 80)):
            # take profit
            self.close_buy_margin()
            diff = close_p - self.order
            self.budget += diff
            self.reset()
            logging.warning("{} | Close Buy Order At {} | Budget {} | Diff {}".format(current_time_readable,
                                                                                      round(close_p, 2),
                                                                                      round(self.budget, 2),
                                                                                      round(diff, 2)))

        # Place Sell Order
        elif not self.order and \
                minus_di > plus_di and \
                adx > self.adx_threshold and \
                minus_di > prev_minus_di and \
                close_p < middle_band and \
                slowk < slowd and \
                bb_w > self.bbw_threshold:

            self.side = 'sell'
            self.order = close_p
            txt = "{} | Sell Order | Price {} | ADX {} | Stop Loss {}".format(
                current_time_readable,
                round(close_p, 2),
                round(adx, 2),
                round(self.stop_loss, 2)
            )
            self.borrow_btc()
            self.mailer.notification(txt)
            logging.warning(txt)

        # Close Sell Order
        elif self.side == 'sell' and self.order and \
                (close_p > middle_band or 20 > slowk > slowd):
            diff = self.order - close_p
            self.budget += diff
            self.side = None
            self.reset()
            txt = "{} | Close Sell Order At {} | Budget {} | Diff {}".format(
                current_time_readable, round(close_p, 2),
                round(self.budget, 2), round(diff, 2)
            )
            self.repay_btc()
            logging.warning(txt)
            self.mailer.notification(txt)

    def reset(self):
        self.order = 0
        self.side = None
        self.take_profit = 0
        self.stop_loss = 0

    def test_buy_order(self):
        info = self.binace_client.get_margin_account()
        get_margin_asset = self.binace_client.get_margin_asset(asset='BTC')

        usdt_amount = info['userAssets'][2]['free']
        details = self.binace_client.get_max_margin_transfer(asset='BTC')
        print("Margin Lever: {}".format(info['marginLevel']))
        print("Amount in BTC: {}".format(info['totalAssetOfBtc']))
        print("Account Status: {}".format(info['tradeEnabled']))

        self.buy_margin()
        time.sleep(1)
        self.close_buy_margin()

        print("Test Done!!")

    def test_sell_order(self):
        self.borrow_btc()
        time.sleep(1)
        self.repay_btc()
        info = self.binace_client.get_margin_account()
        print("Margin Lever: {}".format(info['marginLevel']))
        print("Amount in BTC: {}".format(info['totalAssetOfBtc']))
        print("Account Status: {}".format(info['tradeEnabled']))
        print("Test Done!!")

    def borrow_btc(self):
        """
        Borrow BTC with amount = 0.5 * max amount
        :return:
        """
        try:
            # Borrow BTC
            get_max_margin_loan = self.binace_client.get_max_margin_loan(asset='BTC')
            amount = float(get_max_margin_loan['amount'])
            precision = 5
            amt_str = "{:0.0{}f}".format(amount * self.trade_amount, precision)
            transaction = self.binace_client.create_margin_loan(asset='BTC', amount=amt_str)

            # Sell borrowed BTC
            symbol = 'BTCUSDT'
            sell_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str
            )
            txt = 'You borrow: {}'.format(amt_str)
            logging.warning(txt)
            print(txt)
        except Exception as ex:
            self.mailer.notification(str(ex))

    def repay_btc(self):
        """
        Trả lại số BTC đã vay bằng cách mua BTC sau đó mở lệnh repay
        :return:
        """
        try:
            # check amount you borrowed
            symbol = 'BTCUSDT'
            symbol_detail = 'BTC'
            # check account info
            info = self.binace_client.get_margin_account()
            for market in info['userAssets']:
                if market['asset'] == symbol_detail:
                    btc_borrowed = market['borrowed']
                    precision = 5
                    amt_str = "{:0.0{}f}".format(float(btc_borrowed), precision)

                    # # buy btc
                    buy_order = self.binace_client.create_margin_order(
                        symbol=symbol,
                        side=SIDE_BUY,
                        type=ORDER_TYPE_MARKET,
                        quantity=amt_str
                    )

            info = self.binace_client.get_margin_account()
            for market in info['userAssets']:
                if market['asset'] == symbol_detail:
                    btc_free = market['free']
                    precision = 5
                    amt_str = "{:0.0{}f}".format(float(btc_free), precision)
                    # repay
                    transaction = self.binace_client.repay_margin_loan(asset='BTC', amount=btc_free)
                    txt = 'You repay : {} {}'.format(amt_str, symbol_detail)
                    logging.warning(txt)
                    print(txt)
        except Exception as ex:
            self.mailer.notification(str(ex))

    def buy_margin(self):
        try:
            symbol = 'BTCUSDT'
            info = self.binace_client.get_margin_account()
            usdt_amount = info['userAssets'][2]['free']
            price_index = self.binace_client.get_margin_price_index(symbol=symbol)
            amount = int(float(usdt_amount)) / float(price_index['price'])
            precision = 5
            amt_str = "{:0.0{}f}".format(amount * self.trade_amount, precision)
            txt = "Buy successfully | Amount {} | Price {}".format(amt_str, price_index['price'])
            print(txt)
            buy_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str)
            self.mailer.notification(txt)
            logging.warning(txt)
            self.buy_mount = amt_str
            with open("config.txt", "w") as file:
                file.write("{},{}".format(price_index['price'], amt_str))
            return True
        except Exception as ex:
            print(ex)
            self.mailer.notification(str(ex))
            logging.warning(ex)
            return False

    def close_buy_margin(self):
        try:
            symbol = 'BTCUSDT'
            symbol_detail = 'BTC'
            amt_str = 0
            # check account info
            info = self.binace_client.get_margin_account()

            for market in info['userAssets']:
                if market['asset'] == symbol_detail:
                    btc_free = market['free']
                    precision = 5
                    # amt_str = btc_free
                    amt_str = "{:0.0{}f}".format(float(btc_free) - 0.0001, precision)

                    # Sell btc
                    buy_order = self.binace_client.create_margin_order(
                        symbol=symbol,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_MARKET,
                        quantity=amt_str
                    )

            price_index = self.binace_client.get_margin_price_index(symbol=symbol)
            info = self.binace_client.get_margin_account()
            current_btc = info['totalAssetOfBtc']
            usdt = info['userAssets'][2]['free']
            txt = "Sell successfully | Balance {} | Sell Amount {} | At Price {} | Owned In BTC {}".format(
                usdt,
                amt_str,
                price_index['price'],
                current_btc
            )
            logging.warning(txt)
            print(txt)
            self.mailer.notification(txt)
            with open("config.txt", "w") as file:
                file.write("{},{}".format(0, 0))
            return True
        except Exception as ex:
            print(ex)
            self.mailer.notification(str(ex))
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
    # bottrading.test_buy_order()
    # bottrading.test_sell_order()
