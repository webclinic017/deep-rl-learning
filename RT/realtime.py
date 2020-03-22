import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from binance.enums import *
from binance.client import Client
from binance.enums import KLINE_INTERVAL_1HOUR
from binance.websockets import BinanceSocketManager
from pymongo import MongoClient

logging.basicConfig(filename='../log/realtime.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class RealTimeAnalytics:
    def __init__(self):
        self.api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        self.api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        self.binace_client = Client(self.api_key, self.api_secret)
        self.bm = BinanceSocketManager(self.binace_client)
        self.client = MongoClient()
        self.db = self.client.crypto

        # env
        self.windows = 500
        self.trading_data = []
        self.delta = []
        self.prev_timestamp = 0
        self.prev_price = 0
        self.budget = 0
        self.order = 0
        self.total_profit = 0
        self.total_loss = 0

        # matplotlib
        # self.fig = plt.figure()
        # self.ax1 = self.fig.add_subplot(211, label='ax1')
        # self.ax2 = self.fig.add_subplot(212, label='ax2')
        # plt.show()
        # self.exp5 = []
        # self.exp6 = []
        # self.exp4 = []
        # self.exp7 = []

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        conn_key = self.bm.start_kline_socket('BTCUSDT', self.process_message, interval=KLINE_INTERVAL_1MINUTE)
        # then start the socket manager
        self.bm.start()

    def process_message(self, msg):
        msg = msg['k']
        current_time = time.time()
        msg['timestamp'] = current_time
        _open = float(msg['o'])
        _high = float(msg['h'])
        _low = float(msg['l'])
        _close = float(msg['c'])
        _timestamp = msg['timestamp']
        _latest = msg['x']

        self.trading_data.append(_close)

        # calculate diff
        diff = (_close - self.prev_price) / (current_time - self.prev_timestamp)
        diff = diff if abs(diff) > 0.03 else 0
        self.prev_timestamp = current_time
        self.prev_price = _close

        self.delta.append(diff)

        moving_df = pd.DataFrame(self.delta, columns=['ma'])
        moving_df = moving_df.ma.rolling(window=14).mean().fillna(0).values

        log_txt = "Price: {} | Budget: {} | Total Loss: {} | Total Profit: {}".format(_close, round(self.budget, 2),
                                                                                      round(self.total_loss, 2),
                                                                                      round(self.total_profit, 2))
        logging.warning(log_txt)

        if len(self.trading_data) >= self.windows:
            self.trading_data.pop(0)
            self.delta.pop(0)
            self.trader(moving_df, self.trading_data)
        insert = self.db.btc1minutes.insert_one(msg).inserted_id

    def trader(self, moving_df, prices):
        raw_df = pd.DataFrame(prices, columns=['c'])
        moving_avg = raw_df.c.rolling(window=14).mean().fillna(prices[0]).values
        # self.ax1.plot(moving_df, label='Moving')
        # self.ax2.plot(prices, label='Raw data')
        price = prices[-1]
        current_moving = moving_df[-1]
        std_moving = moving_df[-5]
        first_moving = moving_df[-7]

        # if self.moving_avg[_idx-75] < self.moving_avg[_idx]:
        if not self.order and first_moving > std_moving < current_moving < -0.05:
            # buy signal
            log_txt = "Buy Order: {}".format(price)
            logging.warning(log_txt)
            # self.exp4.append(len(moving_df))
            # self.exp5.append(price)
            self.order = price

        elif not self.order and abs(current_moving - std_moving) > 0.01 and 0.01 > current_moving > 0 and first_moving < std_moving < current_moving:
            # self.exp4.append(len(moving_df))
            # self.exp5.append(price)
            log_txt = "Buy Order: {}".format(price)
            logging.warning(log_txt)
            self.order = price

        elif self.order and first_moving < std_moving > current_moving >= -0.02 and moving_avg[-70] < moving_avg[-1]:
            # self.exp6.append(len(moving_df))
            # self.exp7.append(price)
            diff = price - self.order
            self.budget += diff
            if diff > 0:
                log_txt = "Take Profit: {} | From {} to {}".format(diff, self.order, price)
                logging.warning(log_txt)
                self.total_profit += diff
            else:
                log_txt = "Loss: {} | From {} to {}".format(diff, self.order, price)
                logging.warning(log_txt)
                self.total_loss += diff

            self.order = 0

        elif self.order and min(self.trading_data[-50:]) > self.order:
            # self.exp6.append(len(self.trading_data))
            # self.exp7.append(price)

            diff = price - self.order
            log_txt = "Stop Loss: {} | From {} to {}".format(diff, self.order, price)
            logging.warning(log_txt)
            self.total_loss += diff
            self.order = 0
            self.budget += diff


if __name__ == '__main__':
    bot = RealTimeAnalytics()
    bot.start_socket()
