import datetime
import json
import os
import time
import logging
import numpy as np
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from binance.enums import *
from binance.client import Client
from binance.enums import KLINE_INTERVAL_1HOUR
from binance.websockets import BinanceSocketManager
from pymongo import MongoClient


class AutoTrading(object):
    def __init__(self):
        self.api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        self.api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        # self.binace_client = Client(self.api_key, self.api_secret)
        # self.bm = BinanceSocketManager(self.binace_client)
        # mongodb
        self.client = MongoClient()
        self.db = self.client.crypto
        self.order = 0
        self.budget = 1000
        self.order_type = 'sell'
        self.trading_data = []
        self.norm_data = []
        self.windows = 20
        self.threshold = 0.05
        self.vec_threshold = 0.15
        self.trade_threshold = 90
        self.queue_size = 500
        plt.show(block=False)
        self.order_history = []

    def process_message(self, msg):
        current_time = time.time()
        msg['k']['timestamp'] = current_time
        insert = self.db.btc_data.insert_one(msg['k']).inserted_id

        if len(self.trading_data) > self.queue_size:
            self.trading_data.pop(0)
        if len(self.norm_data) > self.queue_size:
            self.norm_data.pop(0)

        self.trading_data.append(float(msg['k']['c']))

        if len(self.trading_data) > 0:
            # calculate norm data used for plot
            min_x = min(self.trading_data)
            max_x = max(self.trading_data)
            normalized = 0
            if max_x - min_x != 0:
                normalized = 2 * (float(msg['k']['c']) - min_x) / (max_x - min_x)
            self.norm_data.append(normalized)

            self.calculate_macd()

    def norm_list(self, list_needed):
        min_x = min(list_needed)
        max_x = max(list_needed)
        if not min_x or not max_x:
            return list_needed

        return_data = list(map(lambda x: 2 * (x - min_x) / (max_x - min_x), list_needed))
        return return_data

    def calculate_macd(self):
        index = [i for i, val in enumerate(self.trading_data)]
        df = pd.DataFrame({'index': index, 'close': self.trading_data})
        df = df[['close']]
        df.reset_index(level=0, inplace=True)
        df.columns = ['ds', 'y']
        # plt.plot(df.ds, self.norm_data, label='Price')
        # plt.show()

        exp1 = df.y.ewm(span=12, adjust=False).mean()
        exp2 = df.y.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        exp3 = macd.ewm(span=9, adjust=False).mean()

        # calculate break point
        exp4 = []
        exp5 = []
        exp6 = []
        exp7 = []

        t = self.windows
        exp3_cp = self.norm_list(list(exp3.copy()))
        threshold = 0.05
        vec_threshold = 0.07

        for chunk in exp3_cp[self.windows:]:
            d = t - self.windows + 1
            block = exp3_cp[d:t + 1]
            min_el = min(block)
            min_el_idx = block.index(min_el) + t - self.windows
            vec_1 = min_el - block[0]
            vec_2 = block[-1] - min_el
            if abs(vec_1 + vec_2) < self.threshold and (abs(vec_1) > self.vec_threshold or abs(vec_2) > self.vec_threshold):
                if min_el < 0.5:
                    # vector dao chieu
                    # gia se tang len
                    exp4.append(t)
                    exp5.append(self.norm_data[t])
                    self.buy_order(self.trading_data[t])
            t += 1

        t = self.windows
        for chunk in exp3_cp[self.windows:]:
            d = t - self.windows + 1
            block = exp3_cp[d:t + 1]
            max_el = max(block)
            max_el_idx = block.index(max_el) + t - self.windows
            vec_1 = max_el - block[0]
            vec_2 = block[-1] - max_el
            if abs(vec_1 + vec_2) < self.threshold and (abs(vec_1) > self.vec_threshold or abs(vec_2) > self.vec_threshold):
                if max_el > 1.5:
                    # vector dao chieu
                    # gia se giam
                    exp6.append(t)
                    exp7.append(self.norm_data[t])
                    self.sell_order(self.trading_data[t])
            t += 1

        # plt.cla()
        # plt.plot(df.ds, self.norm_data, label='Budget: {}'.format(self.budget))
        # # plt.plot(df.ds, macd, label='AMD MACD', color='#EBD2BE')
        # plt.plot(df.ds, exp3_cp, label='Signal Line', color='#E5A4CB')
        # plt.legend(loc='upper left')
        # plt.plot(exp4, exp5, 'ro', color='blue')
        # plt.plot(exp6, exp7, 'ro', color='red')
        # plt.pause(0.0001)
        print(self.budget)

    def buy_order(self, price):
        order_info = {
            'id': str(uuid.uuid4()),
            'price': price,
            'type': 'buy'
        }
        if len(list(filter(lambda d: d['price'] == price, self.order_history))) == 0:
            if self.order_type == 'sell':
                self.order_history.append(order_info)
                self.order = price
                self.order_type = 'buy'

    def sell_order(self, price):
        order_info = {
            'id': str(uuid.uuid4()),
            'price': price,
            'type': 'sell'
        }
        if len(list(filter(lambda d: d['price'] == price, self.order_history))) == 0:
            self.order_history.append(order_info)
            if self.order_type == 'buy':
                diff = price - self.order
                self.budget += diff
                self.order_type = 'sell'

    def test_order(self):
        """Create a new test order"""
        order = self.binace_client.create_test_order(
            symbol='BTCUSDT',
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=100
        )

    def getStockDataVec(self, key):
        vec = []
        lines = open("data/" + key + ".csv", "r").read().splitlines()
        prices = []
        delimiter = ','
        for _index, line in enumerate(lines[2:]):
            prices.append(float(line.split(delimiter)[0]))

        return vec, prices

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        conn_key = self.bm.start_kline_socket('BTCUSDT', self.process_message, interval=KLINE_INTERVAL_1HOUR)
        # then start the socket manager
        self.bm.start()


if __name__ == '__main__':
    trading_bot = AutoTrading()
    vec_data, price_data = trading_bot.getStockDataVec('train')
    for item in price_data:
        msg = {
            'k': {
                'c': item
            }
        }
        trading_bot.process_message(msg)
    # trading_bot.start_socket()
