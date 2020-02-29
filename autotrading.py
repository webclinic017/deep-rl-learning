import math
import time
import logging
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from binance.enums import *
from binance.client import Client
from binance.enums import KLINE_INTERVAL_1HOUR
from binance.websockets import BinanceSocketManager
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np

logging.basicConfig(filename='log/autotrade.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class AutoTrading(object):
    def __init__(self):
        self.api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        self.api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        self.binace_client = Client(self.api_key, self.api_secret)
        self.bm = BinanceSocketManager(self.binace_client)
        # mongodb
        self.client = MongoClient()
        self.db = self.client.crypto
        self.order = 0
        self.budget = 1000
        self.order_type = 'sell'
        self.trading_data = []
        self.indexes = []
        self.norm_data = []
        self.windows = 30
        self.threshold = 0.05
        self.vec_threshold = 0.15
        self.trade_threshold = 90
        self.queue_size = 500
        plt.show(block=False)
        self.order_history = []
        self.tqdm_e = None
        self.prev_d = 0
        self.waiting_for_order = False

    def process_message(self, msg):
        current_time = time.time()
        msg['k']['timestamp'] = current_time
        # print(msg)
        insert = self.db.btc_data.insert_one(msg['k']).inserted_id

        if len(self.trading_data) > self.queue_size:
            self.trading_data.pop(0)
        if len(self.norm_data) > self.queue_size:
            self.norm_data.pop(0)

        self.trading_data.append(float(msg['k']['c']))
        # self.indexes.append(float(msg['k']['indexes']))

        # calculate norm data used for plot
        min_x = min(self.trading_data)
        max_x = max(self.trading_data)
        normalized = 0
        if max_x - min_x != 0:
            normalized = 20 * (float(msg['k']['c']) - min_x) / (max_x - min_x)
        self.norm_data.append(normalized)

        self.calculate_macd()

    def norm_list(self, list_needed):
        min_x = min(list_needed)
        max_x = max(list_needed)
        if not min_x or not max_x:
            return list_needed

        return_data = list(map(lambda x: 20 * (x - min_x) / (max_x - min_x), list_needed))
        return return_data

    @staticmethod
    def angle_of_vectors(x, x0, y, y0):
        dotProduct = x - x0
        # for three dimensional simply add dotProduct = a*c + b*d  + e*f
        modOfVector1 = math.sqrt(pow(x-x0, 2) + pow(y-y0, 2))
        # for three dimensional simply add modOfVector = math.sqrt( a*a + b*b + e*e)*math.sqrt(c*c + d*d +f*f)
        angle = min(dotProduct / modOfVector1, 1)
        # print("Cosθ =", angle)
        angleInDegree = math.degrees(math.acos(angle))
        # print("θ =", angleInDegree, "°")
        return angleInDegree

    def calculate_macd(self):
        if len(self.trading_data) > self.windows:
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

            exp3_cp = self.norm_list(list(exp3.copy()))
            threshold = 0.05
            vec_threshold = 0.07

            block = exp3_cp[self.prev_d:]
            chunk_idx = self.prev_d + self.windows//2
            angel_degree_1 = self.angle_of_vectors(x=(chunk_idx + (self.windows//2)), x0=chunk_idx,
                                                   y=block[self.windows//2], y0=block[0])
            angel_degree_2 = self.angle_of_vectors(x=len(exp3_cp), x0=(chunk_idx + (self.windows//2)),
                                                   y=block[-1], y0=block[self.windows//2])

            if block[self.windows//2] - block[0] < 0:
                # trend down
                angel_degree_1 = -angel_degree_1
            if block[-1] - block[self.windows//2] < 0:
                # trend down
                angel_degree_2 = -angel_degree_2

            # print(angel_degree_1, angel_degree_2)

            if 0 < angel_degree_1 < 5:
                # bắt đầu ngừng tăng hoặc giảm.
                # sell
                if not self.waiting_for_order:
                    self.sell_order(self.trading_data[-1])

                    # exp6.append(chunk_idx)
                    # exp7.append(exp3_cp[chunk_idx])
                    self.waiting_for_order = True

            if angel_degree_2 < -15:
                # tín hiệu bắt đầu giảm
                pass

            if angel_degree_2 > 15:
                # tín hiệu bắt đầu tăng
                if self.waiting_for_order:
                    self.buy_order(self.trading_data[-1])
                    # exp4.append(chunk_idx)
                    # exp5.append(exp3_cp[chunk_idx])
                    self.waiting_for_order = False

            if len(self.trading_data) < self.queue_size:
                self.prev_d += 1

            # plt.cla()
            # plt.plot(df.ds, self.norm_data, label='Budget: {}, {}'.format(angel_degree_1, angel_degree_2))
            # # plt.plot(df.ds, macd, label='AMD MACD', color='#EBD2BE')
            # plt.plot(df.ds, exp3_cp, label='Signal Line {}'.format(self.budget), color='#E5A4CB')
            # plt.legend(loc='upper left')
            # plt.plot(exp4, exp5, 'ro', color='blue')
            # plt.plot(exp6, exp7, 'ro', color='red')
            # plt.pause(0.00001)
            # self.tqdm_e.set_description("Profit: " + str(self.budget))
            # self.tqdm_e.refresh()

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
                logging.warning("Close order: {} => {} profit {} budget: {}".format(self.order, price,
                                                                                    round(diff, 2), self.budget))
                self.budget = self.budget + round(diff, 2)
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
        indexes = []
        lines = open("data/" + key + ".csv", "r").read().splitlines()
        prices = []
        delimiter = ','
        first_index = float(lines[2].split(delimiter)[1])
        for _index, line in enumerate(lines[2:]):
            prices.append(float(line.split(delimiter)[0]))
            indexes.append(float(line.split(delimiter)[1]) - first_index)

        return indexes, prices

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        conn_key = self.bm.start_kline_socket('BTCUSDT', self.process_message, interval=KLINE_INTERVAL_1HOUR)
        # then start the socket manager
        self.bm.start()

    def start_mockup(self):
        indexes, price_data = trading_bot.getStockDataVec('train')
        self.tqdm_e = tqdm(price_data, desc='Steps', leave=True, unit=" episodes")
        for item in self.tqdm_e:
            msg = {
                'k': {
                    'c': item
                }
            }
            trading_bot.process_message(msg)


if __name__ == '__main__':
    trading_bot = AutoTrading()
    # trading_bot.start_mockup()
    trading_bot.start_socket()
