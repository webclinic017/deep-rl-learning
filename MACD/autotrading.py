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

logging.basicConfig(filename='../log/autotrade.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class AutoTrading(object):
    def __init__(self):
        self.actions, self.states, self.rewards = [], [], []
        self.api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        self.api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        self.binace_client = Client(self.api_key, self.api_secret)
        self.bm = BinanceSocketManager(self.binace_client)
        # mongodb
        self.client = MongoClient()
        self.db = self.client.crypto
        self.order = 0
        self.budget = 0
        self.order_type = 'sell'
        self.trading_data = []
        self.indexes = []
        self.norm_data = []
        self.windows = 10
        self.threshold = 0.05
        self.vec_threshold = 0.15
        self.trade_threshold = 90
        self.queue_size = 32
        self.order_history = []
        self.tqdm_e = None
        self.prev_d = 0
        self.waiting_for_order = True
        self.stop_loss = 0
        self.take_profit = 0
        self.waiting_time = 0
        self.price_max = 0
        self.price_min = 0
        self.stop_loss_nb = 0
        self.take_profit_nb = 0
        self.prev_max_price = 0
        self.prev_min_price = 100000
        self.total_lost = 0
        self.total_profit = 0
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211, label='ax1')
        self.ax2 = self.fig.add_subplot(212, label='ax2')

        # Environment
        self.time = 0
        self.cumul_reward = 0
        self.done = False
        self.old_state = np.array([0, 0])

        # plot point data
        self.exp4 = []
        self.exp5 = []
        self.exp6 = []
        self.exp7 = []

    def reset(self):
        self.time = 0
        self.cumul_reward = 0
        self.done = False
        self.budget = 1000

    def process_message(self, msg):
        current_time = time.time()
        msg['k']['timestamp'] = current_time
        # print(msg)
        # insert = self.db.btc_data.insert_one(msg['k']).inserted_id

        if len(self.trading_data) > self.queue_size:
            self.trading_data.pop(0)
        if len(self.norm_data) > self.queue_size:
            self.norm_data.pop(0)

        self.trading_data.append(float(msg['k']['c']))
        # calculate norm data used for plot
        # min_x = min(self.trading_data)
        # max_x = max(self.trading_data)
        # normalized = 0
        # if max_x - min_x != 0:
        #     normalized = 20 * (float(msg['k']['c']) - min_x) / (max_x - min_x)
        # self.norm_data.append(normalized)

        if len(self.trading_data) > self.windows:
            self.calculate_macd()

    def norm_list(self, list_needed):
        """Scale feature to 0-20"""
        min_x = min(list_needed)
        max_x = max(list_needed)
        if not min_x or not max_x:
            return list_needed

        return_data = list(map(lambda x: (x - min_x) / (max_x - min_x), list_needed))
        return return_data

    @staticmethod
    def angle_of_vectors(x, x0, y, y0):
        """
        Calculate angel of the two vector a = (x, y) b = (x0, y0)
        :param x:
        :param x0:
        :param y:
        :param y0:
        :return: angel
        """
        dotProduct = math.sqrt(pow(x - x0, 2))
        # for three dimensional simply add dotProduct = a*c + b*d  + e*f
        modOfVector1 = math.sqrt(pow(x - x0, 2) + pow(y - y0, 2))
        # for three dimensional simply add modOfVector = math.sqrt( a*a + b*b + e*e)*math.sqrt(c*c + d*d +f*f)
        angle = min(dotProduct / modOfVector1, 1)
        # print("Cosθ =", angle)
        angleInDegree = math.degrees(math.acos(angle))
        # print("θ =", angleInDegree, "°")
        return angleInDegree

    def calculate_macd(self):
        """
        Simulation trading and RL learning
        :return:
        """
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
        histogram = macd - exp3
        # macd = self.norm_list(list(macd.copy()))
        if len(self.trading_data) == self.queue_size + 1:
            exp4 = self.exp4.copy()
            for x4 in exp4:
                if x4 < 0:
                    self.exp4.pop(0)
                    self.exp5.pop(0)

            exp6 = self.exp6.copy()
            for x6 in exp6:
                if x6 < 0:
                    self.exp6.pop(0)
                    self.exp7.pop(0)

            self.exp4 = [x1 - 1 for x1 in self.exp4]
            self.exp6 = [x - 1 for x in self.exp6]

        histogram_cp = list(histogram)
        start_point = len(histogram_cp) - self.windows - 1
        end_point = len(histogram_cp) - 1
        block = histogram_cp[start_point:]
        chunk_idx = len(histogram_cp) - self.windows // 2 - 1
        angel_degree_1 = self.angle_of_vectors(x=chunk_idx, x0=start_point,
                                               y=histogram_cp[chunk_idx], y0=histogram_cp[start_point])
        angel_degree_2 = self.angle_of_vectors(x=end_point, x0=chunk_idx,
                                               y=histogram_cp[end_point], y0=histogram_cp[chunk_idx])

        if block[self.windows // 2] < block[0]:
            # trend down
            angel_degree_1 = - angel_degree_1
        if block[-1] < block[self.windows // 2]:
            # trend down
            angel_degree_2 = - angel_degree_2

        min_histogram = min(histogram_cp)
        max_histogram = max(histogram_cp)
        if len(self.trading_data) > self.queue_size:
            # if not self.check_profit(self.trading_data[-1]) and not self.check_lost(self.trading_data[-1]):
            #     if abs(histogram_cp[-1]) < max_histogram * 0.1 and histogram_cp[-3] < 0 and self.waiting_for_order:
            #         if abs(max_histogram) > abs(min_histogram):
            #             # buy order
            #             # up trend
            #             self.exp4.append(len(histogram_cp))
            #             self.exp5.append(histogram_cp[-1])
            #             self.buy_order(self.trading_data[-1])
            #     elif abs(histogram_cp[-1]) < abs(min_histogram) * 0.1 and histogram_cp[-3] > 0 and not self.waiting_for_order:
            #         # stop loss
            #         self.exp6.append(len(histogram_cp))
            #         self.exp7.append(histogram_cp[-1])
            #         self.sell_order(self.trading_data[-1])
            #     elif histogram_cp[-1] >= max_histogram*0.4 and not self.waiting_for_order:
            #         # sell
            #         self.exp6.append(len(histogram_cp))
            #         self.exp7.append(histogram_cp[-1])
            #         self.sell_order(self.trading_data[-1])

            # if not self.waiting_for_order:
            #     self.waiting_time += 1
            if histogram_cp[-1] > 0 and histogram_cp[-3] < 0:
                self.buy_order(self.trading_data[-1])
            elif histogram_cp[-1] < 0 and histogram_cp[-3] > 0:
                self.sell_order(self.trading_data[-1])

        # print(avg_histogram, self.waiting_for_order)

        # # new ways
        # macd_cp = list(macd.copy())
        # exp3_cp = list(exp3.copy())
        # signal_value = abs(macd_cp[-3]) - abs(exp3_cp[-3])
        # if signal_value <= 0.02:
        #     if len(self.trading_data) > self.queue_size:
        #         if not self.check_profit(self.trading_data[-1]):
        #             if not self.check_lost(self.trading_data[-1]):
        #                 if macd_cp[-2] > exp3_cp[-2]:
        #                     # up trend
        #                     if self.waiting_for_order:
        #                         self.exp4.append(len(exp3_cp))
        #                         self.exp5.append(exp3_cp[-1])
        #                         self.buy_order(self.trading_data[-1])
        #                         self.waiting_for_order = False
        #                 if macd_cp[-2] < exp3_cp[-2]:
        #                     # down trend
        #                     if not self.waiting_for_order:
        #                         self.exp6.append(len(exp3_cp))
        #                         self.exp7.append(exp3_cp[-1])
        #                         self.sell_order(self.trading_data[-1])
        #                         self.waiting_for_order = True
        #
        #             self.waiting_time += 1
        # avg_histogram = sum([x for x in histogram[-5:]]) / 5
        # if avg_histogram > 0.2:
        #     self.exp4.append(len(histogram_cp))
        #     self.exp5.append(histogram_cp[-1])
        # if -0.2 > avg_histogram:
        #     self.exp6.append(len(histogram_cp))
        #     self.exp7.append(histogram_cp[-1])

        self.ax1.cla()
        self.ax2.cla()
        self.ax2.axhline(0.5, color='black')
        self.ax1.plot(df.ds, self.trading_data, label='Raw data')
        # self.ax2.plot(df.ds, macd, label='AMD MACD', color='#EBD2BE')
        # self.ax2.plot(df.ds, exp3, label='Signal Line', color='#E5A4CB')
        self.ax2.plot(df.ds, histogram_cp, label='Histogram', color='#ABD2BE')
        self.ax2.legend(loc='upper left')
        self.ax2.plot(self.exp4, self.exp5, 'ro', color='g')
        self.ax2.plot(self.exp6, self.exp7, 'ro', color='r')
        # plt.plot([len(self.trading_data) - self.windows//2], [histogram_cp[len(self.trading_data) - self.windows//2]], 'ro', color='k')
        self.fig.canvas.draw()
        plt.pause(0.0001)

        self.tqdm_e.set_description("Profit: {}, Stop Loss: {}, Take Profit: {}".format(round(self.budget, 2),
                                                                                        round(self.total_lost, 2),
                                                                                        round(self.total_profit, 2)))
        self.tqdm_e.refresh()

    def check_lost(self, price):
        """Close order when loss $5"""
        # if not self.waiting_for_order and price < self.stop_loss:
        #     self.sell_order(price)
        #     logging.warning("Stop loss: {} => {} profit {} budget: {}".format(self.order, price,
        #                                                                       round(price - self.order, 2),
        #                                                                       self.budget))
        #     return True
        return False

    def check_profit(self, price):
        """Close order when take $5 profit"""
        # if not self.waiting_for_order and price >= self.take_profit:
        #     self.sell_order(price)
        #     self.waiting_for_order = True
        #     return True
        return False

    def buy_order(self, price):
        if self.waiting_for_order:
            order_info = {
                'id': str(uuid.uuid4()),
                'price': price,
                'type': 'buy',
                'stop_loss': price
            }
            custom_range = self.trading_data[self.queue_size*2//3:]
            min_price = min(custom_range)
            max_price = max(custom_range)
            take_profit, stop_loss = self.fibonacci(price_max=max_price, price_min=min_price)
            # if len(list(filter(lambda d: d['price'] == price, self.order_history))) == 0:
            self.order_history.append(order_info)
            self.order = price
            self.stop_loss = price - 150
            self.take_profit = take_profit
            self.order_type = 'buy'
            self.waiting_for_order = False

    def sell_order(self, price):
        if not self.waiting_for_order:
            order_info = {
                'id': str(uuid.uuid4()),
                'price': price,
                'type': 'sell'
            }

            # if len(list(filter(lambda d: d['price'] == price, self.order_history))) == 0:
            #     self.order_history.append(order_info)
            self.waiting_for_order = True
            diff = price - self.order
            self.budget = self.budget + round(diff, 2)
            if diff > 0:
                logging.warning("Take Profit: {} => {} profit {} budget: {}".format(self.order, price,
                                                                                    round(diff, 2), self.budget))
                self.take_profit_nb += 1
                self.total_profit += diff
            else:
                # logging.warning("Loss: {} => {} profit {} budget: {}".format(self.order, price,
                #                                                              round(diff, 2), self.budget))
                self.stop_loss_nb += 1
                self.total_lost += diff
            # clear status
            self.order_type = 'sell'
            self.waiting_time = 0

            return diff

    def test_order(self):
        """Create a new test order"""
        order = self.binace_client.create_test_order(
            symbol='BTCUSDT',
            side=SIDE_BUY,
            type=ORDER_TYPE_MARKET,
            quantity=100
        )

    @classmethod
    def fibonacci(cls, price_max, price_min):
        diff = price_max - price_min
        level1 = price_max - 0.236 * diff
        level2 = price_max - 0.382 * diff
        level3 = price_max - 0.618 * diff
        stop_loss = price_max - 1.382 * diff
        return level1, stop_loss

    def getStockDataVec(self, key):
        indexes = []
        lines = open("../data/" + key + ".csv", "r").read().splitlines()
        prices = []
        delimiter = ','
        for _index, line in enumerate(lines[2:]):
            prices.append(float(line.split(delimiter)[5]))

        return indexes, prices

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        conn_key = self.bm.start_kline_socket('BTCUSDT', self.process_message, interval=KLINE_INTERVAL_1HOUR)
        # then start the socket manager
        self.bm.start()

    def start_mockup(self, kind_of_run):
        indexes, price_data = trading_bot.getStockDataVec('1hours')
        start_idx = 0
        end_idx = -1
        # price_data = list(reversed(price_data[start_idx: end_idx]))
        # price_data = list(reversed(price_data))
        total_sample = len(price_data)
        index = [i for i, val in enumerate(price_data)]
        df = pd.DataFrame({'index': index, 'close': price_data})
        df = df[['close']]
        df.reset_index(level=0, inplace=True)
        df.columns = ['ds', 'y']
        # self.ax2.plot(df.ds, price_data, label='Price')
        # plt.show()
        # self.fig.show()
        # self.fig.canvas.draw()
        # time.sleep(0.1)

        print("Max profit: {}".format(price_data[-1] - price_data[0]))
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
    trading_bot.start_mockup("train")
    # trading_bot.start_socket()
