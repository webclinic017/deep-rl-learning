import math
import time
import logging
import uuid
import random

import pandas as pd
import matplotlib.pyplot as plt
from binance.enums import *
from binance.client import Client
from binance.enums import KLINE_INTERVAL_1HOUR
from binance.websockets import BinanceSocketManager
from keras import Model
from keras.utils import to_categorical
from pymongo import MongoClient
from tqdm import tqdm
import numpy as np
from keras.layers import Input, Dense, Flatten, LSTM, concatenate
from A2C.a2c import A2C

logging.basicConfig(filename='log/autotrade.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class AutoTrading(A2C):
    def __init__(self, act_dim, env_dim, k):
        super().__init__(act_dim, env_dim, k)
        self.actions, self.states, self.rewards = [], [], []
        self.api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        self.api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        self.binace_client = Client(self.api_key, self.api_secret)
        self.bm = BinanceSocketManager(self.binace_client)
        # mongodb
        self.client = MongoClient()
        self.db = self.client.crypto
        self.order = 0
        self.norm_order = 0
        self.budget = 0
        self.order_type = 'sell'
        self.trading_data = []
        self.indexes = []
        self.norm_data = []
        self.windows = 6
        self.threshold = 0.05
        self.vec_threshold = 0.15
        self.trade_threshold = 90
        self.queue_size = 100
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

        # plot point data
        self.exp4 = []
        self.exp5 = []
        self.exp6 = []
        self.exp7 = []

    def reset(self):
        self.order = 0
        self.norm_order = 0
        self.budget = 0
        self.order_type = 'sell'
        self.trading_data = []
        self.indexes = []
        self.norm_data = []
        self.waiting_for_order = True

    def buildNetwork(self):
        """ Assemble shared layers"""
        initial_input = Input(shape=(10, 10))
        secondary_input = Input(shape=(2,))

        lstm = LSTM(128, dropout=0.1, recurrent_dropout=0.3)(initial_input)
        dense = Dense(128, activation='relu')(secondary_input)
        merge = concatenate([lstm, dense])

        first_dense = Dense(128, activation='relu')(merge)
        second_dense = Dense(64, activation='relu')(first_dense)
        output = Dense(32, activation='relu')(second_dense)
        model = Model(inputs=[initial_input, secondary_input], outputs=output)
        return model

    def process_message(self, msg, a):
        current_time = time.time()
        msg['k']['timestamp'] = current_time
        # print(msg)
        # insert = self.db.btc_data.insert_one(msg['k']).inserted_id

        self.trading_data.append(float(msg['k']['c']))

        if len(self.trading_data) > self.queue_size:
            self.trading_data.pop(0)
        if len(self.norm_data) > self.queue_size:
            self.norm_data.pop(0)

        # return for a2c
        new_state = None
        r = -1
        done = False
        info = {}

        if len(self.trading_data) == self.queue_size:
            self.norm_data = self.norm_list(self.trading_data)  # normally data
            new_state, r, done, info = self.calculate_macd(a)

        return new_state, r, done, info

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

    def calculate_macd(self, action):
        """
        Simulation trading and RL learning
        :return:
        """
        index = [i for i, val in enumerate(self.trading_data)]
        df = pd.DataFrame({'index': index, 'close': self.trading_data})
        df = df[['close']]
        df.reset_index(level=0, inplace=True)
        df.columns = ['ds', 'y']

        exp1 = df.y.ewm(span=12, adjust=False).mean()
        exp2 = df.y.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        exp3 = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - exp3

        # if len(self.trading_data) == self.queue_size + 1:
        #     exp4 = self.exp4.copy()
        #     for x4 in exp4:
        #         if x4 < 0:
        #             self.exp4.pop(0)
        #             self.exp5.pop(0)
        #
        #     exp6 = self.exp6.copy()
        #     for x6 in exp6:
        #         if x6 < 0:
        #             self.exp6.pop(0)
        #             self.exp7.pop(0)
        #
        #     self.exp4 = [x1 - 1 for x1 in self.exp4]
        #     self.exp6 = [x - 1 for x in self.exp6]

        histogram_cp = list(histogram)
        # start_point = len(histogram_cp) - self.windows - 1
        # end_point = len(histogram_cp) - 1
        # block = histogram_cp[start_point:]
        # chunk_idx = len(histogram_cp) - self.windows // 2 - 1
        # angel_degree_1 = self.angle_of_vectors(x=chunk_idx, x0=start_point,
        #                                        y=histogram_cp[chunk_idx], y0=histogram_cp[start_point])
        # angel_degree_2 = self.angle_of_vectors(x=end_point, x0=chunk_idx,
        #                                        y=histogram_cp[end_point], y0=histogram_cp[chunk_idx])
        #
        # if block[self.windows // 2] < block[0]:
        #     # trend down
        #     angel_degree_1 = - angel_degree_1
        # if block[-1] < block[self.windows // 2]:
        #     # trend down
        #     angel_degree_2 = - angel_degree_2
        #
        # total_degree = angel_degree_1 + angel_degree_2
        # compare_list = [x for x in histogram_cp[-5:]]
        # avg_histogram = np.mean(compare_list)
        # max_histogram = max(compare_list)
        # min_histogram = min(compare_list)
        # is_close = list(np.isclose([histogram_cp[-1]], [0.0], atol=0.07))[0]
        # if is_close:
        #     if histogram_cp[-1] > avg_histogram:
        #         self.exp4.append(len(histogram_cp))
        #         self.exp5.append(histogram_cp[-1])
        #     elif histogram_cp[-1] < avg_histogram:
        #         self.exp6.append(len(histogram_cp))
        #         self.exp7.append(histogram_cp[-1])

        diff = self.trading_data[-1] - self.order
        r = 0
        if action == 0:
            # hold
            if self.order != 0:
                r = 0.01 * diff
            else:
                r = 0
        elif action == 1:
            # close
            if self.order != 0:
                r = 0.2 * diff
                self.sell_order(self.trading_data[-1])
            else:
                r = 0
        elif action == 2:
            # buy
            if self.order == 0:
                r = 0.01
                self.buy_order(self.trading_data[-1])
            else:
                r = 0

        # self.ax1.cla()
        # self.ax2.cla()
        # self.ax2.axhline(0.0, color='black')
        # self.ax1.plot(df.ds, self.trading_data, label='Raw data')
        # # self.ax2.plot(df.ds, macd, label='AMD MACD', color='#EBD2BE')
        # # self.ax2.plot(df.ds, exp3, label='Signal Line', color='#E5A4CB')
        # self.ax2.plot(df.ds, histogram_cp, label='Histogram: {}'.format(round(total_degree, 2)), color='#ABD2BE')
        # self.ax2.legend(loc='upper left')
        # self.ax2.plot(self.exp4, self.exp5, 'ro', color='g')
        # self.ax2.plot(self.exp6, self.exp7, 'ro', color='r')
        # self.ax2.plot([len(self.trading_data) - self.windows//2], [histogram_cp[len(self.trading_data) - self.windows//2]], 'ro', color='k')
        # self.fig.canvas.draw()
        # plt.pause(0.00001)

        d = 80
        data = []
        for i in range(10):
            block = histogram_cp[d:d+10]
            data.append(block)
            d += 1

        state = np.array(data)
        done = True
        info = {'order': self.order/10000, 'price': self.trading_data[-1]/10000}

        return state, r, done, info

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
            self.norm_order = self.norm_data[-1]
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
            self.order = 0
            self.norm_order = 0
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
        lines = open("data/" + key + ".csv", "r").read().splitlines()
        prices = []
        delimiter = ','
        for _index, line in enumerate(lines[2:]):
            prices.append(float(line.split(delimiter)[5]))

        return indexes, prices

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        conn_key = self.bm.start_kline_socket('BTCUSDT', self.process_message, interval=KLINE_INTERVAL_1MINUTE)
        # then start the socket manager
        self.bm.start()

    def start_mockup(self, kind_of_run):
        # self.fig.show()
        indexes, raw_data = trading_bot.getStockDataVec('1minutes')
        total_sample = len(raw_data)
        episode = total_sample // 500
        start_idx = 0

        for e in range(10000):
            # A2C parameters
            time, cumul_reward, done = 0, 0, False
            actions, states, rewards = [], [], []
            start_idx = random.randint(0, len(raw_data) - 2000)
            end_idx = start_idx + 500
            # price_data = list(reversed(price_data[start_idx: end_idx]))
            price_data = raw_data[start_idx: end_idx]
            inp1, inp2 = self.getState()

            self.tqdm_e = tqdm(price_data, desc='Score', leave=True, unit=" budget")
            for item in self.tqdm_e:
                a = self.policy_action(inp1, inp2)
                msg = {'k': {'c': item}}
                new_state, r, ready, info = trading_bot.process_message(msg, a)
                if ready:
                    cumul_reward += r
                    inp1 = new_state
                    inp2 = np.array([info['order'], info['price']])
                    actions.append(to_categorical(a, self.act_dim))
                    rewards.append(r)
                    states.append([inp1, inp2])
                    self.tqdm_e.set_description("Profit: {}, Stop Loss: {}, Take Profit: {}, Cumul reward: {}, EP: {}".format(
                        round(self.budget, 2),
                        round(self.total_lost, 2),
                        round(self.total_profit, 2),
                        round(cumul_reward, 2),
                        e)
                    )
                    self.tqdm_e.refresh()
            done = True if self.budget > 0 else False
            self.train_models(states, actions, rewards, done)
            trading_bot.reset()

    def getState(self):
        inp1 = np.random.randint(0, 1, (10, 10))
        inp2 = np.random.randint(0, 1, (2,))
        return inp1, inp2


if __name__ == '__main__':
    state_dim = (1,)
    action_dim = 3
    act_range = 2
    consecutive_frames = 10
    trading_bot = AutoTrading(action_dim, state_dim, consecutive_frames)
    trading_bot.start_mockup("train")

    trading_bot.save_weights('models/new_way')
    # trading_bot.start_socket()
