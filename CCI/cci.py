from pprint import pprint

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import time
import logging
import uuid
import random

from binance.enums import *
from binance.client import Client
from binance.enums import KLINE_INTERVAL_1HOUR
from binance.websockets import BinanceSocketManager
from pymongo import MongoClient
from tqdm import tqdm
from CCI.mailer import SendMail
from keras.layers import Input, Dense, Flatten, LSTM, concatenate
from keras import Model
from keras.utils import to_categorical
from A2C.a2c import A2C

logging.basicConfig(filename='log/cci.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class AutoTrading(A2C):
    def __init__(self, act_dim, env_dim, k):
        super().__init__(act_dim, env_dim, k)
        # Actor Critic Parameters
        self.actions, self.states, self.rewards = [], [], []
        self.time = 0
        self.cumul_reward = 0
        self.done = False
        self.waiting_time = 0

        self.api_key = "9Hj6HLNNMGgkqj6ngouMZD1kjIbUb6RZmIpW5HLiZjtDT5gwhXAzc20szOKyQ3HW"
        self.api_secret = "ioD0XICp0cFE99VVql5nuxiCJEb6GK8mh08NYnSYdIUfkiotd1SZqLTQsjFvrXwk"
        self.binace_client = Client(self.api_key, self.api_secret)
        self.bm = BinanceSocketManager(self.binace_client)
        # mongodb
        self.client = MongoClient()
        self.db = self.client.crypto

        # env
        self.order = 0
        self.budget = 0
        self.total_step = 0
        self.header_list = ["Open", "High", "Low", "Close"]
        # self.data = pd.read_csv("data/bnb5minute.csv", sep=',')
        self.exp4 = []
        self.exp5 = []
        self.exp6 = []
        self.exp7 = []
        self.delta = []
        self.budget_list = []
        klines = self.binace_client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_5MINUTE, "13 Mar, 2020")
        df = pd.DataFrame(klines, columns=['open_time', 'Open', 'High', 'Low', 'Close',
                                                'Volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                                                'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])
        df = df.drop(['open_time', 'Volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                      'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'], axis=1)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        self.data = df

        # draw canvas
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(411, label='Price')
        self.bx = self.fig.add_subplot(412, label='CCI')
        self.cx = self.fig.add_subplot(413, label='Budget')

    def buildNetwork(self):
        """ Assemble shared layers"""
        initial_input = Input(shape=(1, 40))
        secondary_input = Input(shape=(2,))

        lstm = LSTM(128, dropout=0.1, recurrent_dropout=0.3)(initial_input)
        dense = Dense(128, activation='relu')(secondary_input)
        merge = concatenate([lstm, dense])

        first_dense = Dense(128, activation='relu')(merge)
        second_dense = Dense(128, activation='relu')(first_dense)
        output = Dense(32, activation='relu')(second_dense)
        model = Model(inputs=[initial_input, secondary_input], outputs=output)
        return model

    @staticmethod
    def calculate_cci(dataRaw, ndays):
        """Commodity Channel Index"""
        TP = (dataRaw['High'] + dataRaw['Low'] + dataRaw['Close']) / 3
        rawCCI = pd.Series((TP - TP.rolling(ndays).mean()) / (0.015 * TP.rolling(ndays).std()), name='CCI')
        return_data = dataRaw.join(rawCCI)
        return return_data

    def test_order(self):
        info = self.binace_client.get_margin_account()
        get_margin_asset = self.binace_client.get_margin_asset(asset='BTC')

        usdt_amount = info['userAssets'][2]['free']
        details = self.binace_client.get_max_margin_transfer(asset='BTC')
        logging.warning("Margin Lever: {}".format(info['marginLevel']))
        print("Margin Lever: {}".format(info['marginLevel']))
        logging.warning("Amount in BTC: {}".format(info['totalAssetOfBtc']))
        print("Amount in BTC: {}".format(info['totalAssetOfBtc']))
        logging.warning("Account Status: {}".format(info['tradeEnabled']))
        print("Account Status: {}".format(info['tradeEnabled']))

        symbol = 'BTCUSDT'
        # self.buy_margin()
        # time.sleep(5)
        self.sell_margin()

        print("Test Done!!")

    def sell_margin(self):
        info = self.binace_client.get_margin_account()
        symbol = 'BTCUSDT'
        amount = info['totalAssetOfBtc']
        precision = 5
        amt_str = "{:0.0{}f}".format(float(amount)*0.98, precision)
        logging.warning("Sell: {}".format(amt_str))
        mailer = SendMail()
        try:
            sell_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str)

            info = self.binace_client.get_margin_account()
            current_btc = info['totalAssetOfBtc']
            usdt = info['userAssets'][2]['free']
            txt = "Sell successfully: Balance: {} Sell Amount: {} Owned BTC: {}".format(usdt, amt_str, current_btc)
            logging.warning(txt)
            mailer.notification(txt)
            return True
        except Exception as ex:
            amt_str = "{:0.0{}f}".format(float(amount) * 0.9, precision)
            sell_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str)

            info = self.binace_client.get_margin_account()
            current_btc = info['totalAssetOfBtc']
            usdt = info['userAssets'][2]['free']
            txt = "Sell successfully: Balance: {} Sell Amount: {} Owned BTC: {}".format(usdt, amt_str, current_btc)
            logging.warning(txt)
            mailer.notification(txt)
            logging.error(ex)
            return False

    def buy_margin(self):
        symbol = 'BTCUSDT'
        info = self.binace_client.get_margin_account()
        usdt_amount = info['userAssets'][2]['free']
        price_index = self.binace_client.get_margin_price_index(symbol=symbol)
        amount = float(usdt_amount)/float(price_index['price'])
        precision = 5
        amt_str = "{:0.0{}f}".format(amount*0.98, precision)
        mailer = SendMail()
        logging.warning("Buy : {}".format(amt_str))
        try:
            buy_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str)
            txt = "Buy successfully: Amount: {}".format(amt_str)
            logging.warning(txt)
            mailer.notification(txt)
            return True
        except Exception as ex:
            logging.error(ex)
            return False

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        conn_key = self.bm.start_kline_socket('BTCUSDT', self.process_message, interval=KLINE_INTERVAL_5MINUTE)
        # then start the socket manager
        self.bm.start()

    def process_message(self, msg):
        msg = msg['k']
        _open = float(msg['o'])
        _high = float(msg['h'])
        _low = float(msg['l'])
        _close = float(msg['c'])
        _latest = msg['x']
        insert = self.db.btc15minutes.insert_one(msg).inserted_id

        if _latest:
            df2 = pd.DataFrame([[_open, _high, _low, _close]], columns=['Open', 'High', 'Low', 'Close'])
            self.data = pd.concat((self.data, df2), ignore_index=True)
            n = 20
            data1 = self.data.iloc[-200:, :]
            NI = self.calculate_cci(data1, n)

            CCI = NI['CCI']
            CCI = CCI.fillna(0)
            current_cci = round(list(CCI)[-1], 2)
            prev_cci = round(list(CCI)[-2], 2)
            prev_prev_cci = round(list(CCI)[-3], 2)

            anomaly_point = np.mean([current_cci, prev_cci, prev_prev_cci])
            current_price = list(data1['Close'])[-1]
            logging.warning("Current Price: {}".format(current_price))
            if self.order == 0 and anomaly_point < -100:
                if prev_cci < current_cci:
                    is_close_buy_signal = list(np.isclose([anomaly_point], [-140.0], atol=20))[0]
                    if is_close_buy_signal and anomaly_point > -150:
                        if self.buy_margin():
                            self.order = list(data1['Close'])[-1]
                            print("buy: {}".format(current_price))

            elif self.order != 0 and list(np.isclose([anomaly_point], [0.0], atol=10))[0]:
                # close order 50:50
                if current_cci > 0.0:
                    if self.sell_margin():
                        self.budget += current_price - self.order
                        print(
                            "sell: {} budget: {} total step: {}".format(current_price, round(self.budget, 2),
                                                                        self.total_step))
                        self.order = 0
                        self.total_step = 0

            elif self.order != 0 and list(np.isclose([anomaly_point], [100.0], atol=10))[0]:
                # take profit
                if current_cci > 100.0:
                    if self.sell_margin():
                        self.budget += current_price - self.order
                        print(
                            "sell: {} budget: {} total step: {}".format(current_price, round(self.budget, 2),
                                                                        self.total_step))
                        self.order = 0
                        self.total_step = 0

            elif self.order != 0 and list(np.isclose([anomaly_point], [200.0], atol=20))[0]:
                # Super take profit
                if current_cci > 180.0:
                    if self.sell_margin():
                        self.budget += current_price - self.order
                        print(
                            "sell: {} budget: {} total step: {}".format(current_price, round(self.budget, 2),
                                                                        self.total_step))
                        self.order = 0
                        self.total_step = 0

            if self.order != 0:
                self.total_step += 1

    def mock_data(self):
        df = pd.read_csv("../data/5minute.csv", sep=',')
        for x in range(0, len(df) - 200):
            data = df.iloc[x:x+160, :]
            self.process_mock(data)

        # Plotting the Price Series chart and the Commodity Channel index below
        self.ax.cla()
        self.bx.cla()
        self.cx.cla()
        index = [i for i, val in enumerate(list(df['Close']))]
        self.ax.set_xticklabels([])
        self.ax.plot(index, df['Close'], lw=1, label='Price')
        self.ax.plot(self.exp4, self.exp5, 'ro', color='g')
        self.ax.plot(self.exp6, self.exp7, 'ro', color='r')
        self.ax.legend(loc='upper left')
        self.ax.grid(True)
        self.bx.set_xticklabels([])
        self.bx.plot(self.delta, label='CCI')
        self.bx.legend(loc='upper left')
        self.bx.grid(True)
        self.cx.set_xticklabels([])
        self.cx.plot(self.budget_list, label='Budget')
        self.cx.legend(loc='upper left')
        self.cx.grid(True)
        self.fig.canvas.draw()
        plt.show()

    def process_mock(self, data1):
        """
        Mock data used for test algorithm
        :param data1:
        :return:
        """
        ndays = 20
        data_cp = data1.copy()
        NI = self.calculate_cci(data_cp, ndays)

        CCI = NI['CCI']
        CCI = CCI.fillna(0)
        current_cci = round(list(CCI)[-1], 2)
        prev_cci = round(list(CCI)[-2], 2)
        prev_prev_cci = round(list(CCI)[-3], 2)

        self.exp4 = [x1 - 1 for x1 in self.exp4]
        self.exp6 = [x - 1 for x in self.exp6]

        anomaly_point = np.mean([current_cci, prev_cci, prev_prev_cci])

        self.delta.append(anomaly_point)
        current_price = list(data1['Close'])[-1]
        is_close_buy_signal = list(np.isclose([current_cci], [-150.0], atol=50))[0]
        if self.order == 0 and is_close_buy_signal:
            if prev_cci < current_cci and anomaly_point > -200:
                self.order = list(data1['Close'])[-1]
                print("buy: {}".format(current_cci))
                self.exp4.append(len(list(data1['Close'])))
                self.exp5.append(list(data1['Close'])[-1])

        elif self.order != 0 and list(np.isclose([current_cci], [0.0], atol=20))[0]:
            if current_cci > 0:
                # close order 50:50
                self.budget += current_price - self.order
                print(
                    "sell: {} budget: {} total step: {}".format(current_cci, round(self.budget, 2), self.total_step))
                self.order = 0
                self.total_step = 0
                self.exp6.append(len(list(data1['Close'])))
                self.exp7.append(current_price)
                self.budget_list.append(self.budget)

        elif self.order != 0 and list(np.isclose([current_cci], [100.0], atol=20))[0]:
            # take profit
            if current_cci > 100.0:
                self.budget += current_price - self.order
                print(
                    "sell: {} budget: {} total step: {}".format(current_cci, round(self.budget, 2), self.total_step))
                self.order = 0
                self.total_step = 0
                self.exp6.append(len(list(data1['Close'])))
                self.exp7.append(current_price)
                self.budget_list.append(self.budget)

        elif self.order != 0 and list(np.isclose([current_cci], [150.0], atol=20))[0]:
            # Super take profit
            if current_cci > 150.0:
                self.budget += current_price - self.order
                print(
                    "sell: {} budget: {} total step: {}".format(current_cci, round(self.budget, 2), self.total_step))
                self.order = 0
                self.total_step = 0
                self.exp6.append(len(list(data1['Close'])))
                self.exp7.append(current_price)
                self.budget_list.append(self.budget)

        elif self.order != 0 and list(np.isclose([current_cci], [200.0], atol=30))[0]:
            # Super take profit
            if current_cci > 200.0:
                self.budget += current_price - self.order
                print(
                    "sell: {} budget: {} total step: {}".format(current_cci, round(self.budget, 2), self.total_step))
                self.order = 0
                self.total_step = 0
                self.exp6.append(len(list(data1['Close'])))
                self.exp7.append(current_price)
                self.budget_list.append(self.budget)

        elif self.order != 0 and self.total_step > 25:
            self.budget += current_price - self.order
            print(
                "sell: {} budget: {} total step: {}".format(current_cci, round(self.budget, 2), self.total_step))
            self.order = 0
            self.total_step = 0
            self.exp6.append(len(list(data1['Close'])))
            self.exp7.append(current_price)
            self.budget_list.append(self.budget)

        if self.order != 0:
            self.total_step += 1

        for x4 in self.exp4:
            if x4 < 20:
                self.exp4 = self.exp4[1:]
                self.exp5 = self.exp5[1:]

        for x6 in self.exp6:
            if x6 < 20:
                self.exp6 = self.exp6[1:]
                self.exp7 = self.exp7[1:]

        if len(self.delta) >= 200:
            self.delta.pop(0)

    def learning_start(self):
        df = pd.read_csv("data/5minute.csv", sep=',')
        tqdm_e = tqdm(range(0, len(df) - 200), desc='Score', leave=True, unit=" budget")
        time, cumul_reward, done = 0, 0, False
        actions, states, rewards = [], [], []
        inp1, inp2 = self.getState()
        for x in tqdm_e:
            data = df.iloc[x:x + 160, :]
            a = self.policy_action(inp1, inp2)
            new_state, r, done, info = self.learning(a, data)
            cumul_reward += r
            inp1 = np.array([new_state])
            inp2 = np.array([[info['diff'], info['order']]])
            actions.append(to_categorical(a, self.act_dim))
            rewards.append(r)
            states.append([inp1, inp2])

            if x % 128 == 0 and x > 0:
                self.train_models(states, actions, rewards, done)
                tqdm_e.set_description(
                    "Profit: {}, Cumul reward: {}, EP: {}".format(
                        round(self.budget, 2),
                        round(cumul_reward, 2), x)
                    )
                tqdm_e.refresh()
                time, cumul_reward, done = 0, 0, False
                actions, states, rewards = [], [], []
                self.reset()

    def reset(self):
        self.order = 0
        self.budget = 0

    def learning(self, action, data1):
        ndays = 20
        data_cp = data1.copy()
        NI = self.calculate_cci(data_cp, ndays)

        CCI = NI['CCI']
        CCI = CCI.fillna(0)
        current_cci = round(list(CCI)[-1], 2)
        prev_cci = round(list(CCI)[-2], 2)
        prev_prev_cci = round(list(CCI)[-3], 2)

        self.exp4 = [x1 - 1 for x1 in self.exp4]
        self.exp6 = [x - 1 for x in self.exp6]

        anomaly_point = np.mean([current_cci, prev_cci, prev_prev_cci])

        self.delta.append(anomaly_point)
        current_price = list(data1['Close'])[-1]

        diff = current_price - self.order if self.order else 0
        r = 0
        done = False
        if action == 0:
            # hold
            pass

        elif action == 1:
            # sell
            if self.order:
                r = 2 * diff
                self.budget += diff
                self.order = 0
                self.total_step = 0
                self.exp6.append(len(list(data1['Close'])))
                self.exp7.append(current_price)
                self.budget_list.append(self.budget)
            else:
                r = -10

        elif action == 2:
            # buy
            if not self.order:
                self.order = list(data1['Close'])[-1]
                self.exp4.append(len(list(data1['Close'])))
                self.exp5.append(list(data1['Close'])[-1])
            else:
                r = -10
        state = np.array([list(CCI)[-40:]])
        info = {'diff': diff, 'order': 1 if self.order else 0}
        return state, r, done, info

    def getState(self):
        inp1 = np.random.randint(0, 1, (1, 1, 40))
        inp2 = np.random.randint(0, 1, (1, 2))
        return inp1, inp2


if __name__ == '__main__':
    state_dim = (1,)
    action_dim = 3
    act_range = 2
    consecutive_frames = 10
    trading_bot = AutoTrading(action_dim, state_dim, consecutive_frames)
    for _ in range(100):
        trading_bot.learning_start()
        trading_bot.save_weights('CCI/models/new_nodel')

    # actor critic learning
    # trading_bot.mock_data()
    # trading_bot.test_order()
    # trading_bot.start_socket()
