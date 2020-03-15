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

logging.basicConfig(filename='log/cci.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class AutoTrading:
    def __init__(self):
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
        self.dx = self.fig.add_subplot(414, label='Histogram')

    @staticmethod
    def calculate_cci(dataRaw, ndays):
        """Commodity Channel Index"""
        TP = (dataRaw['High'] + dataRaw['Low'] + dataRaw['Close']) / 3
        rawCCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)), name='CCI')
        return_data = dataRaw.join(rawCCI)
        return return_data

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

    def sell_margin(self):
        info = self.binace_client.get_margin_account()
        symbol = 'BTCUSDT'
        amount = info['totalAssetOfBtc']
        precision = 5
        amt_str = "{:0.0{}f}".format(float(amount)*0.98, precision)
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
            print(txt)
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
            print(txt)
            mailer.notification(txt)
            print(ex)
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

        try:
            txt = "Buy successfully: Amount: {}".format(amt_str)
            print(txt)
            buy_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str)
            mailer.notification(txt)
            return True
        except Exception as ex:
            print(ex)
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
            print("Current Price: {}".format(current_price))
            if self.order == 0 and anomaly_point < -100:
                if prev_cci < current_cci:
                    is_close_buy_signal = list(np.isclose([anomaly_point], [-150.0], atol=20))[0]
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
        df = pd.read_csv("data/5minute.csv", sep=',')
        for x in range(0, len(df)):
            data = df.iloc[x:x+200, :]
            self.process_mock(data)

    def process_mock(self, data1):
        ndays = 20
        data_cp = data1.copy()
        TP = (data_cp['High'] + data_cp['Low'] + data_cp['Close']) / 3
        rawCCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)), name='CCI')
        NI = data_cp.join(rawCCI)

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
        if self.order == 0 and anomaly_point < -100:
            if prev_cci < current_cci:
                is_close_buy_signal = list(np.isclose([anomaly_point], [-150.0], atol=20))[0]
                if is_close_buy_signal and anomaly_point > -150:
                    self.order = list(data1['Close'])[-1]
                    # print("buy: {}".format(current_cci))
                    self.exp4.append(len(list(data1['Close'])))
                    self.exp5.append(list(data1['Close'])[-1])

        elif self.order != 0 and anomaly_point > 100:
            # up trend. must close order and buy
            if prev_cci < current_cci:
                is_close_buy_signal = list(np.isclose([anomaly_point], [120.0], atol=20))[0]
                if is_close_buy_signal and anomaly_point > 120:
                    # close order
                    self.budget += current_price - self.order
                    self.budget_list.append(self.budget)
                    self.order = 0
                    self.total_step = 0

                    # new order
                    self.order = list(data1['Close'])[-1]
                    self.exp4.append(len(list(data1['Close'])))
                    self.exp5.append(list(data1['Close'])[-1])

        elif self.order != 0 and list(np.isclose([anomaly_point], [0.0], atol=10))[0]:
            # close order 50:50
            if current_cci > 0.0:
                self.budget += current_price - self.order
                print(
                    "sell: {} budget: {} total step: {}".format(anomaly_point, round(self.budget, 2), self.total_step))
                self.order = 0
                self.total_step = 0
                self.exp6.append(len(list(data1['Close'])))
                self.exp7.append(current_price)
                self.budget_list.append(self.budget)

        elif self.order != 0 and list(np.isclose([anomaly_point], [100.0], atol=10))[0]:
            # take profit
            if current_cci > 100.0:
                self.budget += current_price - self.order
                print(
                    "sell: {} budget: {} total step: {}".format(anomaly_point, round(self.budget, 2), self.total_step))
                self.order = 0
                self.total_step = 0
                self.exp6.append(len(list(data1['Close'])))
                self.exp7.append(current_price)
                self.budget_list.append(self.budget)

        elif self.order != 0 and list(np.isclose([anomaly_point], [200.0], atol=20))[0]:
            # Super take profit
            if current_cci > 180.0:
                self.budget += current_price - self.order
                print(
                    "sell: {} budget: {} total step: {}".format(anomaly_point, round(self.budget, 2), self.total_step))
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

        # Plotting the Price Series chart and the Commodity Channel index below
        # self.ax.cla()
        # self.bx.cla()
        # self.cx.cla()
        # # self.dx.cla()
        # index = [i for i, val in enumerate(list(data1['Close']))]
        # self.ax.set_xticklabels([])
        # self.ax.plot(index, data1['Close'], lw=1, label='Price')
        # self.ax.plot(self.exp4, self.exp5, 'ro', color='g')
        # self.ax.plot(self.exp6, self.exp7, 'ro', color='r')
        # self.ax.legend(loc='upper left')
        # self.ax.grid(True)
        # self.bx.set_xticklabels([])
        # self.bx.plot(self.delta, label='CCI')
        # self.bx.legend(loc='upper left')
        # self.bx.grid(True)
        # self.cx.set_xticklabels([])
        # self.cx.plot(self.budget_list, label='Budget')
        # self.cx.legend(loc='upper left')
        # self.cx.grid(True)
        # # self.dx.set_xticklabels([])
        # # self.dx.plot(histogram, label='Histogram')
        # # self.dx.legend(loc='upper left')
        # # self.dx.grid(True)
        # self.fig.canvas.draw()
        # plt.pause(0.0001)


if __name__ == '__main__':
    trading_bot = AutoTrading()
    # trading_bot.mock_data()
    # trading_bot.test_order()
    trading_bot.start_socket()
