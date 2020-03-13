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

logging.basicConfig(filename='log/cci.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class AutoTrading:
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

    def CalculateCCI(dataRaw, ndays):
        """Commodity Channel Index"""
        TP = (dataRaw['High'] + dataRaw['Low'] + dataRaw['Close']) / 3
        rawCCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)), name='CCI')
        return_data = dataRaw.join(rawCCI)
        return return_data


# Retrieve the Nifty data from Yahoo finance:
# data = web.get_data_yahoo("^NSEI", start="2019-01-01", end="2019-03-13")
order = 0
budget = 0
total_step = 0
fig = plt.figure(figsize=(7, 5))
header_list = ["Open", "High", "Low", "Close"]
data = pd.read_csv("data/bnb5minute.csv", sep=',')
exp4 = []
exp5 = []
exp6 = []
exp7 = []
delta = []
budget_list = []
for x in range(0, len(data) - 200):
    data1 = data.iloc[x:x+200, :]

    # Compute the Commodity Channel Index(CCI) for NIFTY based on the 20-day Moving average
    n = 20
    NI = CalculateCCI(data1, n)
    CCI = NI['CCI']
    CCI = CCI.fillna(0)
    current_cci = round(list(CCI)[-1], 2)
    prev_cci = round(list(CCI)[-2], 2)
    prev_prev_cci = round(list(CCI)[-3], 2)

    exp4 = [x1 - 1 for x1 in exp4]
    exp6 = [x - 1 for x in exp6]
    point_1 = list(data1['Close'])[0]
    delta.append(np.mean([current_cci, prev_cci, prev_prev_cci]))
    anomaly_point = delta[-1]
    current_price = list(data1['Close'])[-1]

    if order == 0 and anomaly_point < -100:
        if prev_cci < current_cci:
            is_close_buy_signal = list(np.isclose([anomaly_point], [-150.0], atol=20))[0]
            if is_close_buy_signal and anomaly_point > -150:
                order = list(data1['Close'])[-1]
                # print("buy: {}".format(current_cci))
                exp4.append(len(list(data1['Close'])))
                exp5.append(list(data1['Close'])[-1])
                total_step = 0

    # elif order == 0 and 50 < np.mean([x for x in list(CCI)[-5:]]) <= 100:
    #     if current_cci > 100:
    #         order = list(data1['Close'])[-1]
    #         # print("buy: {}".format(current_cci))
    #         exp4.append(len(list(data1['Close'])))
    #         exp5.append(list(data1['Close'])[-1])
    #         total_step = 0

    elif order != 0 and list(np.isclose([anomaly_point], [0.0], atol=10))[0]:
        if current_cci > 0.0:
            budget += list(data1['Close'])[-1] - order
            print("sell: {} budget: {} total step: {}".format(anomaly_point, round(budget, 2), total_step))
            order = 0
            total_step = 0
            exp6.append(len(list(data1['Close'])))
            exp7.append(list(data1['Close'])[-1])
            budget_list.append(budget)

    elif order != 0 and list(np.isclose([anomaly_point], [100.0], atol=20))[0]:
        if current_cci > 100.0:
            budget += list(data1['Close'])[-1] - order
            print("sell: {} budget: {} total step: {}".format(anomaly_point, round(budget, 2), total_step))
            order = 0
            total_step = 0
            exp6.append(len(list(data1['Close'])))
            exp7.append(list(data1['Close'])[-1])
            budget_list.append(budget)

    # if order != 0:
    #     if list(np.isclose([anomaly_point], [200.0], atol=30))[0]:
    #         budget += list(data1['Close'])[-1] - order
    #         print("sell: {} budget: {} total step: {}".format(anomaly_point, round(budget, 2), total_step))
    #         order = 0
    #         total_step = 0
    #         exp6.append(len(list(data1['Close'])))
    #         exp7.append(list(data1['Close'])[-1])

    # if order != 0 and current_price - order < -10:
    #     # stop loss
    #     budget += list(data1['Close'])[-1] - order
    #     print("stop loss: {} budget: {} total step: {}".format(current_cci, round(budget, 2), total_step))
    #     order = 0
    #     total_step = 0
    #     exp6.append(len(list(data1['Close'])))
    #     exp7.append(list(data1['Close'])[-1])

    if order != 0:
        total_step += 1

    for x4 in exp4:
        if x4 < 20:
            exp4 = exp4[1:]
            exp5 = exp5[1:]

    for x6 in exp6:
        if x6 < 20:
            exp6 = exp6[1:]
            exp7 = exp7[1:]

    if len(delta) >= 200:
        delta.pop(0)

    # if len(budget_list) >= 200:
    #     budget_list.pop(0)

    if x == len(data) - 200 - 1:
        # Plotting the Price Series chart and the Commodity Channel index below
        index = [i for i, val in enumerate(list(data['Close']))]
        ax = fig.add_subplot(2, 1, 1)
        ax.set_xticklabels([])
        plt.plot(index, data['Close'], lw=1)
        plt.plot(exp4, exp5, 'ro', color='g')
        plt.plot(exp6, exp7, 'ro', color='r')
        plt.title('BNBUSDT Chart')
        plt.ylabel('Close Price')
        plt.grid(True)
        # bx = fig.add_subplot(4, 1, 2)
        # plt.plot(index, CCI, 'k', lw=0.75, linestyle='-', label='CCI')
        # plt.legend(loc=2, prop={'size': 9.5})
        # plt.ylabel('CCI values')
        # plt.grid(True)
        # cx = fig.add_subplot(4, 1, 3)
        # plt.plot(delta)
        # plt.ylabel('Delta')
        # plt.grid(True)
        dx = fig.add_subplot(2, 1, 2)
        plt.plot(budget_list)
        plt.ylabel('Budget')
        plt.grid(True)
        plt.setp(plt.gca().get_xticklabels(), rotation=30)

        plt.pause(0.0001)
        plt.show()
        # ax.cla()
        # bx.cla()
        # cx.cla()
        # plt.cla()
