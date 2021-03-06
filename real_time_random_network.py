import datetime
import json
import os
import time
import logging
from time import strftime
import numpy as np
from binance.client import Client
from binance.enums import KLINE_INTERVAL_1DAY, KLINE_INTERVAL_15MINUTE
from binance.websockets import BinanceSocketManager
from pymongo import MongoClient

import argparse
import pandas as pd
from models import CuriosityNet
import tensorflow as tf
print(tf.__version__)

from keras.backend.tensorflow_backend import set_session
from utils.networks import get_session
logger = logging.getLogger(__name__)
f_handler = logging.FileHandler('log/file.log')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
f_handler.setFormatter(f_format)
logger.addHandler(f_handler)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#
graph = tf.get_default_graph()
state_dim = (1,)
action_dim = 3
windows = 10
with graph.as_default():
    algo = CuriosityNet(n_a=action_dim, n_s=windows, lr=0.000001, output_graph=True)
    algo.load_weight()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
binace_client = Client(api_key, api_secret)
bm = BinanceSocketManager(binace_client)


order = 0
budget = 1000

# mongodb
client = MongoClient()
db = client.crypto

# state
vec = list()
prev_price = 0.0
prev_time = 0


def select_action(a, current_price):
    global order, budget
    current_price = float(current_price)
    info = dict()
    info['status'] = 'hold'
    info['current_price'] = current_price

    if a == 0:
        pass
    elif a == 1:
        # buy
        if order == 0:
            info['status'] = 'buy'
            order = current_price
    elif a == 2:
        # close
        if order:
            info['status'] = 'sell'
            diff = (current_price - order)
            budget += diff
            order = 0
            if diff > 0:
                info['profit'] = True
    info['budget'] = budget
    logger.warning(json.dumps(info))
    return info


def get_state(current_price, current_time):
    global prev_time, prev_price, vec, order
    current_price = float(current_price)
    diff = current_time - prev_time
    delta = (current_price - prev_price) / diff
    vec.append(delta)  # normalize
    prev_price = current_price
    prev_time = current_time
    if len(vec) > windows:
        vec.pop(0)
        return np.array(vec)
    return False


def process_message(msg):
    with graph.as_default():
        current_time = time.time()
        msg['k']['timestamp'] = current_time
        state = get_state(msg['k']['c'], current_time)
        if type(state) is np.ndarray:
            a = algo.choose_action(state)
            info = select_action(a, msg['k']['c'])
            print(info)
        inset = db.btc_test_1.insert_one(msg['k']).inserted_id


def start_socket():
    # start any sockets here, i.e a trade socket
    print("Auto trading")
    conn_key = bm.start_kline_socket('BTCUSDT', process_message, interval=KLINE_INTERVAL_15MINUTE)
    # then start the socket manager
    bm.start()


if __name__ == '__main__':
    start_socket()

