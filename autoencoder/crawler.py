import time
import logging
from binance.enums import KLINE_INTERVAL_1MINUTE, KLINE_INTERVAL_1HOUR, KLINE_INTERVAL_5MINUTE, SIDE_SELL, \
    ORDER_TYPE_MARKET, SIDE_BUY, KLINE_INTERVAL_15MINUTE
from pymongo import MongoClient
from binance.client import Client
from binance.websockets import BinanceSocketManager

logging.basicConfig(filename='../log/autotrade.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class RegressionMA:
    def __init__(self):
        self.client = MongoClient()
        self.db = self.client.crypto
        # self.db.btc_5minute_realtime.drop()

    def start_socket_5m(self):
        api_key = "9Hj6HLNNMGgkqj6ngouMZD1kjIbUb6RZmIpW5HLiZjtDT5gwhXAzc20szOKyQ3HW"
        api_secret = "ioD0XICp0cFE99VVql5nuxiCJEb6GK8mh08NYnSYdIUfkiotd1SZqLTQsjFvrXwk"
        binace_client = Client(api_key, api_secret)
        # start any sockets here, i.e a trade socket
        bm = BinanceSocketManager(binace_client)
        conn_key = bm.start_kline_socket('BTCUSDT', self.process_message_5m, interval=KLINE_INTERVAL_5MINUTE)
        # then start the socket manager
        bm.start()

    def process_message_5m(self, msg):
        msg['timestamp'] = time.time()
        insert = self.db.btc_5m.insert_one(msg).inserted_id
        print("Data 5m: {}".format(msg['k']['c']))

    def start_socket_1h(self):
        api_key = "9Hj6HLNNMGgkqj6ngouMZD1kjIbUb6RZmIpW5HLiZjtDT5gwhXAzc20szOKyQ3HW"
        api_secret = "ioD0XICp0cFE99VVql5nuxiCJEb6GK8mh08NYnSYdIUfkiotd1SZqLTQsjFvrXwk"
        binace_client = Client(api_key, api_secret)
        # start any sockets here, i.e a trade socket
        bm = BinanceSocketManager(binace_client)
        conn_key = bm.start_kline_socket('BTCUSDT', self.process_message_1h, interval=KLINE_INTERVAL_1HOUR)
        # then start the socket manager
        bm.start()

    def process_message_1h(self, msg):
        msg['k']['timestamp'] = time.time()
        insert = self.db.btc_1h.insert_one(msg).inserted_id
        print("Data 1h: {}".format(msg['k']['c']))

    def start_socket_15m(self):
        api_key = "9Hj6HLNNMGgkqj6ngouMZD1kjIbUb6RZmIpW5HLiZjtDT5gwhXAzc20szOKyQ3HW"
        api_secret = "ioD0XICp0cFE99VVql5nuxiCJEb6GK8mh08NYnSYdIUfkiotd1SZqLTQsjFvrXwk"
        binace_client = Client(api_key, api_secret)
        # start any sockets here, i.e a trade socket
        bm = BinanceSocketManager(binace_client)
        conn_key = bm.start_kline_socket('BTCUSDT', self.process_message_15m, interval=KLINE_INTERVAL_15MINUTE)
        # then start the socket manager
        bm.start()

    def process_message_15m(self, msg):
        msg['timestamp'] = time.time()
        insert = self.db.btc_15m.insert_one(msg).inserted_id
        print("Data 15M: {}".format(msg['k']['c']))


if __name__ == '__main__':
    bottrading = RegressionMA()
    bottrading.start_socket_5m()
    bottrading.start_socket_1h()
    bottrading.start_socket_15m()
