import threading
import time
import logging
from binance.enums import KLINE_INTERVAL_1HOUR, KLINE_INTERVAL_5MINUTE, KLINE_INTERVAL_15MINUTE, KLINE_INTERVAL_1MINUTE
from pymongo import MongoClient
from bson import ObjectId
from binance.client import Client
from binance.websockets import BinanceSocketManager

logging.basicConfig(filename='log/crawler.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class RegressionMA:
    def __init__(self, market, interval):
        self.client = MongoClient(username='admin', password='jkahsduk12387a89sdjk@#', authSource='admin')
        self.db = self.client.crypto
        self.api_key = "9Hj6HLNNMGgkqj6ngouMZD1kjIbUb6RZmIpW5HLiZjtDT5gwhXAzc20szOKyQ3HW"
        self.api_secret = "ioD0XICp0cFE99VVql5nuxiCJEb6GK8mh08NYnSYdIUfkiotd1SZqLTQsjFvrXwk"
        self.market = market
        self.interval = interval

    def start_socket(self):
        binace_client = Client(self.api_key, self.api_secret)
        # start any sockets here, i.e a trade socket
        bm = BinanceSocketManager(binace_client)
        conn_key = bm.start_kline_socket(self.market, self.process_message, interval=self.interval)
        # then start the socket manager
        bm.start()

    def process_message(self, msg):
        msg['k']['timestamp'] = time.time()
        msg['_id'] = str(ObjectId())
        market = msg['s']
        collection = '{}_{}'.format(market, self.interval)
        insert = self.db[collection].insert(msg)
        txt = "{} Data {}: {}".format(market, self.interval, msg['k']['c'])
        print(txt)
        logging.warning(txt)


if __name__ == '__main__':
    symbols_list = ['BTCUSDT']
    symbol = 'BTCUSDT'
    # crawler_1 = RegressionMA(symbol, KLINE_INTERVAL_1HOUR)
    # crawler_1.start_socket()
    #
    crawler_2 = RegressionMA(symbol, KLINE_INTERVAL_15MINUTE)
    crawler_2.start_socket()
    #
    # crawler_3 = RegressionMA(symbol, KLINE_INTERVAL_5MINUTE)
    # crawler_3.start_socket()

    crawler_3 = RegressionMA(symbol, KLINE_INTERVAL_1MINUTE)
    crawler_3.start_socket()
