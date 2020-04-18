import time
import logging
from binance.enums import KLINE_INTERVAL_1HOUR
from pymongo import MongoClient
from bson import ObjectId
from binance.client import Client
from binance.websockets import BinanceSocketManager

logging.basicConfig(filename='log/crawler.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


class RegressionMA:
    def __init__(self):
        self.client = MongoClient(username='admin', password='jkahsduk12387a89sdjk@#', authSource='admin')
        self.db = self.client.crypto
        self.api_key = "9Hj6HLNNMGgkqj6ngouMZD1kjIbUb6RZmIpW5HLiZjtDT5gwhXAzc20szOKyQ3HW"
        self.api_secret = "ioD0XICp0cFE99VVql5nuxiCJEb6GK8mh08NYnSYdIUfkiotd1SZqLTQsjFvrXwk"

    def start_socket(self, market):
        binace_client = Client(self.api_key, self.api_secret)
        # start any sockets here, i.e a trade socket
        bm = BinanceSocketManager(binace_client)
        conn_key = bm.start_kline_socket(market, self.process_message, interval=KLINE_INTERVAL_1HOUR)
        # then start the socket manager
        bm.start()

    def process_message(self, msg):
        msg['k']['timestamp'] = time.time()
        msg['_id'] = str(ObjectId())
        market = msg['s']
        collection = '{}_{}'.format(market, KLINE_INTERVAL_1HOUR)
        insert = self.db[collection].insert(msg)
        txt = "{} Data {}: {}".format(market, KLINE_INTERVAL_1HOUR, msg['k']['c'])
        print(txt)
        logging.warning(txt)


if __name__ == '__main__':
    bottrading = RegressionMA()
    symbols_list = ['BTCUSDT', 'ZEC_BTC', 'DASH_BTC', 'BNB_BTC', 'LINK_BTC', 'XTZ_BTC', 'ONT_BTC', 'XRP_BTC', 'ADA_BTC',
                    'XLM_BTC', 'EOS_BTC', 'NEO_BTC']
    for symbol in symbols_list:
        bottrading.start_socket(symbol)
