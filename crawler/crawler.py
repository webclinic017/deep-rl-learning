import time
import json
import schedule
import requests
import datetime
import socketio
from bs4 import BeautifulSoup
from bson import ObjectId
from pymongo import MongoClient


# client = MongoClient(host='66.42.37.163', username="admin", password="jkahsduk12387a89sdjk@#", authSource='admin')
# db = client.forex
sio = socketio.Client()
sio.connect('wss://fxpricing.com', transports='websocket')

# http_url = "https://fcsapi.com/api-v2/forex/history"
# "With demo API_KEY, only these 'EUR\/USD,USD\/JPY,GBP\/CHF,JPY\/CHF,BTC\/USD,ETH\/USD,XRP\/USD,USD, BTC'
# sending get request and saving the response as response object
# list_symbol = [r'EUR/USD', r'USD/JPY', r'GBP/CHF', r'JPY/CHF', r'BTC/USD', r'ETH/USD', r'XRP/USD']
# list_symbol = [r'EUR/USD']
# for symbol in list_symbol:
#     PARAMS = {
#         'symbol': symbol,
#         'period': '1h',
#         'access_key': 'API_KEY',
#         'from': '2019-10-01',
#         'to': '2020-04-17',
#         'output': 'json'
#     }
#
#     r = requests.get(url=http_url, params=PARAMS)
#     if r.ok:
#         # extracting data in json format
#         soup = BeautifulSoup(r.text, 'html.parser')
#         # print(soup.prettify())
#         json_res = soup.find('pre').text
#         response_body = json.loads(json_res)['response']
#         info = json.loads(json_res)['info']
#
#         oa_summary = response_body['oa_summary']
#         server_time = info['server_time']
#
#         print("{} | {} | Summary: {} ".format(server_time, symbol, oa_summary))


def heartbeat():
    sio.emit('heartbeat', 'API_KEY')
    print("send heartbeat")


schedule.every().hour.do(heartbeat)


@sio.on('connect')
def connected():
    print("connect")
    sio.emit('heartbeat', 'API_KEY')
    sio.emit('real_time_join', '1,1984')


@sio.on('disconnect')
def socket_re_connection():
    print("reconnect")
    sio = socketio.Client()
    sio.connect('wss://fxpricing.com', transports='websocket')
    sio.emit('heartbeat', 'API_KEY')
    sio.emit('real_time_join', '1,1984')


@sio.on('data_received')
def on_data_received(data):
    _id = ObjectId()
    data['_id'] = str(_id)
    data['timestamp'] = time.time()
    now = datetime.datetime.now()
    current_hour = now.hour
    if now.minute == now.second == 0:
        # insert new sample
        _open = _high = _low = _close = 0
    # if 'XAU-USD' in data.keys():
    #     inserted = db.xauusd.insert_one(data)
    # else:
    #     inserted = db.eurusd.insert_one(data)

    schedule.run_pending()
    print(data)


