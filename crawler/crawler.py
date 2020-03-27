import time

import socketio
from bson import ObjectId
from pymongo import MongoClient

client = MongoClient()
db = client.forex
sio = socketio.Client()
sio.connect('wss://fxpricing.com', transports='websocket')


@sio.on('connect')
def connected():
    print("connect")
    sio.emit('heartbeat', 'API_KEY')
    sio.emit('real_time_join', '1,1984')


@sio.on('disconnect')
def socket_re_connection():
    print("reconnect")
    sio.connect('wss://fxpricing.com', transports='websocket')
    sio.emit('heartbeat', 'API_KEY')
    sio.emit('real_time_join', '1,1984')


@sio.on('data_received')
def on_data_received(data):
    _id = ObjectId()
    data['_id'] = str(_id)
    data['timestamp'] = time.time()
    inserted = db.eurusd.insert_one(data)
    print(data)
