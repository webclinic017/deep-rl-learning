import time
from binance_f import RequestClient
from binance_f.constant.test import *
from binance_f.base.printobject import *
from binance_f.model.constant import *

request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)
# result = request_client.post_order(symbol="BTCUSDT", side=OrderSide.BUY, ordertype=OrderType.MARKET, quantity=0.001)
# PrintMix.print_data(result)
# result = request_client.post_order(symbol="BTCUSDT", side=OrderSide.SELL, ordertype=OrderType.MARKET, quantity=0.001)
# result = request_client.cancel_order(symbol="BTCUSDT", orderId=2965110746)
# PrintMix.print_data(result)
# clientOrderId:rfZtzg87GmT0A78fKankeX
# orderId:2965110746

result = request_client.get_balance()
print(result[0].__dict__)
