import os
import json
import pandas as pd
import MetaTrader5 as mt5
from talib._ta_lib import EMA, ADX
from pymongo import MongoClient
from bson.objectid import ObjectId


class AutoOrder():
    # display data on the MetaTrader 5 package
    print("MetaTrader5 package author: ", mt5.__author__)
    print("MetaTrader5 package version: ", mt5.__version__)
    lot = 0.1
    position_id = None
    order_list = []
    client = MongoClient()
    db = client.stockprice

    def __init__(self):
        # establish connection to the MetaTrader 5 terminal
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()

        if os.path.isfile("order.json"):
            with open("order.json") as position_file:
                self.order_list = json.load(position_file)

    def get_ema(self, symbol):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 300)
        # Deinitializing MT5 connection

        # create DataFrame out of the obtained data
        rates_frame = pd.DataFrame(rates)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')

        # display data
        # print("\nDisplay dataframe with data")
        # print(rates_frame)

        # uses close prices (default)
        rates_frame['EMA70'] = EMA(rates_frame.close, timeperiod=70)
        rates_frame['EMA100'] = EMA(rates_frame.close, timeperiod=100)
        rates_frame['DMI'] = ADX(rates_frame.high, rates_frame.low, rates_frame.close)
        return rates_frame['EMA70'].iat[-1], rates_frame['EMA100'].iat[-1], rates_frame['DMI'].iat[-1], rates_frame['close'].iat[-1]

    def save_frame(self, request):
        """
        Save frame
        """
        symbol = request['symbol']
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
        # Deinitializing MT5 connection

        # create DataFrame out of the obtained data
        rates_frame = pd.DataFrame(rates)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        request['frame'] = rates_frame
        # display data
        # print("\nDisplay dataframe with data")
        # print(rates_frame)

        # uses close prices (default)
        # rates_frame['EMA70'] = EMA(rates_frame.close, timeperiod=70)
        # rates_frame['EMA100'] = EMA(rates_frame.close, timeperiod=100)
        # rates_frame['DMI'] = ADX(rates_frame.high, rates_frame.low, rates_frame.close)
        results = self.db.posts.insert_one(request).inserted_id

    def check_symbol(self, symbol):
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(symbol, "not found, can not call order_check()")
            mt5.shutdown()
            quit()

        # if the symbol is unavailable in MarketWatch, add it
        if not symbol_info.visible:
            print(symbol, "is not visible, trying to switch on")
            if not mt5.symbol_select(symbol, True):
                print("symbol_select({}}) failed, exit", symbol)
                mt5.shutdown()
                quit()

    def buy_order(self, symbol, tp1, tp2):

        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).ask
        deviation = 20
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": price-tp2,
            "tp": price+tp2,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # send a trading request
        result = mt5.order_send(request)
        print(result)
        # check the execution result
        print("order_send(): by {} {} lots at {} with deviation={} points".format(symbol, self.lot, price, deviation))
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("2. order_send failed, retcode={}".format(result.retcode))
            # request the result as a dictionary and display it element by element
            result_dict = result._asdict()
            for field in result_dict.keys():
                print("   {}={}".format(field, result_dict[field]))
                # if this is a trading request structure, display it element by element as well
                if field == "request":
                    traderequest_dict = result_dict[field]._asdict()
                    for tradereq_filed in traderequest_dict:
                        print("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))
            print("shutdown() and quit")
            mt5.shutdown()
            quit()

        request['tp1'] = price+tp1
        request['cross_tp1'] = False
        self.order_list.append(request)
        self.save_frame(request)
        with open("order.json", 'w') as position_file:
            json.dump(self.order_list, position_file, indent=4)

        print("order_send done, ", result)

    def sell_order(self, symbol, tp1, tp2):

        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).bid
        deviation = 20
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": self.lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price+tp2,
            "tp": price-tp2,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # send a trading request
        result = mt5.order_send(request)
        # check the execution result
        print("order_send(): by {} {} lots at {} with deviation={} points".format(symbol, self.lot, price, deviation))
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print("2. order_send failed, retcode={}".format(result.retcode))
            # request the result as a dictionary and display it element by element
            result_dict = result._asdict()
            for field in result_dict.keys():
                print("   {}={}".format(field, result_dict[field]))
                # if this is a trading request structure, display it element by element as well
                if field == "request":
                    traderequest_dict = result_dict[field]._asdict()
                    for tradereq_filed in traderequest_dict:
                        print("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))
            print("shutdown() and quit")
            mt5.shutdown()
            quit()

        request['tp1'] = price-tp1
        request['cross_tp1'] = False
        self.order_list.append(request)
        self.save_frame(request)
        with open("order.json", 'w') as position_file:
            json.dump(self.order_list, position_file, indent=4)

        print("order_send done, ", result)

    def close_order(self, symbol, percent=1.0):
        # create a close request
        print("Close order")
        # display data on active orders on GBPUSD
        orders = mt5.positions_get(symbol=symbol)
        if orders is None:
            print(f"No orders on {symbol}, error code={mt5.last_error()}")
        else:
            print(f"Total orders on {symbol}: {len(orders)}")
            # display all active orders
            for order in orders:
                position_id = order.identifier
                comment = order.comment
                if comment == 'Sell':
                    price = mt5.symbol_info_tick(symbol).bid
                    deviation = 20
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": self.lot/percent,
                        "type": mt5.ORDER_TYPE_BUY,
                        "position": position_id,
                        "price": price,
                        "deviation": deviation,
                        "magic": 234000,
                        "comment": "python script close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                elif comment == 'Buy':
                    price = mt5.symbol_info_tick(symbol).ask
                    deviation = 20
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": self.lot/percent,
                        "type": mt5.ORDER_TYPE_SELL,
                        "position": position_id,
                        "price": price,
                        "deviation": deviation,
                        "magic": 234000,
                        "comment": "python script close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                else:
                    print("Can not close order")
                    return

                # send a trading request
                result = mt5.order_send(request)
                # check the execution result
                print(
                    "close position #{}: sell {} {} lots at {} with deviation={} points".format(position_id, symbol,
                                                                                                self.lot/percent, price,
                                                                                                deviation))
                if result:
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        print("order_send failed, retcode={}".format(result.retcode))
                        print("   result", result)
                    else:
                        print("position #{} closed, {}".format(position_id, result))
                        # request the result as a dictionary and display it element by element
                        result_dict = result._asdict()
                        for field in result_dict.keys():
                            print("   {}={}".format(field, result_dict[field]))
                            # if this is a trading request structure, display it element by element as well
                            if field == "request":
                                traderequest_dict = result_dict[field]._asdict()
                                for tradereq_filed in traderequest_dict:
                                    print("       traderequest: {}={}".format(tradereq_filed,
                                                                              traderequest_dict[tradereq_filed]))

    def check_take_profit(self):
        for idx, order in enumerate(self.order_list):
            symbol = order['symbol']
            tp1 = order['tp1']
            action = order['type']
            bid = mt5.symbol_info_tick(symbol).bid
            ask = mt5.symbol_info_tick(symbol).ask
            if action == mt5.ORDER_TYPE_SELL and bid < tp1:
                self.close_order(symbol=symbol, percent=0.5)
                self.order_list.pop(idx)
                with open("order.json", 'w') as position_file:
                    json.dump(self.order_list, position_file, indent=4)
            if action == mt5.ORDER_TYPE_BUY and ask > tp1:
                self.close_order(symbol=symbol, percent=0.5)
                self.order_list.pop(idx)
                with open("order.json", 'w') as position_file:
                    json.dump(self.order_list, position_file, indent=4)