import decimal
import os
import json
import pandas as pd
import MetaTrader5 as mt5
from talib import stream
from talib._ta_lib import EMA, ADX
from pymongo import MongoClient
from bson.objectid import ObjectId


class AutoOrder:
    # display data on the MetaTrader 5 package
    print("MetaTrader5 package author: ", mt5.__author__)
    print("MetaTrader5 package version: ", mt5.__version__)
    lot = 0.1
    position_id = None
    # order_list = []
    client = MongoClient()
    db = client.stockprice

    def __init__(self):
        # establish connection to the MetaTrader 5 terminal
        if not mt5.initialize():
            print("initialize() failed, error code =", mt5.last_error())
            quit()

        # orders = mt5.positions_get(symbol='USDCAD')
        # print(orders)
        # if os.path.islfile("order.json"):
        #     with open("order.json") as position_file:
        #         self.order_list = json.load(position_file)
        # self.sell_order('AUDUSD', 50, 0.5)
        # self.close_order('USDCAD')  # type 0  Buy
        # self.close_order('AUDUSD')  # type 1  Sell

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
        process_data = request.copy()
        # process_data['tp_2'] = request_2['tp']
        symbol = process_data['symbol']
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
        # Deinitializing MT5 connection

        # create DataFrame out of the obtained data
        rates_frame = pd.DataFrame(rates)
        rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        process_data['frame'] = rates_frame.to_numpy().tolist()
        # display data
        # print("\nDisplay dataframe with data")
        # print(rates_frame)

        # uses close prices (default)
        # rates_frame['EMA70'] = EMA(rates_frame.close, timeperiod=70)
        # rates_frame['EMA100'] = EMA(rates_frame.close, timeperiod=100)
        # rates_frame['DMI'] = ADX(rates_frame.high, rates_frame.low, rates_frame.close)
        process_data['_id'] = str(ObjectId())
        results = self.db.posts.insert_one(process_data)

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

    def buy_order(self, symbol, tp, lot):
        self.check_symbol(symbol)
        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).ask
        deviation = 20
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "sl": price - tp * point / 10,
            "tp": price + tp * point,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # send a trading request
        result = mt5.order_send(request)
        print("order_send done, ", result)
        self.save_frame(request)
        # check the execution result
        print("order_send(): by {} {} lots at {} with deviation={} points".format(symbol, self.lot, price, deviation))

        # if result.retcode != mt5.TRADE_RETCODE_DONE:
        #     print("2. order_send failed, retcode={}".format(result.retcode))
        #     # request the result as a dictionary and display it element by element
        #     result_dict = result._asdict()
        #     for field in result_dict.keys():
        #         print("   {}={}".format(field, result_dict[field]))
        #         # if this is a trading request structure, display it element by element as well
        #         if field == "request":
        #             traderequest_dict = result_dict[field]._asdict()
        #             for tradereq_filed in traderequest_dict:
        #                 print("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))
        #     print("shutdown() and quit")
        #     mt5.shutdown()
        #     quit()

    def sell_order(self, symbol, tp, lot):
        self.check_symbol(symbol)
        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).bid
        deviation = 20
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price + tp * point / 10,
            "tp": price - tp * point,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # send a trading request
        result = mt5.order_send(request)
        print("order_send done, ", result)
        self.save_frame(request)
        # check the execution result
        print("order_send(): by {} {} lots at {} with deviation={} points".format(symbol, self.lot, price, deviation))

        # if result.retcode != mt5.TRADE_RETCODE_DONE:
        #     print("2. order_send failed, retcode={}".format(result.retcode))
        #     # request the result as a dictionary and display it element by element
        #     result_dict = result._asdict()
        #     for field in result_dict.keys():
        #         print("   {}={}".format(field, result_dict[field]))
        #         # if this is a trading request structure, display it element by element as well
        #         if field == "request":
        #             traderequest_dict = result_dict[field]._asdict()
        #             for tradereq_filed in traderequest_dict:
        #                 print("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))
        #     print("shutdown() and quit")
        #     mt5.shutdown()
        #     quit()

    def close_order(self, symbol):
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
                order_type = order.type
                if order_type == 1:
                    price = mt5.symbol_info_tick(symbol).bid
                    deviation = 20
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": order.volume,
                        "type": mt5.ORDER_TYPE_BUY,
                        "position": position_id,
                        "price": price,
                        "deviation": deviation,
                        "magic": 234000,
                        "comment": "python script close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                elif order_type == 0:
                    price = mt5.symbol_info_tick(symbol).ask
                    deviation = 20
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": order.volume,
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
                                                                                                self.lot, price,
                                                                                                deviation))
                # if result:
                #     if result.retcode != mt5.TRADE_RETCODE_DONE:
                #         print("order_send failed, retcode={}".format(result.retcode))
                #         print("   result", result)
                #     else:
                #         print("position #{} closed, {}".format(position_id, result))
                #         # request the result as a dictionary and display it element by element
                #         result_dict = result._asdict()
                #         for field in result_dict.keys():
                #             print("   {}={}".format(field, result_dict[field]))
                #             # if this is a trading request structure, display it element by element as well
                #             if field == "request":
                #                 traderequest_dict = result_dict[field]._asdict()
                #                 for tradereq_filed in traderequest_dict:
                #                     print("       traderequest: {}={}".format(tradereq_filed,
                #                                                               traderequest_dict[tradereq_filed]))

    @staticmethod
    def modify_stoploss():
        positions = mt5.positions_get()
        if len(positions) > 0:
            # print("Total positions",":",len(positions))
            # display all active orders
            for position in positions:
                symbol = position.symbol
                if position.type == 0:
                    ptype = "Buy"
                elif position.type == 1:
                    ptype = "Sell"
                else:
                    ptype = None
                # print("id:", position.identifier, ptype, position.profit, position.volume)
                sticks = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 0, 50)
                sticks_frame = pd.DataFrame(sticks)
                sticks_frame['time'] = pd.to_datetime(sticks_frame['time'], unit='s')
                # stick_time = str(sticks_frame['time'].iloc[-1])
                atr_real = stream.ATR(sticks_frame.high, sticks_frame.low, sticks_frame.close, timeperiod=14)
                lot = position.volume
                # stop loss
                prev_sl = position.sl
                position_id = position.identifier
                if ptype == "Sell":  # if ordertype sell
                    sl = sticks_frame["high"].iloc[-2] + atr_real
                    sl = round(sl, len(str(prev_sl).split('.')[1]))
                    if prev_sl < sl or prev_sl == 0:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position_id,
                            "sl": sl,
                            "comment": f"python trailing sl",
                            "deviation": 20,
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_RETURN,
                            "volume": position.volume
                        }
                        # send a trading request
                        result = mt5.order_send(request)
                        # check the execution result
                        print("3. SL SELL update sent on position #{}: {} {} lots".format(position_id, symbol, lot));
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print("4. order_send failed, retcode={}".format(result.retcode))
                            print("   result", result)
                        else:
                            print("4. position #{} SL Updated, {}".format(position_id, result))
                elif ptype == 'Buy':  # if ordertype buy
                    sl = sticks_frame["low"].iloc[-2] - atr_real
                    sl = round(sl, len(str(prev_sl).split('.')[1]))
                    if prev_sl > sl or prev_sl == 0:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "position": position_id,
                            "sl": sl,
                            "comment": f"python trailing sl",
                            "deviation": 20,
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": mt5.ORDER_FILLING_RETURN,
                            "volume": position.volume
                        }
                        # send a trading request
                        result = mt5.order_send(request)
                        # check the execution result
                        print("3. SL BUY update sent on position #{}: {} {} lots".format(position_id, symbol, lot))
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            print("4. order_send failed, retcode={}".format(result.retcode))
                            print("   result", result)
                        else:
                            print("4. position #{} SL Updated, {}".format(position_id, result))
