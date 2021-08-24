import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from talib import EMA, stream, MACD
from bson.objectid import ObjectId
from utils import ST, logger
from pymongo import MongoClient


class AutoOrder:
    # display data on the MetaTrader 5 package
    logger.info("MetaTrader5 package author: ", mt5.__author__)
    logger.info("MetaTrader5 package version: ", mt5.__version__)
    lot = 0.1
    position_id = None
    # order_list = []
    client = MongoClient()
    db = client.stockprice
    # default_time_frame = mt5.TIMEFRAME_H1
    default_time_frame = mt5.TIMEFRAME_M15

    def __init__(self):
        # establish connection to the MetaTrader 5 terminal
        if not mt5.initialize():
            logger.info("initialize() failed, error code =", mt5.last_error())
            quit()

        # orders = mt5.positions_get(symbol='USDCAD')
        # print(orders)
        # if os.path.islfile("order.json"):
        #     with open("order.json") as position_file:
        #         self.order_list = json.load(position_file)
        # self.sell_order('AUDUSD', 50, 0.5)
        # self.close_order('USDCAD')  # type 0  Buy
        # self.close_order('AUDUSD')  # type 1  Sell

    @staticmethod
    def heikin_ashi(df):
        heikin_ashi_df = pd.DataFrame(index=df.index.values, columns=['open', 'high', 'low', 'close'])

        heikin_ashi_df['close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        for i in range(len(df)):
            if i == 0:
                heikin_ashi_df.iat[0, 0] = df['open'].iloc[0]
            else:
                heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i - 1, 0] + heikin_ashi_df.iat[i - 1, 3]) / 2

        heikin_ashi_df['high'] = heikin_ashi_df.loc[:, ['open', 'close']].join(df['high']).max(axis=1)
        heikin_ashi_df['low'] = heikin_ashi_df.loc[:, ['open', 'close']].join(df['low']).min(axis=1)
        return heikin_ashi_df

    def get_frames(self, symbol):
        logger.info(f"Generate super trend for {symbol}")
        self.check_symbol(symbol)
        rates = mt5.copy_rates_from_pos(symbol, self.default_time_frame, 0, 300)
        # create DataFrame out of the obtained data
        rates_frame = pd.DataFrame(rates)
        # rates_frame['time'] = pd.to_datetime(rates_frame['time'], unit='s')
        df = self.heikin_ashi(rates_frame)
        df['EMA_200'] = EMA(df.close, timeperiod=200)
        df['EMA_50'] = EMA(df.close, timeperiod=50)
        df['EMA_20'] = EMA(df.close, timeperiod=20)
        df['MACD'], df['SIGNAL'], df['HIST'] = MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)
        df = df.dropna().reset_index(drop=True)
        df = ST(df, f=1, n=12)
        df = ST(df, f=3, n=10)

        # trend conditional
        conditions = [
            (df['SuperTrend310'] > df['close']) & (df['SuperTrend112'] > df['close']) & (
                        df['HIST'] < df['HIST'].shift(3)),
            (df['SuperTrend310'] < df['close']) & (df['SuperTrend112'] < df['close']) & (
                        df['HIST'] > df['HIST'].shift(3))
        ]
        values = ['Sell', 'Buy']
        df['Trend'] = np.select(conditions, values)
        return df

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
            logger.info(symbol, "not found, can not call order_check()")
            mt5.shutdown()
            quit()

        # if the symbol is unavailable in MarketWatch, add it
        if not symbol_info.visible:
            logger.info(symbol, "is not visible, trying to switch on")
            if not mt5.symbol_select(symbol, True):
                logger.info("symbol_select({}}) failed, exit", symbol)
                mt5.shutdown()
                quit()

    def buy_order(self, symbol, lot, sl):
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
            "sl": sl,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # send a trading request
        result = mt5.order_send(request)
        # logger.info("order_send done, ", result)
        # self.save_frame(request)
        # check the execution result
        logger.info("order_send(): by {} {} lots at {} with deviation={} points".format(symbol, self.lot, price, deviation))
        logger.info(result)
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

    def sell_order(self, symbol, lot, sl):
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
            "sl": sl,
            "deviation": deviation,
            "magic": 234000,
            "comment": "Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # send a trading request
        result = mt5.order_send(request)
        # logger.info(f"order_send done {result}")
        # self.save_frame(request)
        # check the execution result
        logger.info("order_send(): by {} {} lots at {} with deviation={} points".format(symbol, self.lot, price, deviation))
        logger.info(result)
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

    def close_order(self, symbol):
        # create a close request
        logger.info("Close order")
        # display data on active orders on GBPUSD
        orders = mt5.positions_get(symbol=symbol)
        if orders is None:
            logger.info(f"No orders on {symbol}, error code={mt5.last_error()}")
        else:
            logger.info(f"Total orders on {symbol}: {len(orders)}")
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
                    logger.info("Can not close order")
                    return

                # send a trading request
                result = mt5.order_send(request)
                # check the execution result
                logger.info(
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

    def modify_stoploss(self):
        positions = mt5.positions_get()
        if len(positions) > 0:
            # print("Total positions",":",len(positions))
            # display all active orders
            for position in positions:
                symbol = position.symbol
                price_open = position.price_open
                price_current = position.price_current
                lot = position.volume
                prev_sl = position.sl
                position_id = position.identifier
                profit = position.profit
                diff = 0
                pip_profit = 0
                ptype = None
                if position.type == 0:
                    ptype = "Buy"
                    diff = price_current - price_open
                    if diff != 0:
                        pip_profit = (diff * profit * lot) / diff
                elif position.type == 1:
                    ptype = "Sell"
                    diff = price_open - price_current
                    if diff != 0:
                        pip_profit = diff * profit * lot / diff
                # print("id:", position.identifier, ptype, position.profit, position.volume)
                sticks = mt5.copy_rates_from_pos(symbol, self.default_time_frame, 0, 70)
                sticks_frame = pd.DataFrame(sticks)
                # sticks_frame['time'] = pd.to_datetime(sticks_frame['time'], unit='s')
                # stick_time = str(sticks_frame['time'].iloc[-1])
                atr_real = stream.ATR(sticks_frame.high, sticks_frame.low, sticks_frame.close, timeperiod=14)
                if pip_profit != 0:
                    price_per_pip = profit * lot / pip_profit

                if ptype == "Sell":  # if ordertype sell
                    sl = sticks_frame.high.iloc[-2] + atr_real
                    # pip_sl = price_open - price_per_pip
                    # logger.info(f"atr_sl {sl} pip_sl {pip_sl}")
                    # if pip_profit > 4 and pip_sl < sl:
                    #     sl = pip_sl

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
                        logger.info("SL SELL update sent on position #{}: {} {} lots".format(position_id, symbol, lot))
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            logger.info("order_send failed, retcode={}".format(result.retcode))
                            logger.info("result", result)
                        else:
                            logger.info("position #{} SL Updated, {}".format(position_id, result))
                elif ptype == 'Buy':  # if ordertype buy
                    sl = sticks_frame.low.iloc[-2] - atr_real
                    # pip_sl = price_open + price_per_pip
                    # logger.info(f"atr_sl {sl} pip_sl {pip_sl}")
                    # if pip_profit > 4 and pip_sl > sl:
                    #     sl = pip_sl

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
                        logger.info("SL BUY update sent on position #{}: {} {} lots".format(position_id, symbol, lot))
                        if result.retcode != mt5.TRADE_RETCODE_DONE:
                            logger.info("order_send failed, retcode={}".format(result.retcode))
                            logger.info("result", result)
                        else:
                            logger.info("position #{} SL Updated, {}".format(position_id, result))

    @staticmethod
    def check_order_exist(order_symbol: str, order_type: str) -> bool:
        """
        order_symbol: str required
            example: Bitcoin
        order_type: str require
            example: Buy

        return:
            True if exist order
            False if not exist
        """
        positions = mt5.positions_get()
        if len(positions) > 0:
            logger.info(f"Total positions: {len(positions)}")
            # display all active orders
            for position in positions:
                symbol = position.symbol
                position_type = None
                if position.type == 0:
                    position_type = "Buy"
                if position.type == 1:
                    position_type = "Sell"

                if symbol == order_symbol and order_type == position_type:
                    logger.info(f"{order_symbol} {order_type} exist")
                    return True
        return False
