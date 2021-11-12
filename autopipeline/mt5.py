import math
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import pytz
from talib import EMA, stream, MACD, ADX, BBANDS, ATR, NATR
from datetime import datetime
from utils import ST, logger


class AutoOrder:
    # display data on the MetaTrader 5 package
    logger.info(f"MetaTrader5 package author: {mt5.__author__}")
    logger.info(f"MetaTrader5 package version: {mt5.__version__}")
    lot = 0.1
    position_id = None
    # order_list = []
    # client = MongoClient()
    # db = client.stockprice
    # default_time_frame = mt5.TIMEFRAME_H1
    default_time_frame = mt5.TIMEFRAME_M15

    def __init__(self):
        # establish connection to the MetaTrader 5 terminal
        if not mt5.initialize():
            logger.info(f"initialize() failed, error code = {mt5.last_error()}", )
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

    @staticmethod
    def get_frames(timeframe, symbol):
        rates_frame = mt5.copy_rates_from_pos(symbol, timeframe, 0, 300)
        # create DataFrame out of the obtained data
        df = pd.DataFrame(rates_frame, columns=['time', 'open', 'high', 'low', 'close'])
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df = df.drop(['time'], axis=1)
        df = df.reindex(columns=['Date', 'open', 'high', 'low', 'close'])
        df['EMA_50'] = df.close.ewm(span=50, adjust=False).mean()
        df['EMA_20'] = df.close.ewm(span=20, adjust=False).mean()
        df['MACD'], df['SIGNAL'], df['HIST'] = MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)
        conditions = [
            (df['close'] > df['EMA_50']) & (df['HIST'] > 0),
            (df['close'] < df['EMA_50']) & (df['HIST'] < 0)
        ]
        values = ['Buy', 'Sell']
        df['Trend'] = np.select(conditions, values)
        return df.close.iat[-1], df.Trend.iat[-1], str(df.Date.iat[-1])

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
        # process_data['_id'] = str(ObjectId())
        # results = self.db.posts.insert_one(process_data)

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

    def buy_order(self, symbol, lot, sl, tp):
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
            "deviation": deviation,
            "magic": 234000,
            "comment": "Buy",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if tp is not None:
            request['tp'] = tp

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
            # mt5.shutdown()
            # quit()

    def sell_order(self, symbol, lot, sl, tp):
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
            "deviation": deviation,
            "magic": 234000,
            "comment": "Sell",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        if tp is not None:
            request['tp'] = tp

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
            # mt5.shutdown()
            # quit()

    def close_order(self, symbol):
        # create a close request
        logger.info("close order")
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
                profit = order.profit
                deviation = 20
                if order_type == 1:
                    price = mt5.symbol_info_tick(symbol).bid
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
                logger.info("close position #{}: symbol {} profit {}".format(position_id, symbol, profit))
                # if result:
                # if result.retcode != mt5.TRADE_RETCODE_DONE:
                #     logger.error("order_send failed, retcode={}".format(result.retcode))
                #     logger.error("   result", result)
                # else:
                #     # logger.error("position #{} closed, {}".format(position_id, result))
                #     # request the result as a dictionary and display it element by element
                #     result_dict = result._asdict()
                #     for field in result_dict.keys():
                #         logger.info("   {}={}".format(field, result_dict[field]))
                #         # if this is a trading request structure, display it element by element as well
                #         if field == "request":
                #             traderequest_dict = result_dict[field]._asdict()
                #             for tradereq_filed in traderequest_dict:
                #                 logger.info("       traderequest: {}={}".format(tradereq_filed,
                #                                                           traderequest_dict[tradereq_filed]))

    @staticmethod
    def modify_stoploss(symbol, sl, atr):
        positions = mt5.positions_get(symbol=symbol)
        # print("Total positions",":",len(positions))
        # display all active orders
        for position in positions:
            symbol = position.symbol
            symbol_info = mt5.symbol_info(symbol)
            price_open = position.price_open
            price_current = position.price_current
            lot = position.volume
            prev_sl = position.sl
            position_id = position.identifier
            profit = position.profit
            comment = position.comment
            diff = 0
            pip_profit = 0
            ptype = None
            if position.type == 0:
                ptype = "Buy"
                diff = price_current - price_open
                # pip_profit = diff / profit
                if diff > atr and price_open > sl:
                    sl = price_open
                if prev_sl > sl:
                    return True
            elif position.type == 1:
                ptype = "Sell"
                diff = price_open - price_current
                # pip_profit = diff / profit
                if diff > atr and price_open < sl:
                    sl = price_open
                if prev_sl < sl:
                    return True

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
                logger.info(f"result: {result}")
            else:
                logger.info("position #{} SL Updated at: {} profit {}".format(position_id, price_open, profit))

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
        positions = mt5.positions_get(symbol=order_symbol)
        # logger.info(f"Total positions: {len(positions)}")
        # display all active orders
        for position in positions:
            symbol = position.symbol
            position_type = None
            if position.type == 0:
                position_type = "Buy"
            if position.type == 1:
                position_type = "Sell"

            if order_type == position_type:
                logger.info(f"{order_symbol} {order_type} exist")
                return True
        return False
