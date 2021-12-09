import math
import time

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import pytz
import uuid
from talib import EMA, stream, MACD, ADX, BBANDS, ATR, NATR, MACDEXT
from datetime import datetime
from utils import logger, Supertrend, trend_table
import talib


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
        #     print(rates_frame)
        # create DataFrame out of the obtained data
        df = pd.DataFrame(rates_frame, columns=['time', 'open', 'high', 'low', 'close'])
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df = df.drop(['time'], axis=1)
        df = df.reindex(columns=['Date', 'open', 'high', 'low', 'close'])

        # convert to float to avoid sai so.
        df.open = df.open.astype(float)
        df.high = df.high.astype(float)
        df.low = df.low.astype(float)
        df.close = df.close.astype(float)

        # calculate indicator
        atr_multiplier = 3.0
        atr_period = 10
        supertrend = Supertrend(df, atr_period, atr_multiplier)
        df = df.join(supertrend)
        df['MACD'], df['SIGNAL'], df['HIST'] = MACD(df.close)
        df['ADX'] = ADX(df.high, df.low, df.close)
        # selection trend
        conditions = [
            (df['Supertrend10'] == True) & (df['HIST'] > df['HIST'].shift(3)),
            (df['Supertrend10'] == False) & (df['HIST'] < df['HIST'].shift(3))
        ]
        values = ['Buy', 'Sell']
        df['Trend'] = np.select(conditions, values)
        logger.info(
            f"{timeframe} {df.iloc[-1].Date} Supertrend10 :{df.Supertrend10.iat[-1]} HIST: {df.HIST.iat[-1]} PREV HIST: {df.HIST.iat[-2]}")
        logger.info(
            f"{timeframe} {df.iloc[-1].Date} ADX: {df.ADX.iat[-1]} PREV ADX: {df.ADX.iat[-2]}")
        logger.info(
            f"{timeframe} {df.iloc[-1].Date} Open {df.iloc[-1].open} High {df.iloc[-1].high} Low {df.iloc[-1].low} Close {df.iloc[-1].close}")
        current_trend = df.Trend.iat[-1]
        trend_table.insert_one(
            {
                "_id": str(uuid.uuid4()),
                "timeframe": timeframe,
                "symbol": symbol,
                "current_trend": current_trend,
                "created_date": int(time.time()),
                "open": df.iloc[-1].open,
                "high": df.iloc[-1].high,
                "low": df.iloc[-1].low,
                "close": df.iloc[-1].close
            }
        )
        return df.close.iat[-1], current_trend, str(df.Date.iat[-1])

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
        deviation = 40
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
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
        if result:
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error("2. order_send failed, retcode={}".format(result.retcode))
                # request the result as a dictionary and display it element by element
                result_dict = result._asdict()
                logger.error(f"ERROR {result_dict}")
                # for field in result_dict.keys():
                #     logger.error("   {}={}".format(field, result_dict[field]))
                #     # if this is a trading request structure, display it element by element as well
                #     if field == "request":
                #         traderequest_dict = result_dict[field]._asdict()
                #         for tradereq_filed in traderequest_dict:
                #             logger.error("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))
            else:
                logger.info("order_send(): by {} {} lots at {} with deviation={} points".format(symbol, self.lot, price,
                                                                                                deviation))
                logger.info(result)
        else:
            logger.error(f"Can not send order for {symbol}")

    def sell_order(self, symbol, lot, sl, tp):
        self.check_symbol(symbol)
        point = mt5.symbol_info(symbol).point
        price = mt5.symbol_info_tick(symbol).bid
        deviation = 20
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(lot),
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
        logger.info(f"order_send done {result}")
        # self.save_frame(request)
        # check the execution result
        if result:
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error("2. order_send failed, retcode={}".format(result.retcode))
                # request the result as a dictionary and display it element by element
                result_dict = result._asdict()
                logger.error(f"ERROR {result_dict}")
                # for field in result_dict.keys():
                    # if this is a trading request structure, display it element by element as well
                    # if field == "request":
                    #     traderequest_dict = result_dict[field]._asdict()
                    #     for tradereq_filed in traderequest_dict:
                    #         logger.error("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))
            else:
                logger.info("order_send(): by {} {} lots at {} with deviation={} points".format(symbol, self.lot, price,
                                                                                                deviation))
                logger.info(result)
        else:
            logger.error(f"Can not send order for {symbol}")

    def close_order(self, symbol):
        # create a close request
        logger.info("close order")
        # display data on active orders on GBPUSD
        orders = mt5.positions_get(symbol=symbol)
        account_info = mt5.account_info()._asdict()
        balance = account_info.get("balance")
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
                        "comment": f"close buy order",
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
                        "comment": f"close sell order",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                else:
                    logger.info("Can not close order")
                    return

                # send a trading request
                result = mt5.order_send(request)
                # check the execution result
                if result:
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        logger.error("order_send failed, retcode={}".format(result.retcode))
                        logger.error(f"   result {result}")
                    else:
                        logger.info("close position #{}: symbol {} profit {}".format(position_id, symbol, profit))
                        # request the result as a dictionary and display it element by element
                        result_dict = result._asdict()
                        for field in result_dict.keys():
                            logger.info("   {}={}".format(field, result_dict[field]))
                            # if this is a trading request structure, display it element by element as well
                            if field == "request":
                                traderequest_dict = result_dict[field]._asdict()
                                for tradereq_filed in traderequest_dict:
                                    logger.info("       traderequest: {}={}".format(tradereq_filed,
                                                                                    traderequest_dict[tradereq_filed]))
                else:
                    logger.error(f"Can not close order for {symbol}")

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
    def check_order_exist(order_symbol: str) -> '':
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
            position_type = '0'
            if position.type == 0:
                position_type = "Buy"
            if position.type == 1:
                position_type = "Sell"

            # if order_type == position_type:
            #     logger.info(f"{order_symbol} {order_type} exist")
            return position_type
        return '0'

    @staticmethod
    def ichimoku_cloud(timeframe, symbol):
        rates_frame = mt5.copy_rates_from_pos(symbol, timeframe, 0, 300)
        #     print(rates_frame)
        # create DataFrame out of the obtained data
        df = pd.DataFrame(rates_frame, columns=['time', 'open', 'high', 'low', 'close'])
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df = df.drop(['time'], axis=1)
        df = df.reindex(columns=['Date', 'open', 'high', 'low', 'close'])

        # convert to float to avoid sai so.
        df.open = df.open.astype(float)
        df.high = df.high.astype(float)
        df.low = df.low.astype(float)
        df.close = df.close.astype(float)
        df['MACD'], df['SIGNAL'], df['HIST'] = talib.MACD(df.close)
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2))
        nine_period_high = df['high'].rolling(window=9).max()
        nine_period_low = df['low'].rolling(window=9).min()
        df['tenkan_sen'] = (nine_period_high + nine_period_low) / 2
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2))
        period26_high = df['high'].rolling(window=26).max()
        period26_low = df['low'].rolling(window=26).min()
        df['kijun_sen'] = (period26_high + period26_low) / 2
        # Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2))
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2))
        period52_high = df['high'].rolling(window=52).max()
        period52_low = df['low'].rolling(window=52).min()
        df['senkou_span_b'] = ((period52_high + period52_low) / 2).shift(26)
        # The most current closing price plotted 26 time periods behind (optional)
        df['chikou_span'] = df['close'].shift(-26)
        # create a quick plot of the results to see what we have created
        # df.drop(['Date'], axis=1).plot(figsize=(15,8))
        conditions = [
            (df['tenkan_sen'] > df['kijun_sen']) & (df['close'] > df['senkou_span_a']) & (
                        df['close'] > df['senkou_span_b']) & (df['HIST'] > df['HIST'].shift()) & (
                        df['HIST'].shift() > df['HIST'].shift(2)),
            (df['tenkan_sen'] < df['kijun_sen']) & (df['close'] < df['senkou_span_a']) & (
                        df['close'] < df['senkou_span_b']) & (df['HIST'] < df['HIST'].shift()) & (
                        df['HIST'].shift() < df['HIST'].shift(2))
        ]
        values = ['Buy', 'Sell']
        df['Trend'] = np.select(conditions, values)
        logger.info(f"{df['Date'].iat[-1]} {symbol} tenkan_sen: {df['tenkan_sen'].iat[-1]}  kijun_sen: {df['kijun_sen'].iat[-1]} senkou_span_a: {df['senkou_span_a'].iat[-1]} senkou_span_b: {df['senkou_span_b'].iat[-1]} histogram: {df['HIST'].iat[-1]}")
        return str(df.Date.iat[-1]), df.Trend.iat[-1], df.close.iat[-1]
