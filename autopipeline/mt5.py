import os
import time
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import uuid

import plotly.graph_objs as go
from pyod.models.hbos import HBOS

from talib import EMA, stream, MACD, ADX, CDLENGULFING, ATR
from datetime import datetime
from utils import logger, Supertrend, trend_table


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
        if sl is not None:
            request['sl'] = sl

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
        if sl is not None:
            request['sl'] = sl

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
    def modify_stoploss(symbol, atr, kijun_sen):
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
            sl = None
            if position.type == 0:
                """Sell Order"""
                ptype = "Buy"
                diff = price_current - price_open
                sl = price_open + atr
                if prev_sl is None and diff > atr:
                    sl = price_open
                if prev_sl and sl and sl < kijun_sen:
                    sl = kijun_sen
                # pip_profit = diff / profit
            elif position.type == 1:
                """Sell Order"""
                ptype = "Sell"
                diff = price_open - price_current
                sl = price_open - atr
                if prev_sl is None and diff > atr:
                    sl = price_open
                if prev_sl and sl and sl > kijun_sen:
                    sl = kijun_sen
            if sl:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "position": position_id,
                    "sl": sl,
                    "comment": f"modify trailing sl",
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
    def plot_chart(df, pivots, timeframe):
        os.makedirs('images', exist_ok=True)
        # Set colours for up and down candles
        INCREASING_COLOR = '#26a69a'
        DECREASING_COLOR = '#ef5350'

        # create list to hold dictionary with data for our first series to plot
        # (which is the candlestick element itself)
        data = [dict(
            type='candlestick',
            open=df.open,
            high=df.high,
            low=df.low,
            close=df.close,
            x=df.Date,
            yaxis='y2',
            name='F',
            increasing=dict(line=dict(color=INCREASING_COLOR)),
            decreasing=dict(line=dict(color=DECREASING_COLOR)),
        )]

        # Create empty dictionary for later use to hold settings and layout options
        layout = dict()

        # create our main chart "Figure" object which consists of data to plot and layout settings
        fig = dict(data=data, layout=layout)

        # Assign various seeting and choices - background colour, range selector etc
        fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
        fig['layout']['xaxis'] = dict(rangeselector=dict(visible=False))
        fig['layout']['yaxis'] = dict(domain=[0, 0.2], showticklabels=False)
        fig['layout']['yaxis2'] = dict(domain=[0.2, 0.8])
        fig['layout']['legend'] = dict(orientation='h', y=0.9, x=0.3, yanchor='bottom')
        fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)

        # Populate the "rangeselector" object with necessary settings
        rangeselector = dict(
            visible=True,
            x=0, y=0.9,
            bgcolor='rgba(150, 200, 250, 0.4)',
            font=dict(size=13),
            buttons=list([
                dict(count=1,
                     label='reset',
                     step='all'),
                dict(count=1,
                     label='1yr',
                     step='year',
                     stepmode='backward'),
                dict(count=3,
                     label='3 mo',
                     step='month',
                     stepmode='backward'),
                dict(count=1,
                     label='1 mo',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ]))

        fig['layout']['xaxis']['rangeselector'] = rangeselector

        # Append the Ichimoku elements to the plot
        fig['data'].append(dict(x=df.Date, y=df['tenkan_sen'], type='scatter', mode='lines',
                                line=dict(width=1),
                                marker=dict(color='#33BDFF'),
                                yaxis='y2', name='tenkan_sen'))

        fig['data'].append(dict(x=df.Date, y=df['kijun_sen'], type='scatter', mode='lines',
                                line=dict(width=1),
                                marker=dict(color='#F1F316'),
                                yaxis='y2', name='kijun_sen'))

        fig['data'].append(dict(x=df.Date, y=df['senkou_span_a'], type='scatter', mode='lines',
                                line=dict(width=1),
                                marker=dict(color='#228B22'),
                                yaxis='y2', name='senkou_span_a'))

        fig['data'].append(dict(x=df.Date, y=df['senkou_span_b'], type='scatter', mode='lines',
                                line=dict(width=1), fill='tonexty',
                                marker=dict(color='#FF3342'),
                                yaxis='y2', name='senkou_span_b'))

        fig['data'].append(dict(x=df.Date, y=df['chikou_span'], type='scatter', mode='lines',
                                line=dict(width=1),
                                marker=dict(color='#D105F5'),
                                yaxis='y2', name='chikou_span'))
        have_resistance = False
        have_support = False
        for pivot in reversed(pivots):
            pivot_idx, pivot_price, pivot_type = pivot
            if pivot_type == "resistance":
                have_resistance = True
            if pivot_type == "support":
                have_support = True

            y_sample = [pivot_price if pivot_idx < index else None for index in range(len(df.index))]
            x_sample = [date if pivot_idx < index else None for index, date in enumerate(df['Date'])]

            fig['data'].append(dict(x=x_sample, y=y_sample, type='scatter', mode='lines',
                                    line=dict(width=3),
                                    marker=dict(color='#eb346e' if pivot_type == "resistance" else "#52eb34"),
                                    yaxis='y2', name='pivots'))
            if have_support and have_resistance:
                break

        # Set colour list for candlesticks
        colors = []

        for i in range(len(df.close)):
            if i != 0:
                if df.close[i] > df.close[i - 1]:
                    colors.append(INCREASING_COLOR)
                else:
                    colors.append(DECREASING_COLOR)
            else:
                colors.append(DECREASING_COLOR)

        # iplot(fig)
        figg = go.Figure(fig)
        figg.write_image(f"images/{timeframe}.png")

    def ichimoku_cloud(self, timeframe, symbol, save_frame, order_label):
        rates_frame = mt5.copy_rates_from_pos(symbol, timeframe, 0, 150)
        # create DataFrame out of the obtained data
        df = pd.DataFrame(rates_frame, columns=['time', 'open', 'high', 'low', 'close'])
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df['Date'] = df.Date.dt.strftime('%Y.%m.%d %H:%M:%S')
        df = df.drop(['time'], axis=1)
        df = df.reindex(columns=['Date', 'open', 'high', 'low', 'close'])

        # convert to float to avoid sai so.
        df.open = df.open.astype(float)
        df.high = df.high.astype(float)
        df.low = df.low.astype(float)
        df.close = df.close.astype(float)
        real_close = df.close.iat[-1]
        real_high = df.high.iat[-1]
        real_low = df.low.iat[-1]
        #     df = heikin_ashi(df)
        # CCI
        df['ATR'] = ATR(df.high, df.low, df.close)
        #     df['MACD'], df['SIGNAL'], df['HIST'] = MACD(df.close)
        exp1 = df.close.ewm(span=12, adjust=False).mean()
        exp2 = df.close.ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['HIST'] = df['MACD'] - df['SIGNAL']
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

        df['Engulfing'] = CDLENGULFING(df.open, df.high, df.low, df.close)
        conditions = [
            (df['tenkan_sen'] > df['senkou_span_a']) & (df['tenkan_sen'] > df['senkou_span_b']) &
            (df['kijun_sen'] < df['close']) & (df['HIST'] > df['HIST'].shift()),

            (df['tenkan_sen'] < df['senkou_span_a']) & (df['tenkan_sen'] < df['senkou_span_b']) &
            (df['kijun_sen'] > df['close']) & (df['HIST'] < df['HIST'].shift()),
        ]
        values = ['Buy', 'Sell']
        df['Buy_Signal'] = np.select(conditions, values)

        conditions = [
            (df['kijun_sen'] > df['close']),
            (df['kijun_sen'] < df['close'])
        ]
        values = ['Close_Buy', 'Close_Sell']
        df['Close_Signal'] = np.select(conditions, values)

        mapping_timeframe = {
            mt5.TIMEFRAME_H1: "H1",
            mt5.TIMEFRAME_M15: "M15",
            mt5.TIMEFRAME_M5: "M5",
            mt5.TIMEFRAME_M30: "M30",
            mt5.TIMEFRAME_D1: "D1",
            mt5.TIMEFRAME_H4: "H4",
            mt5.TIMEFRAME_H12: "H12",
        }
        map_timeframe = mapping_timeframe.get(timeframe)

        pivots = self.get_pivot(df)
        resistance, support = None, None
        if len(pivots) >= 2:
            pivot_1 = pivots[-1][1]
            pivot_2 = pivots[-2][1]
            if pivot_1 > pivot_2:
                resistance, support = pivot_1, pivot_2
            else:
                resistance, support = pivot_2, pivot_1

        # if df['Engulfing'].iat[-1] != 0:
        #    print(str(df.Date.iat[-1]), timeframe, df['Engulfing'].iat[-1])
        # return str(df.Date.iat[-1]), df.Trend.iat[-1], df.close.iat[-1], df['kijun_sen'].iat[-1]
        # print(f"{timeframe} {df['Date'].iat[-1]} {symbol} tenkan_sen: {df['tenkan_sen'].iat[-1]}  kijun_sen: {df['kijun_sen'].iat[-1]} senkou_span_a: {df['senkou_span_a'].iat[-1]} senkou_span_b: {df['senkou_span_b'].iat[-1]} HIST: {df['HIST'].iat[-1]}")
        # if save_frame:
        #     date_save = str(df.Date.iat[-1]).replace(":", "")
        #     plot_chart(df, pivots, f"{date_save} - {map_timeframe} - {order_label}")

        #     input_df = df['ATR'].dropna().values.reshape(-1, 1)
        #     scaler = MinMaxScaler()
        #     scaler.fit(input_df)
        #     input_df = scaler.transform(input_df).flatten()
        input_df = df['ATR'].dropna().values.reshape(-1, 1)
        hbos = HBOS(alpha=0.1, contamination=0.1, n_bins=20, tol=0.5)
        hbos.fit(input_df)
        predict = hbos.predict(input_df)
        return str(df.Date.iat[-1]), df.Buy_Signal.iat[-1], df.Close_Signal.iat[-1], real_close, real_high, real_low, \
               df.kijun_sen.iat[-1], df['ATR'].iat[-1], df.ATR[-30:].values.tolist(), resistance, support, input_df[-1]

    @staticmethod
    def is_far_from_level(value, levels, df):
        # to make sure the new level area does not exist already
        ave = np.mean(df['high'] - df['low'])
        return np.sum([abs(value - level) < ave for _, level, _ in levels]) == 0

    def get_atr(self, timeframe, symbol):
        rates_frame = mt5.copy_rates_from_pos(symbol, timeframe, 0, 150)
        df = pd.DataFrame(rates_frame, columns=['time', 'open', 'high', 'low', 'close'])
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df['Date'] = df.Date.dt.strftime('%Y.%m.%d %H:%M:%S')
        df = df.drop(['time'], axis=1)
        df = df.reindex(columns=['Date', 'open', 'high', 'low', 'close'])
        return stream.ATR(df.high, df.low, df.close)

    def get_pivot(self, df):
        pivots = []
        max_list = []
        min_list = []
        for i in range(5, len(df) - 5):
            # taking a window of 9 candles
            high_range = df['high'][i - 5:i + 4]
            current_max = high_range.max()
            # if we find a new maximum value, empty the max_list
            if current_max not in max_list:
                max_list = []
            max_list.append(current_max)
            # if the maximum value remains the same after shifting 5 times
            if len(max_list) == 5 and self.is_far_from_level(current_max, pivots, df):
                pivots.append((high_range.idxmax(), current_max, "resistance"))

            low_range = df['low'][i - 5:i + 5]
            current_min = low_range.min()
            if current_min not in min_list:
                min_list = []
            min_list.append(current_min)
            if len(min_list) == 5 and self.is_far_from_level(current_min, pivots, df):
                pivots.append((low_range.idxmin(), current_min, "support"))
        return pivots
