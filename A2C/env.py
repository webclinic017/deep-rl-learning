import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import preprocessing
import talib
import pytz
import MetaTrader5 as mt5
from datetime import datetime, timedelta
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, plot, iplot
mt5.initialize()


class TradingEnv:
    def __init__(self, consecutive_frames=14):
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.consecutive_frames = consecutive_frames
        self.order_p = None
        self.order_size = None
        self.order_index = None
        self.profit = 0
        self.max_profit = 0
        self.max_loss = 0
        self.num_loss = 0
        self.num_profit = 0
        self.stop_loss = None
        self.all_profit = []
        self.lock_back_frame = consecutive_frames
        self.max_diff = 0
        self.agent_diff = 0
        self.min_diff = 0
        self.order_time = None
        self.accept_next = ""
        self.all_max_diff = []
        self.all_min_diff = []
        self.training_step = 100
        timezone = pytz.timezone("Etc/UTC")
        self.end_time = datetime(2022, 3, 15, 5, 0, 0, tzinfo=timezone)
        self.current_time = self.end_time - timedelta(days=60)
        self.symbol = 'XAUUSD'
        self.timeframe_1 = mt5.TIMEFRAME_H4
        self.timeframe_2 = mt5.TIMEFRAME_D1
        self.current_state = []

    @staticmethod
    def calculate_slope(signal):
        x = np.array([x for x in range(len(signal.index))])
        scaler = MinMaxScaler(feature_range=(0, 100))
        y = scaler.fit_transform(signal.values.reshape(-1, 1))
        slope, intercept, r_value, p_value, std_err = linregress(x, y.flatten())
        return slope

    def get_frame(self, utc_to, timeframe, symbol):
        rates_frame = mt5.copy_rates_from(symbol, timeframe, utc_to + timedelta(hours=2), 100)
        rates_frame = rates_frame.copy()
        # create DataFrame out of the obtained data

        if timeframe == mt5.TIMEFRAME_W1:
            num_frame = utc_to.weekday() + 1
            rates_frame_fixed = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_D1, utc_to + timedelta(hours=2), num_frame)
            df_fixed = pd.DataFrame(rates_frame_fixed, columns=['time', 'open', 'high', 'low', 'close'])
            df_fixed['Date'] = pd.to_datetime(df_fixed['time'], unit='s')
            df_fixed['Date'] = df_fixed.Date.dt.strftime('%Y.%m.%d %H:%M:%S')
            df_fixed = df_fixed.drop(['time'], axis=1)
            df_fixed = df_fixed.reindex(columns=['Date', 'open', 'high', 'low', 'close'])

            new_open, new_high, new_low, new_close = df_fixed.open.iat[0], np.amax(df_fixed.high), np.amin(
                df_fixed.low), df_fixed.close.iat[-1]
            current_date = pd.to_datetime(rates_frame[-1][0], unit='s').strftime('%Y.%m.%d %H:%M:%S')

            rates_frame[-1][0] = rates_frame_fixed[-1][0]
            rates_frame[-1][1] = new_open
            rates_frame[-1][2] = new_high
            rates_frame[-1][3] = new_low
            rates_frame[-1][4] = new_close

        if timeframe == mt5.TIMEFRAME_D1:
            num_frame = utc_to.hour % 24 + 1
            rates_frame_fixed = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, utc_to + timedelta(hours=2), num_frame)
            df_fixed = pd.DataFrame(rates_frame_fixed, columns=['time', 'open', 'high', 'low', 'close'])
            df_fixed['Date'] = pd.to_datetime(df_fixed['time'], unit='s')
            df_fixed['Date'] = df_fixed.Date.dt.strftime('%Y.%m.%d %H:%M:%S')
            df_fixed = df_fixed.drop(['time'], axis=1)
            df_fixed = df_fixed.reindex(columns=['Date', 'open', 'high', 'low', 'close'])

            new_open, new_high, new_low, new_close = df_fixed.open.iat[0], np.amax(df_fixed.high), np.amin(
                df_fixed.low), df_fixed.close.iat[-1]
            current_date = pd.to_datetime(rates_frame[-1][0], unit='s').strftime('%Y.%m.%d %H:%M:%S')

            rates_frame[-1][0] = rates_frame_fixed[-1][0]
            rates_frame[-1][1] = new_open
            rates_frame[-1][2] = new_high
            rates_frame[-1][3] = new_low
            rates_frame[-1][4] = new_close

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

        df['ATR'] = talib.ATR(df.high, df.low, df.close)
        df['RSI'] = talib.RSI(df.close)
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
        # calculate slope
        tenkan_array = df.tenkan_sen[-5:]
        close_array = df.close[-5:]
        kinjun_array = df.kijun_sen[-5:]
        macd_array = df.MACD[-5:]
        signal_array = df.SIGNAL[-5:]
        hist_array = df.HIST[-5:]

        tenkan_slope = self.calculate_slope(tenkan_array)
        close_slope = self.calculate_slope(close_array)
        kinjun_slope = self.calculate_slope(kinjun_array)
        macd_slope = self.calculate_slope(macd_array)
        signal_slope = self.calculate_slope(signal_array)
        hist_slope = self.calculate_slope(hist_array)

        input_df_1 = df['tenkan_sen'][-14:]
        input_df_2 = df['kijun_sen'][-14:]
        input_df_3 = df['senkou_span_a'][-14:]
        input_df_4 = df['senkou_span_b'][-14:]
        input_df_5 = df['HIST'][-14:]
        input_df_6 = df['MACD'][-14:]
        input_df_7 = df['SIGNAL'][-14:]
        input_df_8 = df['close'][-14:]
        input_df_9 = df['RSI'][-14:]
        input_df_10 = df['ATR'][-14:]

        normal_array = [self.calculate_slope(x) for x in
                        [input_df_1, input_df_2, input_df_3, input_df_4, input_df_5, input_df_6, input_df_7, input_df_8,
                         input_df_9, input_df_10]]

        slope = 0
        if tenkan_slope > 0 and close_slope > 0 and kinjun_slope > 0 and macd_slope > 0 and signal_slope > 0 and hist_slope > 0:
            slope = 1
        elif tenkan_slope < 0 and close_slope < 0 and kinjun_slope < 0 and macd_slope < 0 and signal_slope < 0 and hist_slope < 0:
            slope = -1

        conditions = [
            (df['tenkan_sen'] > df['senkou_span_a']) & (df['tenkan_sen'] > df['senkou_span_b']) &
            (df.close > df.kijun_sen) & (df.close > df.tenkan_sen) & (df.tenkan_sen > df.kijun_sen) & (
                        df.HIST > df.HIST.shift()),

            (df['tenkan_sen'] < df['senkou_span_a']) & (df['tenkan_sen'] < df['senkou_span_b']) &
            (df.close < df.kijun_sen) & (df.close < df.tenkan_sen) & (df.tenkan_sen < df.kijun_sen) & (
                        df.HIST < df.HIST.shift())
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

        return str(df.Date.iat[-1]), df.Buy_Signal.iat[-1], df.Close_Signal.iat[-1], real_close, real_high, real_low, \
               df.kijun_sen.iat[-1], df['ATR'].iat[-1], normal_array, resistance, support, slope

    @staticmethod
    def is_support(df, i):
        cond1 = df['low'][i] < df['low'][i - 1]
        cond2 = df['low'][i] < df['low'][i + 1]
        cond3 = df['low'][i + 1] < df['low'][i + 2]
        cond4 = df['low'][i - 1] < df['low'][i - 2]
        return (cond1 and cond2 and cond3 and cond4)
        # determine bearish fractal

    @staticmethod
    def is_resistance(df, i):
        cond1 = df['high'][i] > df['high'][i - 1]
        cond2 = df['high'][i] > df['high'][i + 1]
        cond3 = df['high'][i + 1] > df['high'][i + 2]
        cond4 = df['high'][i - 1] > df['high'][i - 2]
        return (cond1 and cond2 and cond3 and cond4)

    @staticmethod
    # to make sure the new level area does not exist already
    def is_far_from_level(value, levels, df):
        ave = np.mean(df['high'] - df['low'])
        return np.sum([abs(value - level) < ave for _, level, _ in levels]) == 0

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

    @staticmethod
    def get_trend(trend1, trend2, close_signal):
        current_trend = "0"
        if trend1 == trend2 == "Sell":
            current_trend = "Sell"
        elif trend1 == trend2 == "Buy":
            current_trend = "Buy"
        elif trend1 != "Sell" and trend2 != "Sell" and close_signal == 'Close_Sell':
            current_trend = "Close_Sell"
        elif trend1 != "Buy" and trend2 != "Buy" and close_signal == 'Close_Buy':
            current_trend = "Close_Buy"
        return current_trend

    def step(self, action):
        r = -1
        # actions
        # 0 not buy, 1 buy
        df1date, h1_trend, close_signal_1, current_price_1, high_price_1, low_price_1, kijun_sen_1, atr_1, dftrain_1, resistance_1, support_1, signal_slope_1 = self.get_frame(utc_to=self.current_time, timeframe=self.timeframe_1, symbol=self.symbol)
        df2date, h2_trend, close_signal_2, current_price_2, high_price_2, low_price_2, kijun_sen_2, atr_2, dftrain_2, resistance_2, support_2, signal_slope_2 = self.get_frame(utc_to=self.current_time, timeframe=self.timeframe_2, symbol=self.symbol)
        current_price = current_price_1
        current_trend = "0"
        if h1_trend == h2_trend == "Sell":
            current_trend = "Sell"
        elif h1_trend == h2_trend == "Buy":
            current_trend = "Buy"
        elif h1_trend != "Sell" and h2_trend != "Sell" and close_signal_2 == "Close_Sell":
            current_trend = "Close_Sell"
        elif h1_trend != "Buy" and h2_trend != "Buy" and close_signal_2 == "Close_Buy":
            current_trend = "Close_Buy"

        if self.order_p and self.order_size == "Buy" and action == 1:
            diff = current_price - self.order_p
            if self.stop_loss and self.stop_loss > low_price_1:
                diff = self.stop_loss - self.order_p

            print(
                f"{df1date} close \x1b[48;5;4m{self.order_size}\x1b[0m order at price {current_price}, max diff {round(self.max_diff, 5)} diff {round(diff, 5)}")
            self.profit += diff
            # profit -= profit*0.05
            self.all_profit.append(self.profit)
            if diff < 0:
                self.all_min_diff.append(diff)
                self.max_loss += diff
                self.num_loss += 1
                label = 0
            elif diff > 0:
                self.all_max_diff.append(self.max_diff)
                self.max_profit += diff
                self.num_profit += 1
                r = 1

            # reset flags
            # reset flags
            self.order_p = None
            self.order_size = None
            self.stop_loss = None
            self.order_index = None
            self.max_diff = 0
            self.agent_diff = 0
            self.order_time = None
            self.current_state = [0]*15

        elif self.order_p is None and current_trend == "Buy" and resistance_1 and current_price > resistance_1 and signal_slope_1 == 1:
            self.order_p = current_price
            self.order_size = current_trend
            self.stop_loss = None
            self.accept_next = ""
            self.max_diff = 0
            print(f"{df1date} new \x1b[48;5;2m{self.order_size}\x1b[0m order at price {current_price}")

        diff = 0
        if self.order_size == "Buy":
            diff = current_price - self.order_p
            if diff > self.max_diff:
                self.max_diff = diff
            if self.stop_loss is None and diff > atr_1:
                self.stop_loss = self.order_p
            if self.stop_loss and self.stop_loss < kijun_sen_1 < current_price:
                self.stop_loss = kijun_sen_1

        self.current_state.append(diff)

        # get new state
        # new_state, r, done, info
        info = {
            "all_profit": self.all_profit,
            "max_loss": self.max_loss,
            "num_loss": self.num_loss,
            "max_profit": self.max_profit,
            "num_profit": self.num_profit,
            "profit": self.profit
        }

        state = np.array(self.current_state[-15:])
        # state = MinMaxScaler().fit_transform(state.reshape(1, -1))[0]
        self.current_time += timedelta(hours=4)
        done = True if self.current_time >= self.end_time else False
        return state, r, done, info

    def reset(self):
        self.current_time = self.end_time - timedelta(days=30)
        self.order_p = None
        self.order_size = None
        self.order_index = None
        self.profit = 0
        self.max_profit = 0
        self.max_loss = 0
        self.num_loss = 0
        self.num_profit = 0
        self.stop_loss = None
        self.all_profit = []
        self.max_diff = 0
        self.min_diff = 0
        self.agent_diff = 0
        self.accept_next = ""
        self.all_max_diff = []
        self.all_min_diff = []
        self.current_state = [0]*15

        return np.array(self.current_state)

    @staticmethod
    def get_state_size():
        return 15

    @staticmethod
    def get_action_space():
        return 2

    def get_valid_actions(self):
        if self.order_p:
            if self.order_size == "Sell":
                return [0, 1]
            if self.order_size == "Buy":
                return [0, 2]
        return [0]

    @staticmethod
    def plot_chart(df, pivots, timeframe, order_date):
        # Set colours for up and down candles
        INCREASING_COLOR = '#26a69a'
        DECREASING_COLOR = '#ef5350'

        # create list to hold dictionary with data for our first series to plot
        # (which is the candlestick element itself)
        data = [
            dict(
                type='candlestick',
                open=df.open,
                high=df.high,
                low=df.low,
                close=df.close,
                x=df.Date,
                yaxis='y3',
                name='candle stick chart',
                increasing=dict(line=dict(color=INCREASING_COLOR)),
                decreasing=dict(line=dict(color=DECREASING_COLOR)),
            ), dict(
                type='scatter',
                name="SIGNAL",
                mode="lines",
                x=df.Date,
                y=df.SIGNAL,
                yaxis='y2'
            ), dict(
                type='scatter',
                name="MACD",
                x=df.Date,
                y=df.MACD,
                yaxis='y2'
            ), dict(
                type='bar',
                name="MACD Histogram",
                x=df.Date,
                y=df.HIST,
                yaxis='y2'
            ), dict(
                type='scatter',
                name="MACD Histogram Norm",
                x=df.Date,
                y=df.HIST_NORM,
                yaxis='y2'
            ), dict(
                type='scatter',
                mode="lines",
                name="RSI",
                x=df.Date,
                y=df.RSI
            )
        ]

        # Create empty dictionary for later use to hold settings and layout options
        layout = {
            "yaxis": {"domain": [0, 0.33]},
            "yaxis2": {"domain": [0.33, 0.66]},
            "yaxis3": {"domain": [0.66, 1]},
            "xaxis": {"rangeslider": {"visible": False}}
        }

        # create our main chart "Figure" object which consists of data to plot and layout settings
        fig = dict(data=data, layout=layout)

        # Assign various seeting and choices - background colour, range selector etc
        fig['layout']['plot_bgcolor'] = 'rgb(250, 250, 250)'
        # fig['layout']['xaxis'] = dict( rangeselector = dict( visible = False ) )
        # fig['layout']['yaxis'] = dict( domain = [0, 0.2], showticklabels = False )
        # fig['layout']['yaxis2'] = dict( domain = [0.2, 0.8] )
        # fig['layout']['legend'] = dict( orientation = 'h', y=0.9, x=0.3, yanchor='bottom' )
        fig['layout']['margin'] = dict(t=40, b=40, r=40, l=40)
        fig['layout']['width'] = 1920
        fig['layout']['height'] = 1080

        # Populate the "rangeselector" object with necessary settings
        rangeselector = dict(
            visible=False,
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

        # fig['layout']['xaxis']['rangeselector'] = rangeselector

        # Append the Ichimoku elements to the plot
        fig['data'].append(dict(x=df.Date, y=df['tenkan_sen'], type='scatter', mode='lines',
                                line=dict(width=1),
                                marker=dict(color='#33BDFF'),
                                yaxis='y3', name='tenkan_sen'))

        fig['data'].append(dict(x=df.Date, y=df['kijun_sen'], type='scatter', mode='lines',
                                line=dict(width=1),
                                marker=dict(color='#F1F316'),
                                yaxis='y3', name='kijun_sen'))

        fig['data'].append(dict(x=df.Date, y=df['senkou_span_a'], type='scatter', mode='lines',
                                line=dict(width=1),
                                marker=dict(color='#228B22'),
                                yaxis='y3', name='senkou_span_a'))

        fig['data'].append(dict(x=df.Date, y=df['senkou_span_b'], type='scatter', mode='lines',
                                line=dict(width=1), fill='tonexty',
                                marker=dict(color='#FF3342'),
                                yaxis='y3', name='senkou_span_b'))

        fig['data'].append(dict(x=df.Date, y=df['chikou_span'], type='scatter', mode='lines',
                                line=dict(width=1),
                                marker=dict(color='#D105F5'),
                                yaxis='y3', name='chikou_span'))
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
                                    yaxis='y3', name='pivots'))
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
        for index, date in enumerate(df['Date']):
            if date == order_date:
                # print(order_date, df.close.iat[index])
                fig['data'].append(
                    dict(x=[order_date], y=[df.close.iat[index]], yaxis='y3', marker=dict(color='#3256a8')))
                break

        # plot(fig, filename='simple-line.html')


if __name__ == '__main__':
    env = TradingEnv()
