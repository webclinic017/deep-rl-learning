import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import preprocessing
import talib
import pytz
import MetaTrader5 as mt5
from datetime import datetime, timedelta

mt5.initialize()


class TradingEnv:
    def __init__(self, consecutive_frames=14):
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.consecutive_frames = consecutive_frames
        # self.df, self.close_prices = self.load_data()
        # self.X_train = MinMaxScaler().fit_transform(np.expand_dims(self.df.HIST, axis=1)).flatten().tolist()
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
        self.end_time = datetime(2021, 12, 14, 0, 59, 55, tzinfo=timezone)
        self.current_time = self.end_time - timedelta(days=100)
        self.symbol = 'XAUUSD'

    def get_frame(self, time_from, time_to, timeframe, symbol):
        rates_frame = mt5.copy_rates_range(symbol, timeframe, time_from, time_to)
        df = pd.DataFrame(rates_frame, columns=['time', 'open', 'high', 'low', 'close'])
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df = df.drop(['time'], axis=1)
        df = df.reindex(columns=['Date', 'open', 'high', 'low', 'close'])

        # convert to float to avoid sai so.
        df.open = df.open.astype(float)
        df.high = df.high.astype(float)
        df.low = df.low.astype(float)
        df.close = df.close.astype(float)

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
        conditions = [
            (df['tenkan_sen'] > df['kijun_sen']) & (df['close'] > df['kijun_sen']) & (df['close'] > df['senkou_span_a']) & (df['close'] > df['senkou_span_b']),
            (df['tenkan_sen'] < df['kijun_sen']) & (df['close'] < df['kijun_sen']) & (df['close'] < df['senkou_span_a']) & (df['close'] < df['senkou_span_b'])
        ]
        values = ['Buy', 'Sell']
        df['Trend'] = np.select(conditions, values)
        return str(df.Date.iat[-1]), df.Trend.iat[-1], df.close.iat[-1], df.HIST.iloc[-28:-1].values.flatten().tolist()

    @staticmethod
    def get_trend(trend1, trend2):
        current_trend = '0'
        if trend1 == trend2 == "Sell":
            current_trend = "Sell"
        if trend1 == trend2 == "Buy":
            current_trend = "Buy"
        return current_trend

    def step(self, action):
        h1date, h1trend, h1price, h1train = self.get_frame(time_from=self.current_time-timedelta(days=100),
                                                           time_to=self.current_time,
                                                           timeframe=mt5.TIMEFRAME_H4, symbol=self.symbol)
        h4date, h4trend, h4price, h4train = self.get_frame(time_from=self.current_time - timedelta(days=150),
                                                           time_to=self.current_time,
                                                           timeframe=mt5.TIMEFRAME_D1, symbol=self.symbol)

        current_price = h1price
        current_trend = self.get_trend(h1trend, h4trend)
        valid_actions = self.get_valid_actions()
        r = 0 if action in valid_actions else -1
        diff = 0
        if self.order_size == "Sell":
            diff = self.order_p - current_price
        if self.order_size == "Buy":
            diff = current_price - self.order_p

        # actions
        # 0 hold, 1 close buy, 2 close sell
        # current_trend == "Neutral" or current_trend == 'Buy'
        # or h1trend == 'Buy' or h4trend == 'Buy'
        # or h1trend == 'Close_Sell' or h4trend == 'Close_Sell'
        if self.order_p and self.order_size == "Sell" and action == 1:
            time_tmp = self.order_time
            max_diff = 0
            normal_diff = 0
            while True:
                time_tmp += timedelta(hours=4)
                _, h1trend_tmp, h1price_tmp, _ = self.get_frame(time_from=time_tmp - timedelta(days=100),
                                                                time_to=time_tmp,
                                                                timeframe=mt5.TIMEFRAME_H4, symbol=self.symbol)
                _, h4trend_tmp, _, _ = self.get_frame(time_from=time_tmp - timedelta(days=150),
                                                      time_to=time_tmp,
                                                      timeframe=mt5.TIMEFRAME_D1, symbol=self.symbol)
                current_trend_tmp = self.get_trend(h1trend_tmp, h4trend_tmp)

                # calculate diff
                diff_tmp = self.order_p - h1price_tmp
                if diff_tmp > max_diff:
                    max_diff = diff_tmp

                if (current_trend_tmp == "Neutral" or current_trend_tmp == 'Buy' or
                    h1trend_tmp == 'Buy' or h4trend_tmp == 'Buy' or
                        h1trend_tmp == 'Close_Sell' or h4trend_tmp == 'Close_Sell' or time_tmp.timestamp() <= self.end_time.timestamp()):
                    # order stop here. calculate max min and reward
                    normal_diff = self.order_p - h1price_tmp
                    break

            diff = self.order_p - current_price
            self.profit += diff
            self.all_profit.append(self.profit)
            if diff < 0:
                self.all_min_diff.append(self.max_diff)
                self.max_loss += diff
                self.num_loss += 1
            else:
                self.all_max_diff.append(self.max_diff)
                self.max_profit += diff
                self.num_profit += 1
            r = 100 if normal_diff < diff < max_diff else 0

            # reset flags
            self.order_p = None
            self.order_size = None
            self.stop_loss = None
            self.order_index = None
            self.max_diff = 0
            self.agent_diff = 0
            self.order_time = None

        elif self.order_p and self.order_size == "Buy" and action == 2:
            time_tmp = self.order_time
            max_diff = 0
            normal_diff = 0
            while True:
                time_tmp += timedelta(hours=4)
                _, h1trend_tmp, h1price_tmp, _ = self.get_frame(time_from=time_tmp - timedelta(days=100),
                                                                time_to=time_tmp,
                                                                timeframe=mt5.TIMEFRAME_H4, symbol=self.symbol)
                _, h4trend_tmp, _, _ = self.get_frame(time_from=time_tmp - timedelta(days=150),
                                                      time_to=time_tmp,
                                                      timeframe=mt5.TIMEFRAME_D1, symbol=self.symbol)
                current_trend_tmp = self.get_trend(h1trend_tmp, h4trend_tmp)

                # calculate diff
                diff_tmp = h1price_tmp - self.order_p
                if diff_tmp > max_diff:
                    max_diff = diff_tmp

                if (current_trend_tmp == "Neutral" or current_trend_tmp == 'Sell' or
                    h1trend_tmp == 'Sell' or h4trend_tmp == 'Sell' or
                        h1trend_tmp == 'Close_Buy' or h4trend_tmp == 'Close_Buy' or time_tmp.timestamp() <= self.end_time.timestamp()):
                    # order stop here. calculate max min and reward
                    normal_diff = h1price_tmp - self.order_p
                    break

            diff = current_price - self.order_p
            self.profit += diff
            self.all_profit.append(self.profit)
            if diff < 0:
                self.all_min_diff.append(self.max_diff)
                self.max_loss += diff
                self.num_loss += 1
            else:
                self.all_max_diff.append(self.max_diff)
                self.max_profit += diff
                self.num_profit += 1
            r = 100 if normal_diff < diff < max_diff else 0

            # accept_next = "Sell"
            # reset flags
            self.order_p = None
            self.order_size = None
            self.stop_loss = None
            self.order_index = None
            self.max_diff = 0
            self.agent_diff = 0
            self.order_time = None

        if self.order_p is None and current_trend == "Buy":
            # print("Buy order")
            self.order_time = self.current_time
            self.order_p = current_price
            self.order_size = current_trend
        elif self.order_p is None and current_trend == "Sell":
            # print("Sell order")
            self.order_time = self.current_time
            self.order_p = current_price
            self.order_size = current_trend

        # if self.order_p:
        #     if self.order_size == "Sell":
        #
        #     if self.order_size == "Buy":
        #         diff = current_price - self.order_p
        #         if diff > self.max_diff:
        #             self.max_diff = diff

        # new_state, r, done, info
        new_state = []
        new_state.extend(h1train)
        new_state.extend(h4train)
        new_state.extend([self.max_diff/100, diff/100])
        new_state.append(1 if self.order_p else 0)
        new_state = np.array(new_state)
        done = True if self.current_time.timestamp() >= self.end_time.timestamp() else False
        info = {
            "all_profit": self.all_profit,
            "max_loss": self.max_loss,
            "num_loss": self.num_loss,
            "max_profit": self.max_profit,
            "num_profit": self.num_profit,
            "profit": self.profit
        }

        self.current_time += timedelta(hours=4)
        return new_state, r, done, info

    def reset(self):
        self.current_time = self.end_time - timedelta(days=100)
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

        h1date, h1trend, h1price, h1train = self.get_frame(time_from=self.current_time-timedelta(days=50),
                                                           time_to=self.current_time,
                                                           timeframe=mt5.TIMEFRAME_H4, symbol=self.symbol)
        h4date, h4trend, h4price, h4train = self.get_frame(time_from=self.current_time - timedelta(days=100),
                                                           time_to=self.current_time,
                                                           timeframe=mt5.TIMEFRAME_D1, symbol=self.symbol)

        # new_state, r, done, info
        new_state = []
        new_state.extend(h1train)
        new_state.extend(h4train)
        new_state.extend([0, 0])
        new_state.append(0)
        new_state = np.array(new_state)
        self.training_step += 1
        return new_state

    @staticmethod
    def get_state_size():
        return 57

    @staticmethod
    def get_action_space():
        return 3

    def get_valid_actions(self):
        if self.order_p:
            if self.order_size == "Sell":
                return [0, 1]
            if self.order_size == "Buy":
                return [0, 2]
        return [0]


if __name__ == '__main__':
    env = TradingEnv()
