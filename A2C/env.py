import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import preprocessing
import talib
import pytz
import MetaTrader5 as mt5
from datetime import datetime
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
        self.min_diff = 0
        self.accept_next = ""
        self.all_max_diff = []
        self.all_min_diff = []
        self.training_step = 100
        self.end_time = 1637722548
        self.current_time = self.end_time - 86400*120
        self.symbol = 'XAUUSD'

    def get_frame(self, time_from, time_to, timeframe, symbol):
        timezone = pytz.timezone("Etc/UTC")
        utc_from = datetime.fromtimestamp(time_from)
        utc_to = datetime.fromtimestamp(time_to)
        rates_frame = mt5.copy_rates_range(symbol, timeframe, utc_from, utc_to)
        # create DataFrame out of the obtained data
        df = pd.DataFrame(rates_frame, columns=['time', 'open', 'high', 'low', 'close'])
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        df = df.drop(['time'], axis=1)
        df = df.reindex(columns=['Date', 'open', 'high', 'low', 'close'])
        df['EMA_50'] = df.close.ewm(span=50, adjust=False).mean()
        atr_multiplier = 3.0
        atr_period = 10
        supertrend = self.super_trend(df, atr_period, atr_multiplier)
        df = df.join(supertrend)
        df['MACD'], df['SIGNAL'], df['HIST'] = talib.MACD(df.close, fastperiod=12, slowperiod=26, signalperiod=9)
        lookback = 14
        conditions = [
            (df['Supertrend10'] == True) & (df['HIST'] > 0),
            (df['Supertrend10'] == False) & (df['HIST'] < 0)
        ]
        values = ['Buy', 'Sell']
        df['Trend'] = np.select(conditions, values)
        return df.close.iat[-1], df.HIST[-30:].values.flatten().tolist(), df.Trend.iat[-1], str(df.Date.iat[-1])

    @staticmethod
    def super_trend(df, atr_period, multiplier):

        high = df['high']
        low = df['low']
        close = df['close']

        # calculate ATR
        price_diffs = [high - low,
                       high - close.shift(),
                       close.shift() - low]
        true_range = pd.concat(price_diffs, axis=1)
        true_range = true_range.abs().max(axis=1)
        # default ATR calculation in supertrend indicator
        atr = true_range.ewm(alpha=1 / atr_period, min_periods=atr_period).mean()
        # df['atr'] = df['tr'].rolling(atr_period).mean()

        # HL2 is simply the average of high and low prices
        hl2 = (high + low) / 2
        # upperband and lowerband calculation
        # notice that final bands are set to be equal to the respective bands
        final_upperband = upperband = hl2 + (multiplier * atr)
        final_lowerband = lowerband = hl2 - (multiplier * atr)

        # initialize Supertrend column to True
        supertrend = [True] * len(df)

        for i in range(1, len(df.index)):
            curr, prev = i, i - 1

            # if current close price crosses above upperband
            if close[curr] > final_upperband[prev]:
                supertrend[curr] = True
            # if current close price crosses below lowerband
            elif close[curr] < final_lowerband[prev]:
                supertrend[curr] = False
            # else, the trend continues
            else:
                supertrend[curr] = supertrend[prev]

                # adjustment to the final bands
                if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                    final_lowerband[curr] = final_lowerband[prev]
                if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                    final_upperband[curr] = final_upperband[prev]

            # to remove bands according to the trend direction
            if supertrend[curr] == True:
                final_upperband[curr] = np.nan
            else:
                final_lowerband[curr] = np.nan

        return pd.DataFrame({
            f'Supertrend{atr_period}': supertrend,
            f'FinalLowerband{atr_period}': final_lowerband,
            f'FinalUpperband{atr_period}': final_upperband
        }, index=df.index)

    def load_data(self):
        df = pd.read_csv(f"E:\\Code\\Thinh\\deep-rl-learning\\data\\XAUUSDH1.csv")
        close_price = df.Close
        df = self.heikin_ashi(df)
        df['EMA_200'] = talib.EMA(df.Close, timeperiod=200)
        df['EMA_50'] = talib.EMA(df.Close, timeperiod=50)
        df['EMA_20'] = talib.EMA(df.Close, timeperiod=20)
        df['ADX'] = talib.ADX(df.High, df.Low, df.Close, timeperiod=14)
        df['RSI'] = talib.RSI(df.Close, timeperiod=14)
        df['MACD'], df['SIGNAL'], df['HIST'] = talib.MACD(df.Close, fastperiod=12, slowperiod=26, signalperiod=9)
        # df = self.ST(df, f=1, n=12)
        df = self.ST(df, f=3, n=10)
        # df = self.ST(df, f=2, n=11)

        lookback = 1
        conditions = [
            (df['SuperTrend310'] > df['Close']) & (df['HIST'] < df['HIST'].shift(lookback)) & (
                    df['ADX'] > df['ADX'].shift(lookback)),
            (df['SuperTrend310'] < df['Close']) & (df['HIST'] > df['HIST'].shift(lookback)) & (
                    df['ADX'] > df['ADX'].shift(lookback))
        ]
        values = ['Sell', 'Buy']
        df['Trend'] = np.select(conditions, values)

        return df, close_price

    # SuperTrend
    @staticmethod
    def ST(df, f, n):  # df is the dataframe, n is the period, f is the factor; f=3, n=7 are commonly used.
        # Calculation of ATR
        col_name = f"SuperTrend{f}{n}"
        df['H-L'] = abs(df['High'] - df['Low'])
        df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
        df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = np.nan
        df.loc[n - 1, 'ATR'] = df['TR'][:n - 1].mean()  # .ix is deprecated from pandas verion- 0.19
        for i in range(n, len(df)):
            df['ATR'][i] = (df['ATR'][i - 1] * (n - 1) + df['TR'][i]) / n

        # Calculation of SuperTrend
        df['Upper Basic'] = (df['High'] + df['Low']) / 2 + (f * df['ATR'])
        df['Lower Basic'] = (df['High'] + df['Low']) / 2 - (f * df['ATR'])
        df['Upper Band'] = df['Upper Basic']
        df['Lower Band'] = df['Lower Basic']
        for i in range(n, len(df)):
            if df['Close'][i - 1] <= df['Upper Band'][i - 1]:
                df['Upper Band'][i] = min(df['Upper Basic'][i], df['Upper Band'][i - 1])
            else:
                df['Upper Band'][i] = df['Upper Basic'][i]
        for i in range(n, len(df)):
            if df['Close'][i - 1] >= df['Lower Band'][i - 1]:
                df['Lower Band'][i] = max(df['Lower Basic'][i], df['Lower Band'][i - 1])
            else:
                df['Lower Band'][i] = df['Lower Basic'][i]
        df[col_name] = np.nan
        for i in df[col_name]:
            if df['Close'][n - 1] <= df['Upper Band'][n - 1]:
                df[col_name][n - 1] = df['Upper Band'][n - 1]
            elif df['Close'][n - 1] > df['Upper Band'][i]:
                df[col_name][n - 1] = df['Lower Band'][n - 1]
        for i in range(n, len(df)):
            if df[col_name][i - 1] == df['Upper Band'][i - 1] and df['Close'][i] <= df['Upper Band'][i]:
                df[col_name][i] = df['Upper Band'][i]
            elif df[col_name][i - 1] == df['Upper Band'][i - 1] and df['Close'][i] >= df['Upper Band'][i]:
                df[col_name][i] = df['Lower Band'][i]
            elif df[col_name][i - 1] == df['Lower Band'][i - 1] and df['Close'][i] >= df['Lower Band'][i]:
                df[col_name][i] = df['Lower Band'][i]
            elif df[col_name][i - 1] == df['Lower Band'][i - 1] and df['Close'][i] <= df['Lower Band'][i]:
                df[col_name][i] = df['Upper Band'][i]
        return df

    @staticmethod
    def heikin_ashi(df):
        heikin_ashi_df = pd.DataFrame(index=df.index.values, columns=['Open', 'High', 'Low', 'Close'])

        heikin_ashi_df['Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

        for i in range(len(df)):
            if i == 0:
                heikin_ashi_df.iat[0, 0] = df['Open'].iloc[0]
            else:
                heikin_ashi_df.iat[i, 0] = (heikin_ashi_df.iat[i - 1, 0] + heikin_ashi_df.iat[i - 1, 3]) / 2

        heikin_ashi_df['High'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(df['High']).max(axis=1)

        heikin_ashi_df['Low'] = heikin_ashi_df.loc[:, ['Open', 'Close']].join(df['Low']).min(axis=1)

        return heikin_ashi_df

    def step(self, action):
        current_price, h1_train, h1_trend, dfdate = self.get_frame(time_from=self.current_time - 86400 * 14,
                                                                   time_to=self.current_time,
                                                                   timeframe=mt5.TIMEFRAME_H1, symbol=self.symbol)
        _, h4_train, h4_trend, _ = self.get_frame(time_from=self.current_time - 86400 * 35, time_to=self.current_time,
                                                  timeframe=mt5.TIMEFRAME_H4, symbol=self.symbol)
        _, d1_train, d1_trend, _ = self.get_frame(time_from=self.current_time - 86400 * 100, time_to=self.current_time,
                                                  timeframe=mt5.TIMEFRAME_D1, symbol=self.symbol)

        current_trend = '0'
        if h1_trend == h4_trend == d1_trend == "Sell":
            current_trend = "Sell"
        if h1_trend == h4_trend == d1_trend == "Buy":
            current_trend = "Buy"

        r = -1
        diff = 0
        if self.order_size == "Sell":
            diff = self.order_p - current_price
        if self.order_size == "Buy":
            diff = current_price - self.order_p

        # actions
        # 0 hold, 1 buy, 2 sell
        if self.order_p and self.order_size == "Sell" and (action == 1):
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
            # r = 0.01 * diff

            # accept_next = "Buy"
            # reset flags
            self.order_p = None
            self.order_size = None
            self.stop_loss = None
            self.order_index = None
            self.max_diff = 0
        if self.order_p and self.order_size == "Buy" and (action == 2):
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
            # r = 0.01 * diff

            # accept_next = "Sell"
            # reset flags
            self.order_p = None
            self.order_size = None
            self.stop_loss = None
            self.order_index = None
            self.max_diff = 0

        if self.order_p is None and current_trend == "Buy":
            self.order_p = current_price
            self.order_size = current_trend
            # print(f"{dfindex} new \x1b[48;5;2m{self.order_size}\x1b[0m order at price {current_price}")
        if self.order_p is None and current_trend == "Sell":
            # save_result(dfindex, "Sell", adx)
            self.order_p = current_price
            self.order_size = current_trend

        if self.order_p:
            if self.order_size == "Sell":
                diff = self.order_p - current_price
                if diff > self.max_diff:
                    self.max_diff = diff
            if self.order_size == "Buy":
                diff = current_price - self.order_p
                if diff > self.max_diff:
                    self.max_diff = diff

        # new_state, r, done, info
        new_state = []
        new_state.extend(h1_train)
        new_state.extend(h4_train)
        new_state.extend(d1_train)
        new_state.extend([self.max_diff/10, diff/10])
        new_state.append(1 if self.order_p else 0)
        new_state = np.array(new_state)
        done = True if self.current_time >= self.end_time - 86400 else False
        if done:
            r = 10 if self.profit > 150 else 0
        info = {
            "all_profit": self.all_profit,
            "max_loss": self.max_loss,
            "num_loss": self.num_loss,
            "max_profit": self.max_profit,
            "num_profit": self.num_profit,
            "profit": self.profit
        }
        if not done:
            self.current_time += 60*60
        else:
            self.reset()

        return new_state, r, done, info

    def reset(self):
        self.current_time = self.end_time - 86400*120
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
        self.accept_next = ""
        self.all_max_diff = []
        self.all_min_diff = []

        current_price, h1_train, h1_trend, dfdate = self.get_frame(time_from=self.current_time - 86400 * 14,
                                                                   time_to=self.current_time,
                                                                   timeframe=mt5.TIMEFRAME_H1, symbol=self.symbol)
        _, h4_train, h4_trend, _ = self.get_frame(time_from=self.current_time - 86400 * 35, time_to=self.current_time,
                                                  timeframe=mt5.TIMEFRAME_H4, symbol=self.symbol)
        _, d1_train, d1_trend, _ = self.get_frame(time_from=self.current_time - 86400 * 100, time_to=self.current_time,
                                                  timeframe=mt5.TIMEFRAME_D1, symbol=self.symbol)

        # new_state, r, done, info
        new_state = []
        new_state.extend(h1_train)
        new_state.extend(h4_train)
        new_state.extend(d1_train)
        new_state.extend([self.max_diff, 0])
        new_state.append(1 if self.order_p else 0)
        new_state = np.array(new_state)
        self.training_step += 1
        return new_state

    @staticmethod
    def get_state_size():
        return 93

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
