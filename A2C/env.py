import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import preprocessing
import talib


class TradingEnv:
    def __init__(self, consecutive_frames=14):
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.consecutive_frames = consecutive_frames
        self.df, self.close_prices = self.load_data()
        self.X_train = MinMaxScaler().fit_transform(np.expand_dims(self.df.HIST, axis=1)).flatten().tolist()
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
        dfindex = self.training_step
        current_trend = self.df.Trend.iat[self.training_step]
        adx = self.df.ADX.iat[self.training_step]
        current_price = self.close_prices.iat[self.training_step]
        sp_112 = self.df.SuperTrend310.iat[self.training_step]
        atr = self.df.ATR.iat[self.training_step]
        r = 0
        # actions
        # 0 hold, 1 close buy, 2 close sell
        if self.order_p and self.order_size == "Sell" and (action == 2):
            diff = self.order_p - current_price
            #         if max_diff > 10 and diff < 0:
            #             diff = 1

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
                r = 0.01 * diff

            # accept_next = "Buy"
            # reset flags
            self.order_p = None
            self.order_size = None
            self.stop_loss = None
            self.order_index = None
            self.max_diff = 0
        if self.order_p is None and current_trend == "Buy":
            # save_result(dfindex, "Buy", adx)
            self.order_p = current_price
            self.order_size = current_trend
            self.stop_loss = current_price - (1 * atr)
            self.accept_next = ""
            self.order_index = dfindex
            # print(f"{dfindex} new \x1b[48;5;2m{self.order_size}\x1b[0m order at price {current_price}")
        if self.order_p and self.order_size == "Buy" and (action == 1):
            diff = current_price - self.order_p
            #         if max_diff > 10 and diff < 0:
            #             diff = 1

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
                r = 0.01 * diff

            # accept_next = "Sell"
            # reset flags
            self.order_p = None
            self.order_size = None
            self.stop_loss = None
            self.order_index = None
            self.max_diff = 0
        if self.order_p is None and current_trend == "Sell":
            # save_result(dfindex, "Sell", adx)
            self.order_p = current_price
            self.order_size = current_trend
            self.stop_loss = current_price + (1 * atr)
            self.accept_next = ""
            self.order_index = dfindex

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
        new_state = self.X_train[self.training_step-self.lock_back_frame:self.training_step]
        new_state.append(self.profit/100)
        new_state.append(1 if self.order_p else 0)
        new_state = np.array(new_state)
        done = True if self.training_step >= self.df.shape[0] - 1 else False
        if done:
            r += 0.01 * self.max_loss
        info = {
            "all_profit": self.all_profit,
            "max_loss": self.max_loss,
            "num_loss": self.num_loss,
            "max_profit": self.max_profit,
            "num_profit": self.num_profit,
            "profit": self.profit
        }
        if not done:
            self.training_step += 1
        else:
            self.reset()

        return new_state, r, done, info

    def reset(self):
        self.training_step = 100
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

        new_state = self.X_train[self.training_step-self.lock_back_frame:self.training_step]
        new_state.append(self.profit/100)
        new_state.append(1 if self.order_p else 0)
        new_state = np.array(new_state)
        self.training_step += 1
        return new_state

    def get_state_size(self):
        return self.df.X_train.values.shape[1]

    @staticmethod
    def get_action_space():
        return 3

    def get_valid_actions(self):
        if self.order_p:
            if self.order_size == "Sell":
                return [0, 2]
            if self.order_size == "Buy":
                return [0, 1]
        return [0]


if __name__ == '__main__':
    env = TradingEnv()
