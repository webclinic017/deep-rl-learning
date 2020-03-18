import pandas as pd
import numpy as np


class TradingEnv:
    def __init__(self, consecutive_frames=40):
        self.t = 120
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.waiting_time = 0
        self.consecutive_frames = consecutive_frames
        self.train_data, self.normalize_price, self.raw_price, self.moving_avg = self.get_data('../data/5minute.csv')

    def get_data(self, csv_path):
        df = pd.read_csv(csv_path, sep=',')
        normalize_price = self.normalize(df.Close.values)
        raw_price = df.Close.values
        moving_avg = df.Close.rolling(window=14).mean().dropna().values
        return df, normalize_price, raw_price, moving_avg

    def get_train_data(self):
        """
        Return data used for training at time step t, reset t if out of range
        """
        if self.t == len(self.train_data) - 200:
            self.t = 120

        data = self.train_data[self.t-120: self.t]
        return data

    @staticmethod
    def macd(data_frame):
        data = data_frame.copy()
        exp1 = data.Close.ewm(span=12, adjust=False).mean()
        exp2 = data.Close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        return macd, signal, histogram

    @staticmethod
    def stochastics(data_frame, k=14, d=3):
        """
        Fast stochastic calculation
        %K = (Current Close - Lowest Low)/
        (Highest High - Lowest Low) * 100
        %D = 3-day SMA of %K

        Slow stochastic calculation
        %K = %D of fast stochastic
        %D = 3-day SMA of %K

        When %K crosses above %D, buy signal
        When the %K crosses below %D, sell signal
        """
        df = data_frame.copy()

        # Set minimum low and maximum high of the k stoch
        low_min = df.Low.rolling(window=k).min()
        high_max = df.High.rolling(window=k).max()

        # Fast Stochastic
        df['k_fast'] = 100 * (df.Close - low_min) / (high_max - low_min)
        df['d_fast'] = df['k_fast'].rolling(window=d).mean()

        # Slow Stochastic
        df['k_slow'] = df["d_fast"]
        df['d_slow'] = df['k_slow'].rolling(window=d).mean()

        slow_k = df['k_slow'].fillna(0).values
        fast_k = df['k_fast'].fillna(0).values

        return slow_k, fast_k

    @staticmethod
    def calculate_cci(dataraw, ndays):
        """Commodity Channel Index"""
        tp = (dataraw['High'] + dataraw['Low'] + dataraw['Close']) / 3
        raw_cci = pd.Series((tp - tp.rolling(ndays).mean()) / (0.015 * tp.rolling(ndays).std()), name='CCI')
        return_data = dataraw.join(raw_cci)
        return return_data

    @staticmethod
    def normalize(raw_list):
        """Scale feature to 0-1"""
        min_x = min(raw_list)
        max_x = max(raw_list)
        if not min_x or not max_x:
            return raw_list

        return_data = list(map(lambda x: (x - min_x) / (max_x - min_x), raw_list))
        return return_data

    def get_valid_actions(self):
        return [0, 1] if not self.order else [0, 2]

    def get_noncash_reward(self, diff, direction):
        risk_averse = 1
        open_cost = 0
        reward = direction * diff
        if not self.order:
            reward -= open_cost

        if reward < 0:
            reward = reward * risk_averse

        return reward

    def act(self, action):
        # data_frame = self.train_data[self.t-120: self.t]
        # macd, signal, histogram = self.macd(data_frame)
        # cci = self.calculate_cci(data_frame, 14)
        # cci = cci.CCI.dropna().values
        # cci = self.normalize(cci.copy())
        # # normalize data
        # macd = self.normalize(macd.copy())
        # signal = self.normalize(signal.copy())
        # histogram = self.normalize(histogram.copy())

        raw_price = self.raw_price[self.t]
        normalize_price = self.normalize_price[self.t]
        current_ma = self.raw_price[self.t]
        next_ma = self.raw_price[self.t+1]
        next_5_ma = self.raw_price[self.t+5]
        next_14_ma = self.raw_price[self.t+14]

        diff = raw_price - self.order if self.order else 0
        r = 0
        done = False

        if action == 0:
            pass

        elif action == 1:
            # Buy
            if not self.order:
                if next_14_ma - current_ma >= 5:
                    r = 1
                    # done = True

                # else:
                #     r = -0.1
                self.order = raw_price
                # self.normalize_order = normalize_price

        elif action == 2:
            # Sell
            if self.order:
                if current_ma - next_14_ma >= 5:
                    r = 1
                    # done = True

                # else:
                #     r = -0.1
                self.order = 0
                # self.normalize_order = 0
                # self.waiting_time = 0
                self.budget += diff

        # if self.waiting_time > 50:
        #     done = True
        #     # r -= 0.01

        # if self.budget < 0:
        #     done = True
        #     r = 0

        # state_1 = histogram[-self.consecutive_frames:]
        # state_2 = cci[-self.consecutive_frames:]
        # state_3 = macd[-self.consecutive_frames:]
        # state_4 = signal[-self.consecutive_frames:]
        state_5 = self.normalize(self.moving_avg[self.t-self.consecutive_frames:self.t])

        state = np.array([state_5])
        info = {'diff': diff, 'order': 1 if self.order else 0, 'budget': self.budget, 'waiting_time': self.waiting_time}
        state2 = np.array([1, 1, 1, 1])
        self.t += 1
        self.waiting_time += 1
        return state, state2, r, done, info

    def reset(self):
        self.t = 120
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.waiting_time = 0

        inp1 = np.random.rand(1, self.consecutive_frames)
        inp2 = np.random.rand(4,)
        return inp1, inp2

    def get_state_size(self):
        return self.consecutive_frames

    @staticmethod
    def get_action_space():
        return 3


if __name__ == '__main__':
    env = TradingEnv()
