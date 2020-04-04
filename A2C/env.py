import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition


class TradingEnv:
    def __init__(self, consecutive_frames=40):
        self.t = 120
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.waiting_time = 0
        self.consecutive_frames = consecutive_frames
        self.train_data, self.prices = self.get_data('../data/train_3m.csv')

    def get_data(self, csv_path):
        df = pd.read_csv(csv_path, sep=',')
        df = df.drop(columns=['no', 'open_time', 'close_time', 'quote_asset_volume', 'number_of_trades',
                              'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'], axis=1)
        data = ta.add_all_ta_features(df, close='close', open='open',
                                      high='high', low='low', volume='volume', fillna=True)
        raw_price = df.close.astype('float64').values
        data = data.drop(columns=['open', 'high', 'low', 'close', 'volume'], axis=1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = scaler.fit_transform(data)
        pca = decomposition.PCA(n_components=3)
        pca.fit(data)
        X = pca.transform(data)
        return X, raw_price

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
        current_price = self.prices[self.t]

        diff = current_price - self.order if self.order else 0
        r = 0
        done = False

        if action == 0:
            pass

        elif action == 1:
            # Buy
            if not self.order:
                self.order = current_price

        elif action == 2:
            # Sell
            if self.order:
                self.order = 0
                self.budget += diff
                r = diff

        state_5 = self.train_data[self.t-50:self.t]
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
