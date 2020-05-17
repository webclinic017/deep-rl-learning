import pandas as pd
import numpy as np
import ta
from ta import volatility
from ta import momentum
from ta import trend
from ta import volume
from ta.utils import dropna
import math

from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition
from sklearn.externals import joblib


class TradingEnv:
    def __init__(self, consecutive_frames=50, nb_features=15, dataset=None, strategy='train'):
        self.t = consecutive_frames
        self.budget = 100
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.side = None
        self.strategy = strategy
        self.waiting_time = 0
        self.consecutive_frames = consecutive_frames
        self.nb_features = nb_features
        self.train_data, self.prices = self.get_data(dataset)

    def get_data(self, csv_path):
        df = pd.read_csv(csv_path, sep=',')
        df = df.drop(columns=['quote_asset_volume', 'number_of_trades',
                              'buy_base_asset_volume', 'buy_quote_asset_volume', 'open_time', 'close_time', 'ignore'], axis=1)
        df = ta.utils.dropna(df)
        train_data = df.copy()

        data = ta.add_all_ta_features(
            train_data, open='open', high='high', low='low',
            close='close', volume='volume', fillna=True
        )

        raw_price = data.close.astype('float64').values
        data = data.drop(columns=['open', 'high', 'low', 'close', 'volume'], axis=1)

        scaler_filename = 'scaler.pkl'
        pca_filename = 'pca.pkl'
        if self.strategy == 'train':
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaler.fit(data)
            data = scaler.transform(data)
            joblib.dump(scaler, scaler_filename)

            # PCA
            pca = decomposition.PCA(n_components=self.nb_features, random_state=123)
            pca.fit(data)
            joblib.dump(pca, pca_filename)
            data = pca.transform(data)
            print(pca.explained_variance_ratio_)
            print(pca.singular_values_)
        else:
            scaler = joblib.load(scaler_filename)
            pca = joblib.load(pca_filename)
            data = scaler.transform(data)
            # data = data[27:].copy()
            data = pca.transform(data)

        return data, raw_price

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
        return [0, 2] if self.order else [0, 1]

    @staticmethod
    def sigmoid(x):
        if x <= 0:
            return 0
        return 1 / (1 + math.exp(-x))

    def act(self, action):
        current_price = self.prices[self.t]
        done = False

        diff = current_price - self.order if self.order else 0
        r = min(diff / 1000, 1) if diff else 0
        if action not in self.get_valid_actions():
            r = -10

        elif action == 1:
            # Buy
            self.order = current_price
            self.side = 'buy'

        elif action == 2:
            self.budget += diff
            self.side = None
            self.order = 0
            if diff > 100:
                r = 10

        # create state
        state1 = self.train_data[self.t-self.consecutive_frames:self.t+1]

        info = {
            'diff': round(diff, 2),
            'order': 1 if self.order else 0,
            'budget': round(self.budget, 2),
            'waiting_time': self.waiting_time,
            'end_ep': False
        }
        state2 = np.array([1 if self.order else 0, min(diff / 500, 1)])
        self.t += 1
        self.waiting_time += 1

        if self.t >= len(self.prices) - 20:
            done = True
            self.t = self.consecutive_frames
            info['end_ep'] = True

        return state1, state2, r, done, info

    def reset(self):
        if self.strategy == 'test':
            self.t = self.consecutive_frames

        self.budget = 100
        self.side = None
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.waiting_time = 0

        inp1 = self.train_data[self.t-self.consecutive_frames:self.t+1]
        inp2 = np.array([0, 0])
        self.t += 1
        return inp1, inp2


if __name__ == '__main__':
    env = TradingEnv()
