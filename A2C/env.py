import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition


class TradingEnv:
    def __init__(self, consecutive_frames=40):
        self.t = 0
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.consecutive_frames = consecutive_frames
        self.train_data = np.load('A2C/X_encoded.npy')[:1000]
        self.prices = np.load('A2C/raw_price.npy')[:1000]

    def get_valid_actions(self):
        return [0, 1] if not self.order else [0, 2]

    def step(self, action):
        current_price = self.prices[self.t]

        diff = current_price - self.prices[0]
        r = diff / 1000
        done = False

        if action == 1:
            self.budget = diff
            if diff > 200:
                r = 100
            done = True
0
        state = self.train_data[self.t]
        self.t += 1
        if self.t == self.train_data.shape[0]:
            self.t = 0
            done = True
        info = {'diff': diff, 'budget': self.budget}
        return state, r, done, info

    def reset(self):
        self.t = 0
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False

        inp1 = self.train_data[self.t]
        self.t += 1
        return inp1

    def get_state_size(self):
        return self.train_data.shape[1]

    @staticmethod
    def get_action_space():
        return 3


if __name__ == '__main__':
    env = TradingEnv()
