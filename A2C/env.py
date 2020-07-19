import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition


class TradingEnv:
    def __init__(self, consecutive_frames=40):
        self.t = 10
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.waiting_time = 0
        self.consecutive_frames = consecutive_frames
        self.train_data = np.load('A2C/X_encoded_pca.npy')
        self.prices = np.load('A2C/raw_price.npy')

    def get_valid_actions(self):
        return [0, 1] if not self.order else [0, 2]

    def step(self, action):
        current_price = self.prices[self.t]

        diff = current_price - self.order if self.order else 0
        r = -1
        done = False

        if action == 1:
            # Buy
            self.order = current_price

        elif action == 2:
            # Sell
            self.order = 0
            self.budget += diff
            if diff > 10:
                done = True

        state = self.train_data[self.t]
        info = {'diff': diff, 'order': 1 if self.order else 0, 'budget': self.budget, 'waiting_time': self.waiting_time}
        self.t += 1
        self.waiting_time += 1
        if self.t >= self.train_data.shape[0]:
            self.t = 0
            done = True
        return state, r, done, info

    def reset(self):
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.waiting_time = 0

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
