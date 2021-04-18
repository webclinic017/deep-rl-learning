import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import preprocessing


class TradingEnv:
    def __init__(self, consecutive_frames=40):
        self.t = 15
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.consecutive_frames = consecutive_frames
        self.train_data = np.load('A2C/raw_price.npy')[:200]
        self.train_data = preprocessing.StandardScaler().fit_transform(self.train_data.reshape(-1,1)).flatten()
        self.prices = np.load('A2C/raw_price.npy')[:200]
        self.fig, self.ax = plt.subplots()  # Create a figure containing a single axes.
        # self.ax.plot(self.prices)  # Plot some data on the axes.

    def get_valid_actions(self):
        return [0, 1] if not self.order else [0, 2]

    def step(self, action):
        current_price = self.prices[self.t]
        plt.cla()
        plt.plot(self.prices)  # Plot some data on the axes.
        colors = ['#dbdbdb', '#55cc23', '#e31010']
        plt.scatter(self.t, current_price, color=colors[action])
        plt.pause(0.1)

        if self.order:
            diff = current_price - self.order
        else:
            diff = 0

        r = -1 if (action == 2 or action == 1) else 0
        done = False
        if action == 0:
            pass
        elif action == 1:
            self.order = self.prices[self.t]
        elif action == 2:
            self.budget = diff
            if diff > 30:
                r = 10
            self.order = 0
            # done = True

        inp1 = self.train_data[self.t-10:self.t]
        state = np.concatenate((inp1, np.array([diff/100])), axis=0)
        self.t += 1
        if self.t == self.train_data.shape[0]:
            self.t = 0
            self.order = 0
            # r = -10
            done = True
        info = {'diff': diff, 'budget': self.budget}
        return state, r, done, info

    def reset(self):
        self.t = 15
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False

        inp1 = self.train_data[self.t-10:self.t]
        state = np.concatenate((inp1, np.array([0])), axis=0)
        self.t += 1
        return state

    def get_state_size(self):
        return self.train_data.shape[1]

    @staticmethod
    def get_action_space():
        return 3


if __name__ == '__main__':
    env = TradingEnv()
