import pandas as pd
import numpy as np
import ta
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn import preprocessing


class TradingEnv:
    def __init__(self, consecutive_frames=40):
        self.starting_point = 40
        self.t = self.starting_point
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        self.consecutive_frames = consecutive_frames
        df = pd.read_csv(f"D:\Code\Thinh\deep-rl-learning\data\XAUUSD.csv")
        close = df.Close.values
        self.train_data = close[:200]
        self.train_data = preprocessing.StandardScaler().fit_transform(self.train_data.reshape(-1, 1)).flatten()
        self.prices = close[:200]
        self.fig, self.ax = plt.subplots()  # Create a figure containing a single axes.
        # self.ax.plot(self.prices)  # Plot some data on the axes.

    def get_valid_actions(self):
        return [0, 1] if not self.order else [0, 2]

    def step(self, action):
        current_price = self.prices[self.t]
        plt.cla()
        plt.plot(self.prices)  # Plot some data on the axes.
        colors = ['#dbdbdb', '#e31010', '#55cc23']
        plt.scatter(self.t, current_price, color=colors[action])
        plt.pause(0.1)

        diff = current_price - self.prices[self.starting_point]
        r = -1 if action == 1 else -0.01
        done = False
        if action == 1:
            self.budget = diff
            # if diff > 3:
            r = diff
            # done = True
        inp1 = self.train_data[self.t-10:self.t]
        state = np.concatenate((inp1, np.array([diff/np.argmax(self.train_data)])), axis=0)
        self.t += 1
        if self.t == self.train_data.shape[0]:
            self.t = 0
            self.order = 0
            # r = -10
            done = True
        info = {'diff': diff, 'budget': self.budget}
        return state, r, done, info

    def reset(self):
        self.t = self.starting_point
        self.budget = 0
        self.order = 0
        self.normalize_order = 0
        self.done = False
        plt.cla()
        plt.plot(self.prices)  # Plot some data on the axes.
        colors = ['#dbdbdb', '#e31010', '#55cc23']
        plt.scatter(self.t, self.prices[self.t], color="#55cc23")
        plt.pause(0.1)
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
