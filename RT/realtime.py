import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class TrendAnalytics:
    def __init__(self):
        self.data = pd.read_csv("../data/train.csv")
        self.real_price = self.data.c.values
        self.moving_avg = self.data.c.rolling(window=14).mean().fillna(7248).values
        self.timestamp = self.data.timestamp.values
        self.moving = list()
        self.moving_data = list()
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(211, label='ax1')
        self.ax2 = self.fig.add_subplot(212, label='ax2')
        self.budget = 0
        self.order = 0
        self.windows = 100
        self.exp5 = []
        self.exp6 = []
        self.exp4 = []
        self.exp7 = []
        self.start_idx = 150
        self.end_idx = -1

    def plot_data(self):
        prev_price = 7248
        prev_timestamp = 1577775996.11432
        for price, moving in zip(self.moving_avg, self.timestamp):
            diff = (price - prev_price) / (moving - prev_timestamp)
            diff = diff if abs(diff) > 0.03 else 0
            prev_timestamp = moving
            prev_price = price
            self.moving.append(diff)

        moving_df = pd.DataFrame(self.moving, columns=['ma'])
        moving_df = moving_df.ma.rolling(window=14).mean().fillna(0).values
        self.moving_data = moving_df
        self.ax1.plot(moving_df[self.start_idx:self.end_idx], label='Moving')
        self.ax2.plot(self.real_price[self.start_idx:self.end_idx], label='Raw data')

    def test_trading(self):
        _idx = self.start_idx
        total_profit = 0
        total_loss = 0
        for current_moving, price in zip(self.moving_data[self.start_idx:self.end_idx], self.real_price[self.start_idx:self.end_idx]):
            std_moving = self.moving_data[_idx - 5]
            first_moving = self.moving_data[_idx - 7]

            if self.moving_avg[_idx-75] < self.moving_avg[_idx]:
                if not self.order and first_moving > std_moving < current_moving < -0.06:
                    # buy signal
                    self.exp4.append(_idx - self.start_idx)
                    self.exp5.append(price)
                    self.order = price

                elif not self.order and abs(current_moving - std_moving) > 0.01 and 0.01 > current_moving > 0 and first_moving < std_moving < current_moving:
                    self.exp4.append(_idx - self.start_idx)
                    self.exp5.append(price)
                    self.order = price

            elif self.order and first_moving < std_moving > current_moving >= -0.02 and self.moving_avg[_idx-70] > self.moving_avg[_idx]:
                self.exp6.append(_idx - self.start_idx)
                self.exp7.append(price)
                diff = price - self.order
                self.budget += diff
                if diff > 0:
                    total_profit += diff
                else:
                    total_loss += diff

                self.order = 0

            elif self.order and min(self.real_price[_idx-14:_idx]) < self.order:
                self.exp6.append(_idx - self.start_idx)
                self.exp7.append(price)
                diff = price - self.order
                total_loss += diff
                self.order = 0
                self.budget += diff

            _idx += 1

        # self.ax2.plot(self.exp4, self.exp5, 'ro', color='g')
        # self.ax2.plot(self.exp6, self.exp7, 'ro', color='r')
        # plt.show()
        print("profit: {} total loss: {} total profit : {}".format(self.budget, total_loss, total_profit))


if __name__ == '__main__':
    bot = TrendAnalytics()
    bot.plot_data()
    bot.test_trading()

