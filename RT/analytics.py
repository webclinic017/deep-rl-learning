import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib


class TrendAnalytics:
    def __init__(self):
        self.data = pd.read_csv("../data/1minute.csv")
        self.data1day = pd.read_csv("../data/1day.csv")
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
        self.budget_list = []
        self.cci_data = []
        self.start_idx = 150
        self.end_idx = 15000
        self.order_type = 'sell'

    def imochi_cloud(self):
        for x in range(250, len(self.data)):
            plt.cla()
            data = self.data[x - 250:x]
            exp4 = self.exp4.copy()
            for x4 in exp4:
                if x4 < 1:
                    self.exp4.pop(0)
                    self.exp5.pop(0)

            exp6 = self.exp6.copy()
            for x6 in exp6:
                if x6 < 1:
                    self.exp6.pop(0)
                    self.exp7.pop(0)

            self.exp4 = [x1 - 1 for x1 in self.exp4]
            self.exp6 = [x - 1 for x in self.exp6]

            data['SAR'] = talib.SAR(self.data.High, self.data.Low, acceleration=0.02, maximum=0.02)
            data = data.copy()
            # Calculate Tenkan-sen
            high_9 = data.High.rolling(9).max()
            low_9 = data.Low.rolling(9).min()
            data['tenkan_sen_line'] = (high_9 + low_9) / 2
            # Calculate Kijun-sen
            high_26 = data.High.rolling(26).max()
            low_26 = data.Low.rolling(26).min()
            data['kijun_sen_line'] = (high_26 + low_26) / 2
            # Calculate Senkou Span A
            data['senkou_spna_A'] = ((data.tenkan_sen_line + data.kijun_sen_line) / 2).shift(26)
            # Calculate Senkou Span B
            high_52 = data.High.rolling(52).max()
            low_52 = data.High.rolling(52).min()
            data['senkou_spna_B'] = ((high_52 + low_52) / 2).shift(26)
            # Calculate Chikou Span B
            data['chikou_span'] = data.Close.shift(-26)

            # Plot closing price and parabolic SAR
            # komu_cloud = data[['Close', 'SAR']][:-1].plot(figsize=(12, 7))
            # # Plot Komu cloud
            # komu_cloud.fill_between(data.index[:500], data.senkou_spna_A[:500], data.senkou_spna_B[:500],
            #                         where=data.senkou_spna_A[:500] >= data.senkou_spna_B[:500], color='lightgreen')
            # komu_cloud.fill_between(data.index[:500], data.senkou_spna_A[:500], data.senkou_spna_B[:500],
            #                         where=data.senkou_spna_A[:500] < data.senkou_spna_B[:500], color='lightcoral')
            # plt.grid()
            # plt.legend()
            # plt.show()
            data['signal'] = 0
            data.loc[(data.Close > data.senkou_spna_A) & (data.Close > data.senkou_spna_B) & (data.Close > data.SAR), 'signal'] = 1
            data.loc[(data.Close < data.senkou_spna_A) & (data.Close < data.senkou_spna_B) & (data.Close < data.SAR), 'signal'] = -1
            data['signal'].value_counts()
            self.trading(data)

    def trading(self, data):
        trading_data = data.copy()
        self.ax1.cla()
        cci = self.calculate_cci(trading_data, 25)
        cci = cci.CCI.fillna(0).values
        close_price = trading_data.Close.values
        sar_signal = trading_data.SAR.values
        mean_cci = np.mean(cci[-5:])
        self.ax1.plot(cci, label='CCI: {}'.format(mean_cci))
        self.ax2.plot(sar_signal, 'ro', label='Sar Signal: {}'.format(self.budget), color='k', markersize=0.5)
        self.ax2.plot(close_price, label='Raw data')
        self.ax2.legend(loc='upper left')
        self.ax1.legend(loc='upper left')
        _idx = 0
        for index, data in enumerate(zip(close_price, sar_signal)):
            price, sar = data
            if _idx == len(close_price) - 1:
                current = sar_signal[_idx]
                trend_up = True if current < price else False
                trend_down = True if current > price else False
                if self.order_type != 'sell' and trend_down:
                    self.exp6.append(_idx)
                    self.exp7.append(price)
                    self.budget += price - self.order
                    self.order_type = 'sell'
                    self.order = 0
                elif self.order_type != 'buy' and trend_up:
                    self.exp4.append(_idx)
                    self.exp5.append(price)
                    self.order = price
                    self.order_type = 'buy'

                if self.order and (price - self.order) > (price * 0.1):
                    self.exp6.append(_idx)
                    self.exp7.append(price)
                    self.budget += price - self.order
                    self.order_type = 'sell'
                    self.order = 0

                if self.order and (price - self.order) < -5:
                    self.exp6.append(_idx)
                    self.exp7.append(price)
                    self.budget += price - self.order
                    self.order_type = 'sell'
                    self.order = 0

            _idx += 1

        # self.ax2.plot(self.exp4, self.exp5, 'ro', color='g', markersize=5)
        # self.ax2.plot(self.exp6, self.exp7, 'ro', color='r', markersize=5)
        # plt.pause(0.001)
        print(self.budget)

    def strategy(self):
        data = self.data.copy()
        # Calculate daily returns
        daily_returns = data.Close.pct_change()
        # Calculate strategy returns
        strategy_returns = daily_returns * data['signal'].shift(1)
        # Calculate cumulative returns
        (strategy_returns + 1).cumprod().plot(figsize=(10, 5))
        # Plot the strategy returns
        plt.xlabel('Date')
        plt.ylabel('Strategy Returns (%)')
        plt.grid()
        plt.show()

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

    @staticmethod
    def calculate_cci(dataraw, ndays):
        """Commodity Channel Index"""
        tp = (dataraw['High'] + dataraw['Low'] + dataraw['Close']) / 3
        raw_cci = pd.Series((tp - tp.rolling(ndays).mean()) / (0.015 * tp.rolling(ndays).std()), name='CCI')
        return_data = dataraw.join(raw_cci)
        return return_data

    def test_trading(self):
        _idx = self.start_idx
        total_profit = 0
        total_loss = 0
        for current_moving, price in zip(self.moving_data[self.start_idx:self.end_idx], self.real_price[self.start_idx:self.end_idx]):
            std_moving = self.moving_data[_idx - 5]
            first_moving = self.moving_data[_idx - 7]

            # if self.moving_avg[_idx-75] < self.moving_avg[_idx]:
            if not self.order and first_moving > std_moving < current_moving < -0.2:
                # buy signal
                self.exp4.append(_idx - self.start_idx)
                self.exp5.append(price)
                self.order = price

            if self.order and first_moving < std_moving > current_moving:
                diff = price - self.order
                if diff > 5:
                    self.exp6.append(_idx - self.start_idx)
                    self.exp7.append(price)
                    self.budget += diff
                    if diff > 0:
                        total_profit += diff
                    else:
                        total_loss += diff

                    self.order = 0

            if self.order and min(self.real_price[_idx-50:_idx]) > self.order:
                self.exp6.append(_idx - self.start_idx)
                self.exp7.append(price)
                diff = price - self.order
                total_loss += diff
                self.order = 0
                self.budget += diff

            _idx += 1

        self.ax2.plot(self.exp4, self.exp5, 'ro', color='g')
        self.ax2.plot(self.exp6, self.exp7, 'ro', color='r')
        plt.show()
        print("profit: {} total loss: {} total profit : {}".format(self.budget, total_loss, total_profit))


if __name__ == '__main__':
    bot = TrendAnalytics()
    # bot.plot_data()
    bot.imochi_cloud()
    # bot.strategy()

