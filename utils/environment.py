import numpy as np
import random
import matplotlib.pyplot as plt
plt.get_backend()


class Environment:
    def __init__(self, windows, start_step):
        self.data, self.prices = self.getStockDataVec('train1hour')
        self.t = start_step
        self.start_step = start_step
        self.windows = windows
        self.budget = 1000
        self.btc_amount = 0
        self.order = 0
        self.train_interval = 0
        self.data_index = [i for i, val in enumerate(self.prices)]
        plt.plot(self.data_index, self.prices, c='b')
        plt.show(block=False)

    # prints formatted price
    def formatPrice(self, n):
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

    # returns the vector containing stock data from a fixed file
    def getStockDataVec(self, key):
        vec = []
        lines = open("data/" + key + ".csv", "r").read().splitlines()
        prices = []
        delimiter = ','
        for _index, line in enumerate(lines[2:]):
            _index = _index + 2
            current_time = float(line.split(delimiter)[1])
            current_price = float(line.split(delimiter)[5])
            prev_time = float(lines[_index - 1].split(delimiter)[1])
            prev_price = float(lines[_index - 1].split(delimiter)[5])
            diff = current_time - prev_time
            if diff != 0:
                delta = (current_price - prev_price)
            else:
                delta = 0

            _open = float(line.split(delimiter)[2])/10000
            _high = float(line.split(delimiter)[3]) / 10000
            _low = float(line.split(delimiter)[4]) / 10000
            _close = float(line.split(delimiter)[5]) / 10000
            _volume = float(line.split(delimiter)[6]) / 1000

            vec.append([_close, _volume])  # normalize
            prices.append(float(line.split(delimiter)[5]))

        return vec, prices

    # returns an an n-day state representation ending at time t
    def getState(self):
        done = False
        if self.t == len(self.data) - 20:
            self.t = self.start_step
            self.train_interval += 1
            plt.cla()
            plt.plot(self.data_index, self.prices, c='b')
            done = True

        d = self.t - self.windows + 1
        block = self.data[d:self.t + 1]

        return np.array(block), done

    def reset(self):
        # self.t = random.randint(11, len(self.data) - 2048)
        # self.t = self.start_step
        self.budget = 1000
        d = self.t - self.windows + 1
        block = self.data[d:self.t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:self.t + 1]  # pad with t0
        return np.array(block)

    def step(self, a):
        r = 0
        done = False
        info = {
            'total_profit': self.budget, 'status': 'nothing', 'profit': False,
            'current': self.prices[self.t], 'order': self.order
        }
        # if a == 0:
        #     r = 0
        if a == 0:
            # buy btc
            info['status'] = 'buy'
            order = self.prices[self.t]
            next_price = self.prices[self.t + 1]
            diff = order - next_price
            self.budget = self.budget - diff
            if diff <= 0:
                info['profit'] = True
                r = 1
                plt.scatter(self.t, self.prices[self.t], color="g")
                plt.draw()
                plt.pause(0.0001)
            else:
                r = -1
                info['profit'] = False
                plt.scatter(self.t, self.prices[self.t], color="r")
                plt.draw()
                plt.pause(0.0001)
            # r = 0.1
        elif a == 1:
            # sell btc
            info['status'] = 'sell'
            order = self.prices[self.t]
            next_price = self.prices[self.t + 1]
            diff = order - next_price
            self.budget += diff

            if diff >= 0:
                plt.scatter(self.t, self.prices[self.t], color="g")
                plt.draw()
                plt.pause(0.0001)
                info['profit'] = True
                r = 1
            else:
                r = -1
                plt.scatter(self.t, self.prices[self.t], color="r")
                plt.draw()
                plt.pause(0.0001)
                info['profit'] = False

        info['total_profit'] = self.budget
        self.t += 1
        state, done = self.getState()
        return state, r, done, info
