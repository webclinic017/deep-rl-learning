import numpy as np
import math


class Environment:
    def __init__(self):
        self.data, self.prices = self.getStockDataVec('train')
        self.t = 11
        self.windows = 11
        self.budget = 1000
        self.order = 0

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
            current_price = float(line.split(delimiter)[0])
            prev_time = float(lines[_index - 1].split(delimiter)[1])
            prev_price = float(lines[_index - 1].split(delimiter)[0])
            diff = current_time - prev_time
            if diff != 0:
                delta = (current_price - prev_price) / diff
            else:
                delta = current_price - prev_price

            vec.append([delta])  # normalize
            prices.append(float(line.split(delimiter)[0]))

        return vec, prices

    # returns an an n-day state representation ending at time t
    def getState(self):
        if self.t == (len(self.data) - self.windows):
            self.t = 11
            raise Exception("Done")

        d = self.t - self.windows + 1
        block = self.data[d:self.t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:self.t + 1]  # pad with t0
        res = [[1 if self.order > 0 else 0]]
        for i in block:
            res.append(i)

        return np.array(res)

    def reset(self):
        self.t = 11
        self.budget = 1000
        d = self.t - self.windows + 1
        block = self.data[d:self.t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:self.t + 1]  # pad with t0
        res = [[1 if self.order > 0 else 0]]
        for i in block:
            res.append(i)

        return np.array(res)

    def step(self, a):
        r = -1
        info = {'total_profit': self.budget, 'status': 'hold', 'profit': False, 'current': self.prices[self.t]}
        if a == 0:
            r = 0.03
        elif a == 1:
            # buy
            if self.order == 0:
                info['status'] = 'buy'
                self.order = self.prices[self.t]
                r = 0.1
            else:
                r = -0.01
        elif a == 2:
            # close
            if self.order > 0:
                info['status'] = 'sell'
                diff = (self.prices[self.t] - self.order)
                self.budget += diff
                info['total_profit'] = self.budget
                self.order = 0
                if diff > 0:
                    info['profit'] = True
                    r = 0.1 * diff
                else:
                    r = -0.1
            else:
                r = -0.01
        self.t += 1
        state = self.getState()
        # r = 1
        done = True if self.budget > 1100 else False
        return state, r, done, info
