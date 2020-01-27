import numpy as np
import random


class Environment:
    def __init__(self, windows, start_step):
        self.data, self.prices = self.getStockDataVec('train')
        self.t = start_step
        self.start_step = start_step
        self.windows = windows
        self.budget = 1000
        self.btc_amount = 0
        self.order = 0
        self.train_interval = 0

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

            vec.append(delta)  # normalize
            prices.append(float(line.split(delimiter)[0]))

        return vec, prices

    # returns an an n-day state representation ending at time t
    def getState(self):
        done = False
        if self.t == len(self.data) - 2049:
            self.t = self.start_step
            self.train_interval += 1
            done = True

        d = self.t - self.windows + 1
        block = self.data[d:self.t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:self.t + 1]  # pad with t0
        res = []
        for i in block:
            res.append(i)

        return np.array(res), done

    def reset(self):
        # self.t = random.randint(11, len(self.data) - 2048)
        self.t = self.start_step
        self.budget = 1000
        d = self.t - self.windows + 1
        block = self.data[d:self.t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:self.t + 1]  # pad with t0
        res = []
        for i in block:
            res.append(i)

        return np.array(res)

    def step(self, a):
        r = 0
        info = {
            'total_profit': self.budget, 'status': 'hold', 'profit': False,
            'current': self.prices[self.t], 'order': self.order
        }
        if a == 0:
            r = 0
        elif a == 1:
            # buy
            if self.order == 0:
                info['status'] = 'buy'
                self.order = round(self.prices[self.t], 1)
                # r = 0.1
            # else:
            #     r = -0.01
        elif a == 2:
            # close
            if self.order > 0:
                info['status'] = 'sell'
                diff = (self.prices[self.t] - self.order)
                self.budget += round(diff, 1)
                info['total_profit'] = self.budget
                self.order = 0
                if diff > 0:
                    info['profit'] = True
                    r = 0.5
                # else:
                #     r = -0.1
            # else:
            #     r = -0.01
        self.t += 1
        state, done = self.getState()
        return state, r, done, info
