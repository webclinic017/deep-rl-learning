import numpy as np
import matplotlib.pyplot as plt


# prints formatted price
def formatPrice(self, n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
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
        diff = 1
        if diff != 0:
            delta = (current_price - prev_price) / diff
        else:
            delta = 0

        vec.append(delta)  # normalize
        prices.append(float(line.split(delimiter)[0]))

    return vec, prices


x, y = getStockDataVec('btc_test_1_hour')

index = [i for i, val in enumerate(x)]

fig = plt.figure()
ax1 = fig.add_subplot(111)
# ax1.set_title("Plot DAta")
# ax1.set_xlabel('x')
# ax1.set_ylabel('y')
# ax1.set_xticklabels(x)
# ax1.plot(index, x, c='r', label='the data x')
ax1.plot(index, y, c='b', label='the data y')
# leg = ax1.legend()
# plt.locator_params(nbins=len(index) - 1)
plt.show()