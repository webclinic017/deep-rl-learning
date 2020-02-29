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
    lines = [float(x.split(delimiter)[0]) for x in lines[700:1200]]
    min_x = min(lines)
    max_x = max(lines)
    for el in lines:
        normalized = (el - min_x) / (max_x - min_x)
        vec.append(normalized)  # normalize
        prices.append(el)

    return vec, prices


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


vec_data, price_data = getStockDataVec('train')
index = [i for i, val in enumerate(price_data)]

"""MACD"""
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import pyEX as p
ticker = 'AMD'
timeframe = '6m'
# df = p.chartDF(ticker, timeframe)
df = pd.DataFrame({'index': index, 'close': price_data})
df = df[['close']]
df.reset_index(level=0, inplace=True)
df.columns=['ds','y']
plt.plot(df.ds, vec_data, label='Price')
# plt.show()

exp1 = df.y.ewm(span=12, adjust=False).mean()
exp2 = df.y.ewm(span=26, adjust=False).mean()
macd = exp1-exp2
exp3 = macd.ewm(span=9, adjust=False).mean()
# plt.plot(df.ds, macd, label='AMD MACD', color='#EBD2BE')
# plt.show()


exp4 = []
exp5 = []
exp6 = []
exp7 = []
windows = 16
t = windows


def norm_list(list_needed):
    min_x = min(list_needed)
    max_x = max(list_needed)
    if not min_x or not max_x:
        return list_needed

    return_data = list(map(lambda x: (x - min_x) / (max_x - min_x), list_needed))
    return return_data


exp3_cp = list(exp3.copy())
exp3_cp = norm_list(exp3_cp)
threshold = 0.04
vec_threshold = 0.07

for chunk in exp3_cp[windows:]:
    d = t - windows + 1
    block = exp3_cp[d:t + 1]
    min_el = min(block)
    min_el_idx = block.index(min_el) + t - windows
    vec_1 = min_el - block[0]
    vec_2 = block[-1] - min_el
    if abs(vec_1 + vec_2) < threshold and (abs(vec_1) > vec_threshold or abs(vec_2) > vec_threshold):
        # vector dao chieu
        exp4.append(t)
        exp5.append(exp3_cp[t])
    t+=1

t = windows
for chunk in exp3_cp[windows:]:
    d = t - windows + 1
    block = exp3_cp[d:t + 1]
    min_el = max(block)
    min_el_idx = block.index(min_el) + t - windows
    vec_1 = min_el - block[0]
    vec_2 = block[-1] - min_el
    if abs(vec_1 + vec_2) < threshold and (abs(vec_1) > vec_threshold or abs(vec_2) > vec_threshold):
        # vector dao chieu
        exp6.append(t)
        exp7.append(exp3_cp[t])
    t += 1


plt.plot(df.ds, exp3_cp, label='Signal Line', color='#E5A4CB')
plt.legend(loc='upper left')
plt.plot(exp4, exp5, 'ro', color='blue')
plt.plot(exp6, exp7, 'ro', color='red')
plt.show()
