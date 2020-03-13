from pandas_datareader import data as web
import matplotlib.pyplot as plt
import fix_yahoo_finance
import pandas as pd
import numpy as np

# Commodity Channel Index
def CalculateCCI(dataRaw, ndays):
    TP = (dataRaw['High'] + dataRaw['Low'] + dataRaw['Close']) / 3
    rawCCI = pd.Series((TP - pd.rolling_mean(TP, ndays)) / (0.015 * pd.rolling_std(TP, ndays)), name='CCI')
    return_data = dataRaw.join(rawCCI)
    return return_data


# Retrieve the Nifty data from Yahoo finance:
# data = web.get_data_yahoo("^NSEI", start="2019-01-01", end="2019-03-13")
order = 0
budget = 0
max_cci = 0
plt.show()
fig = plt.figure(figsize=(7, 5))
header_list = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
data = pd.read_csv("data/1hour.csv", sep=',')
for x in range(len(data)):
    data1 = data.iloc[x:x+60, :]

    # Compute the Commodity Channel Index(CCI) for NIFTY based on the 20-day Moving average
    n = 20
    NI = CalculateCCI(data1, n)
    CCI = NI['CCI']
    current_cci = list(CCI)[-1]
    is_close_buy_signal = list(np.isclose([current_cci], [100.0], atol=1))[0]
    if is_close_buy_signal and order == 0:
        max_cci = current_cci
        order = list(data1['Close'])[-1]
        # print("buy: {}".format(current_cci))

    is_close_positive_120 = list(np.isclose([current_cci], [150.0], atol=2))[0]
    if is_close_positive_120 and order != 0 and current_cci > 200:
        budget += list(data1['Close'])[-1] - order
        print("take profit: {} budget: {}".format(current_cci, budget))
        max_cci = 0
        order = 0

    is_close_positive_100 = list(np.isclose([current_cci], [100.0], atol=2))[0]
    if is_close_positive_100 and order != 0 and current_cci < 100:
        budget += list(data1['Close'])[-1] - order
        print("stop loss: {} budget: {}".format(current_cci, budget))
        max_cci = 0
        order = 0

    # Plotting the Price Series chart and the Commodity Channel index below
    # ax = fig.add_subplot(2, 1, 1)
    # ax.set_xticklabels([])
    # plt.plot(data1['Close'], lw=1)
    # plt.title('BTCUSDT Chart')
    # plt.ylabel('Close Price')
    # plt.grid(True)
    # bx = fig.add_subplot(2, 1, 2)
    # plt.plot(CCI, 'k', lw=0.75, linestyle='-', label='CCI')
    # plt.legend(loc=2, prop={'size': 9.5})
    # plt.ylabel('CCI values')
    # plt.grid(True)
    # plt.setp(plt.gca().get_xticklabels(), rotation=30)
    # plt.pause(0.0001)
    # ax.cla()
    # bx.cla()
    # plt.cla()
