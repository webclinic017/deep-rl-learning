import matplotlib
import pandas as pd
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates


def quotes_info():
    df = pd.read_csv("data/btc_1h.csv")
    # df.Date = df.Date + ' ' + df.Time
    # df = df.drop(['Time'], axis=1)
    df.Open = df.Open.astype("float64")
    df.High = df.High.astype("float64")
    df.Low = df.Low.astype("float64")
    df.Close = df.Close.astype("float64")
    df.Volume = df.Volume.astype("float64")
    df.Date = date2num(pd.to_datetime(df.Date))
    ohlc_daily_date_axis_w_vol(df.iloc[0:50])


def ohlc_daily_date_axis_w_vol(quotes):
    quotes = quotes.values.tolist()

    f1 = plt.subplot2grid((6, 4), (1, 0), rowspan=6, colspan=4)
    candlestick_ohlc(f1, quotes, width=.01, colorup='#53c156', colordown='#ff1717')
    f1.xaxis_date()
    f1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d %H:%M'))

    plt.xticks(rotation=45)
    plt.ylabel('Stock Price')
    plt.xlabel('Date Hours:Minutes')
    plt.show()


if __name__ == '__main__':
    quotes_info()
