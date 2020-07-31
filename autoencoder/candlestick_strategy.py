import datetime
import time
import json
import logging
from functools import wraps
from multiprocessing import Pool
import numpy as np
import pandas as pd
from talib._ta_lib import MA, MACD, MINUS_DI, PLUS_DI, ADX, BBANDS, SAR, CCI, ROC, WILLR, OBV, RSI, STOCH
from candlestick import candlestick
import matplotlib.pyplot as plt

# create logger with 'spam_application'
autotrade_logger = logging.getLogger('autotrade')
autotrade_logger.setLevel(logging.INFO)
console_logger = logging.getLogger('console')
console_logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh1 = logging.FileHandler('log/autotrade.log')
fh1.setLevel(logging.INFO)
fh2 = logging.FileHandler('log/console.log')
fh2.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh1.setFormatter(formatter)
fh2.setFormatter(formatter)
# add the handlers to the logger
autotrade_logger.addHandler(fh1)
console_logger.addHandler(fh2)


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__, end - start)
        return result
    return wrapper


class RegressionMA:
    def __init__(self):
        self.global_step = 0
        self.order = 0
        self.buy_mount = 0
        self.side = None  # can order
        self.budget = 0
        self.prev_histogram = 0
        self.max_profit = 0
        self.max_loss = 0
        self.budgets = []
        self.train_data = pd.DataFrame(columns=['open_time', 'open', 'high', 'low', 'close',
                                                'volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                                                'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])
        self.higher_price = 0
        self.lower_price = 0
        self.force_close = False
        self.is_latest = False

        # Matplotlib
        self.exp4 = []
        self.exp5 = []
        self.exp6 = []
        self.exp7 = []
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111, label='ax1')

    def fake_socket(self):
        with open('data/BTCUSDT_5m.json', 'r') as json_file:
            data = json.load(json_file)
            for msg in data:
                self.process_message(msg)

        plt.plot(self.budgets, label='MA')
        plt.show()

    def process_message(self, msg):
        _open_time = msg['k']['t']
        _open = float(msg['k']['o'])
        _high = float(msg['k']['h'])
        _low = float(msg['k']['l'])
        _close = float(msg['k']['c'])
        _volume = float(msg['k']['v'])
        _close_time = msg['k']['T']
        _quote_asset_volume = msg['k']['Q']
        _number_of_trades = msg['k']['n']
        _buy_base_asset_volume = msg['k']['V']
        _buy_quote_asset_volume = msg['k']['q']
        _ignore = msg['k']['B']
        _timestamp = _open_time/1000

        if self.is_latest:
            df = pd.DataFrame(
                [
                    [
                        _open_time, _open, _high, _low, _close, _volume, _close_time,
                        _quote_asset_volume, _number_of_trades, _buy_base_asset_volume,
                        _buy_quote_asset_volume, _ignore
                    ]
                ],
                columns=[
                    'open_time', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'
                ]
            )

            self.global_step += 1
            self.train_data = self.train_data.append(df, ignore_index=True, sort=False)
            # if len(self.train_data) > 10:
            #     self.trading(_timestamp, msg['k']['x'])

        elif len(self.train_data) > 1:
            self.train_data.at[len(self.train_data) - 1, 'close'] = _close
            self.train_data.at[len(self.train_data) - 1, 'high'] = _high
            self.train_data.at[len(self.train_data) - 1, 'low'] = _low
            self.train_data.at[len(self.train_data) - 1, 'volume'] = _volume

        self.is_latest = msg['k']['x']

        if len(self.train_data) > 10:
            self.trading(_timestamp, msg['k']['x'])

    def check_profit(self, close_p, high_p, low_p, is_latest):
        """
        Kiểm tra mức lỗ, nếu quá 38,2% so với mức lãi tối đa thì đóng order, bạn
        chỉ có thể mở order ở timeframe tiếp theo
        :param close_p:
        :param is_latest:
        :param high_p:
        :param low_p:
        :return: boolean
        """
        if self.order:
            if self.side is 'buy':
                max_profit = high_p - self.order
                current_diff = close_p - self.order
                if (high_p > self.higher_price or not self.higher_price) and is_latest:
                    self.higher_price = high_p
                    self.max_profit = max_profit

            else:
                max_profit = self.order - low_p
                current_diff = self.order - close_p
                if (low_p < self.lower_price or not self.lower_price) and is_latest:
                    self.lower_price = low_p
                    self.max_profit = max_profit

            if self.max_profit > 10 and current_diff < self.max_profit * 0.382:
                self.force_close = True

            if current_diff < -20:
                self.force_close = True

    @staticmethod
    def check_buy_signal(candles_df):
        bullish_harami = candles_df.bullish_harami.iat[-1]
        bullish_engulfing = candles_df.bullish_engulfing.iat[-1]
        piercing_pattern = candles_df.piercing_pattern.iat[-1]
        morning_star = candles_df.morning_star.iat[-1]
        dragonfly_doji = candles_df.dragonfly_doji.iat[-1]
        morning_star_doji = candles_df.morning_star_doji.iat[-1]

        if morning_star or bullish_engulfing or bullish_harami or piercing_pattern or dragonfly_doji or morning_star_doji:
            return True
        return False

    @staticmethod
    def check_sell_signal(candles_df):
        bearish_harami = candles_df.bearish_harami.iat[-1]
        bearish_engulfing = candles_df.bearish_engulfing.iat[-1]
        dark_cloud_cover = candles_df.dark_cloud_cover.iat[-1]
        shooting_star = candles_df.shooting_star.iat[-1]
        gravestone_doji = candles_df.gravestone_doji.iat[-1]
        hanging_man = candles_df.hanging_man.iat[-1]

        if bearish_engulfing or bearish_harami or dark_cloud_cover or shooting_star or gravestone_doji or hanging_man:
            return True
        return False

    @staticmethod
    def check_close(candles_df):
        doji = candles_df.doji.iat[-1]
        doji_star = candles_df.doji_star.iat[-1]

        # if doji or doji_star or dragonfly_doji or gravestone_doji or morning_star_doji:
        #     return True
        # return False

    def trading(self, _timestamp, is_latest):
        if not self.order:
            cci = CCI(self.train_data.high, self.train_data.low, self.train_data.close, timeperiod=20).iat[-1]
            self.train_data['BAND_UPPER'], _, self.train_data['BAND_LOWER'] = BBANDS(
                self.train_data.close, 20, 2, 2)
            candles_df = candlestick.inverted_hammer(self.train_data[-20:], target="inverted_hammer")
            candles_df = candlestick.doji_star(candles_df, target="doji_star")
            candles_df = candlestick.bearish_harami(candles_df, target="bearish_harami")
            candles_df = candlestick.bullish_harami(candles_df, target="bullish_harami")
            candles_df = candlestick.dark_cloud_cover(candles_df, target="dark_cloud_cover")
            candles_df = candlestick.doji(candles_df, target="doji")
            candles_df = candlestick.dragonfly_doji(candles_df, target="dragonfly_doji")
            candles_df = candlestick.hanging_man(candles_df, target="hanging_man")
            candles_df = candlestick.gravestone_doji(candles_df, target="gravestone_doji")
            candles_df = candlestick.bearish_engulfing(candles_df, target="bearish_engulfing")
            candles_df = candlestick.bullish_engulfing(candles_df, target="bullish_engulfing")
            candles_df = candlestick.hammer(candles_df, target="hammer")
            candles_df = candlestick.morning_star(candles_df, target="morning_star")
            candles_df = candlestick.morning_star_doji(candles_df, target="morning_star_doji")
            candles_df = candlestick.piercing_pattern(candles_df, target="piercing_pattern")
            candles_df = candlestick.rain_drop(candles_df, target="rain_drop")
            candles_df = candlestick.rain_drop_doji(candles_df, target="rain_drop_doji")
            candles_df = candlestick.star(candles_df, target="star")
            candles_df = candlestick.shooting_star(candles_df, target="shooting_star")

            buy_signal = self.check_buy_signal(candles_df)
            sell_signal = self.check_sell_signal(candles_df)

        high_p = self.train_data.high.iat[-1]
        low_p = self.train_data.low.iat[-1]
        close_p = self.train_data.close.iat[-1]

        upper_band = self.train_data.BAND_UPPER.iat[-1]
        lower_band = self.train_data.BAND_LOWER.iat[-1]

        current_time_readable = datetime.datetime.fromtimestamp(_timestamp).strftime('%Y-%m-%d %H:%M')
        log_txt = " Price {}".format(round(close_p, 2))

        console_logger.info(log_txt)

        self.check_profit(close_p, high_p, low_p, is_latest)

        # Place Buy Order
        if not self.order and buy_signal and cci < -100 and lower_band > close_p > high_p:
            # buy signal
            self.side = 'buy'
            # self.budget -= 3
            self.order = close_p
            txt = "{} | Buy Order Price {} | CCI {} | {}".format(current_time_readable, round(self.order, 2), cci, _timestamp)
            print(txt)
            # autotrade_logger.info(txt)

        # CLose Buy Order
        elif self.side is 'buy' and self.order and self.force_close:
            # take profit
            diff = close_p - self.order
            self.budget += diff
            self.budgets.append(self.budget)
            txt = "{} | Close Buy Order Price {} | Budget {} | Diff {} | Max Diff {} | {}".format(
                current_time_readable, round(close_p, 2),
                round(self.budget, 2), round(diff, 2), round(self.max_profit, 2), _timestamp
            )
            print(txt)
            # autotrade_logger.info(txt)
            self.reset()

        # Place Sell Order
        elif not self.order and sell_signal and cci > 100 and upper_band < close_p < high_p:
            self.side = 'sell'
            self.order = close_p
            # self.budget -= 3
            txt = "{} | Sell Order Price {} | CCI {} | {}".format(current_time_readable, round(self.order, 2), cci, _timestamp)
            print(txt)
            # autotrade_logger.info(txt)

        # Close Sell Order
        elif self.side is 'sell' and self.order and self.force_close:
            diff = self.order - close_p
            self.budget += diff
            self.budgets.append(self.budget)
            txt = "{} | Close Sell Order Price {} | Budget {} | Diff {} | Max Diff {} | {}".format(
                current_time_readable, round(close_p, 2),
                round(self.budget, 2), round(diff, 2), round(self.max_profit, 2), _timestamp
            )
            print(txt)
            self.reset()
            # autotrade_logger.info(txt)

    def reset(self):
        self.order = None
        self.side = None
        self.force_close = False
        self.lower_price = 0
        self.higher_price = 0
        self.max_profit = 0


if __name__ == '__main__':
    bottrading = RegressionMA()
    bottrading.fake_socket()

