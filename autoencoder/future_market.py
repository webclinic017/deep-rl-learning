import datetime
import time
import json
import logging
import pandas as pd
from binance.enums import KLINE_INTERVAL_1HOUR, SIDE_SELL, ORDER_TYPE_MARKET, SIDE_BUY
from binance.websockets import BinanceSocketManager
from pymongo import MongoClient
from talib._ta_lib import MA, MACD, MINUS_DI, PLUS_DI, ADX, BBANDS, SAR, CCI, ROC, ROCP, WILLR
from binance.client import Client
import matplotlib.pyplot as plt
from mailer import SendMail
from binance_f import RequestClient
from binance_f import SubscriptionClient
from binance_f.model import *
from binance_f.constant.test import *
from binance_f.exception.binanceapiexception import BinanceApiException

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


class RegressionMA:
    def __init__(self, trading=False):
        self.global_step = 0
        self.order = 0
        self.buy_mount = 0
        self.side = None  # can order
        with open("config.txt", "r") as file:
            order_data = file.readline()
            order_data = order_data.split(',')
            if float(order_data[0]) != 0:
                self.order = float(order_data[0])
                self.buy_mount = float(order_data[1])
        self.budget = 0
        self.prev_histogram = 0
        self.max_profit = 0
        self.max_loss = 0
        self.take_profit, self.stop_loss = 0, 0
        self.is_latest = False
        self.can_order = True
        self.can_open_order = trading
        self.trade_amount = 0.1  # 10% currency you owned
        self.interval = KLINE_INTERVAL_1HOUR
        self.train_data = pd.DataFrame(columns=['openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
                                                'quoteAssetVolume', 'numTrades', 'takerBuyBaseAssetVolume',
                                                'takerBuyQuoteAssetVolume', 'ignore'])

        # Matplotlib
        self.exp4 = []
        self.exp5 = []
        self.exp6 = []
        self.exp7 = []
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(111, label='ax1')
        # self.ax2 = self.fig.add_subplot(212, label='ax2')

        # Global Config
        self.bbw_threshold = 0.03
        self.adx_threshold = 25
        self.prev_frames = 5
        self.force_close = False

        # Email Services
        self.mailer = SendMail()

        # MongoDB
        self.client = MongoClient(host='66.42.37.163', username='admin', password='jkahsduk12387a89sdjk@#',
                                  authSource='admin')
        self.db = self.client.crypto

        # Binane Client
        self.request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)

    def get_data(self):
        request_client = RequestClient(api_key=g_api_key, secret_key=g_secret_key)

        result = request_client.get_candlestick_data(symbol="BTCUSDT",
                                                     interval=CandlestickInterval.HOUR1,
                                                     startTime=None,
                                                     endTime=None, limit=150)
        dict_data = [x.__dict__ for x in result]
        df = pd.DataFrame(dict_data)
        self.train_data = df

    @classmethod
    def fibonacci(cls, price_max, price_min):
        diff = price_max - price_min
        take_profit = price_max + 0.618 * diff
        stop_loss = price_max - 0.382 * diff
        return take_profit, stop_loss

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        sub_client = SubscriptionClient(api_key=g_api_key, secret_key=g_secret_key)
        sub_client.subscribe_candlestick_event("btcusdt", CandlestickInterval.HOUR1, self.process_message, self.error)

    @staticmethod
    def error(e: 'BinanceApiException'):
        print(e.error_code + e.error_message)

    def process_message(self, data_type: 'SubscribeMessageType', event: 'any'):
        if data_type == SubscribeMessageType.RESPONSE:
            print("Event ID: ", event)
        elif data_type == SubscribeMessageType.PAYLOAD:
            # print("Event type: ", event.eventType)
            # print("Event time: ", event.eventTime)
            # print("Symbol: ", event.symbol)
            # print("Data:")
            # PrintBasic.print_obj()
            # print(event.data)
            # sub_client.unsubscribe_all()

            try:
                msg = event.data.__dict__.copy()
                if 'timestamp' not in msg:
                    msg['timestamp'] = time.time()

                _open_time = msg['startTime']
                _open = msg['close']
                _high = msg['high']
                _low = msg['low']
                _close = msg['close']
                _volume = msg['volume']
                _close_time = msg['closeTime']
                _quote_asset_volume = msg['quoteAssetVolume']
                _number_of_trades = msg['numTrades']
                _buy_base_asset_volume = msg['takerBuyBaseAssetVolume']
                _buy_quote_asset_volume = msg['takerBuyQuoteAssetVolume']
                _ignore = msg['ignore']
                _timestamp = msg['timestamp']

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
                            'openTime', 'open', 'high', 'low', 'close', 'volume', 'closeTime',
                            'quoteAssetVolume', 'numTrades', 'takerBuyBaseAssetVolume',
                            'takerBuyQuoteAssetVolume', 'ignore'
                        ]
                    )

                    self.can_order = True
                    self.global_step += 1
                    self.train_data = self.train_data.append(df, ignore_index=True, sort=False)

                elif len(self.train_data) > 1:
                    self.train_data.at[len(self.train_data) - 1, 'close'] = _close
                    self.train_data.at[len(self.train_data) - 1, 'high'] = _high
                    self.train_data.at[len(self.train_data) - 1, 'low'] = _low
                    self.train_data.at[len(self.train_data) - 1, 'volume'] = _volume

                self.is_latest = msg['isClosed']

                if len(self.train_data) >= 100:
                    self.trading(float(_close), _timestamp, msg['isClosed'])
            except Exception as ex:
                print(ex)
        else:
            print("Unknown Data:")

    def check_profit(self, close_p):
        """
        Tính toán max diff từ khi order
        return True nếu diff < 0
        :param close_p:
        :return:
        """
        if self.order:
            diff = close_p - self.order if self.side == 'buy' else self.order - close_p
            if diff > self.max_profit:
                self.max_profit = diff

            if self.max_profit <= 0:
                self.can_order = False
                self.force_close = True

    def check_loss(self, close_p):
        """
        Kiểm tra mức lỗ, nếu quá 38,2% so với mức lãi tối đa thì đóng order, bạn
        chỉ có thể mở order ở timeframe tiếp theo
        :param close_p:
        :return: boolean
        """
        if self.order:
            diff = close_p - self.order if self.side == 'buy' else self.order - close_p
            if self.max_profit != 0 and diff <= self.max_profit * 0.5:
                # bạn đã lỗ 50% so với max profit
                self.can_order = False
                self.force_close = True
            if diff <= -50:
                # bạn đã lỗ $50
                self.can_order = False
                self.force_close = True

    def trading(self, close_p, _timestamp, is_latest):
        df = self.train_data.copy()
        df['MACD'], df['SIGNAL'], df['HISTOGRAM'] = MACD(df.Close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MINUS_DI'] = MINUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['PLUS_DI'] = PLUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['ADX'] = ADX(df.High, df.Low, df.Close, timeperiod=14)
        df['MA'] = MA(df.Close, timeperiod=9)
        df['BAND_UPPER'], df['BAND_MIDDLE'], df['BAND_LOWER'] = BBANDS(df.Close, 20, 2, 2)
        df['BAND_WIDTH'] = (df['BAND_UPPER'] - df['BAND_LOWER']) / df['BAND_MIDDLE']
        df['CCI'] = CCI(df.High, df.Low, df.Close, timeperiod=20)
        df['SAR'] = SAR(df.High, df.Low)
        df['ROC'] = ROC(df.Close, timeperiod=14)
        df['William'] = WILLR(df.High, df.Low, df.Close, timeperiod=14)
        data = df.dropna()

        # MACD
        macd_data = data.MACD.values
        sinal_data = data.SIGNAL.values
        higtogram_data = data.HISTOGRAM.values

        histogram = higtogram_data[-1]
        histogram_prev = higtogram_data[-self.prev_frames:-1]
        macd = macd_data[-1]
        sinal = sinal_data[-1]

        # Bollinger band
        band_middle_data = data.BAND_MIDDLE.values
        middle_band = band_middle_data[-1]

        # Bollinger band width
        bb_w = data.BAND_WIDTH.values[-self.prev_frames:-1]
        last_bb_w = data.BAND_WIDTH.values[-1]

        # Commodity Channel Index (Momentum Indicators)
        cci_data = data.CCI.values
        cci = cci_data[-1]

        # ADX DMI
        plus_di_data = data.PLUS_DI.values
        minus_di_data = data.MINUS_DI.values
        adx_data = data.ADX.values
        adx = adx_data[-1]
        minus_di = minus_di_data[-1]
        plus_di = plus_di_data[-1]

        # Parabolic SAR
        sar_data = data.SAR.values
        sar = sar_data[-1]

        # ROC
        roc = data.ROC.values[-1]

        # WILLR
        willr = data.William.values[-1]

        current_time_readable = datetime.datetime.fromtimestamp(_timestamp).strftime('%d-%m-%Y %H:%M:%S')
        log_txt = " Price {} | DI- {} | DI+ {} | ADX {} | SAR {} | CCI {} |" \
                  " ROC {} | %William {} | Histogram {} | Middle Band {}".format(round(close_p, 2), round(minus_di, 2),
                                                                                 round(plus_di, 2), round(adx, 2),
                                                                                 round(sar, 2), round(cci, 2),
                                                                                 round(roc, 2), round(willr, 2),
                                                                                 round(histogram, 2),
                                                                                 round(middle_band, 2))
        # print(log_txt)
        console_logger.info(log_txt)

        if is_latest:
            self.check_profit(close_p)

        # tính toán max diff, nếu như diff hiện tại nhỏ
        # hơn 50% max profit thì force close order
        if not self.force_close:
            self.check_loss(close_p)

        # Place Buy Order
        if not self.order and self.can_order and \
                (adx > self.adx_threshold and plus_di > self.adx_threshold) and \
                plus_di > minus_di and \
                close_p > middle_band and \
                close_p > sar and \
                cci > 100 and \
                roc > 1 and \
                histogram > 5 and macd > sinal and \
                willr > -20 and \
                all(last_bb_w > x for x in bb_w) and \
                all(histogram > x for x in histogram_prev):
            # buy signal
            # self.buy_margin()
            self.side = 'buy'
            self.order = close_p
            txt = "{} | Buy Order Price {} | DI- {} | DI+ {} | ADX {} | SAR: {} | CCI {} | ROC {}".format(
                current_time_readable, round(close_p, 2), round(minus_di, 2),
                round(plus_di, 2), round(adx, 2), round(sar, 2), round(cci, 2), round(roc, 2)
            )
            autotrade_logger.info(txt)

            if self.can_open_order:
                try:
                    self.request_client.post_order(symbol="BTCUSDT", side=OrderSide.BUY,
                                                   ordertype=OrderType.MARKET, quantity=self.trade_amount)
                    content = "Open Buy order at price: {}".format(close_p)
                    self.mailer.notification(content)
                except Exception as ex:
                    autotrade_logger.error(ex)

        # CLose Buy Order
        elif self.side == 'buy' and self.order and \
                (close_p < middle_band or close_p < sar or self.force_close):
            # take profit
            # self.close_buy_margin()
            diff = close_p - self.order
            self.budget += diff
            self.reset()
            txt = "{} | Close Buy Order Price {} | DI- {} | DI+ {} | ADX {} | " \
                  "SAR: {} | CCI {} | ROC {} | Diff {} | Budget {} | Max Diff {}".format\
                (current_time_readable, round(close_p, 2), round(minus_di, 2), round(plus_di, 2), round(adx, 2),
                 round(sar, 2), round(cci, 2), round(roc, 2), round(diff, 2), round(self.budget, 2),
                 round(self.max_profit, 2))
            autotrade_logger.info(txt)

            if self.can_open_order:
                try:
                    self.request_client.post_order(symbol="BTCUSDT", side=OrderSide.SELL,
                                                   ordertype=OrderType.MARKET, quantity=self.trade_amount)
                    result = self.request_client.get_balance()
                    for res in result:
                        if res.asset == 'USDT':
                            content = "Close buy order at price: {}, Current balance: {}".format(close_p, res.balance)
                            self.mailer.notification(content)
                except Exception as ex:
                    autotrade_logger.error(ex)

        # Place Sell Order
        elif not self.order and self.can_order and \
                (adx > self.adx_threshold and minus_di > self.adx_threshold) and \
                minus_di > plus_di and \
                close_p < middle_band and \
                close_p < sar and \
                cci < -100 and \
                roc < -1 and \
                histogram < -5 and macd < sinal and \
                willr < -80 and \
                all(last_bb_w > x for x in bb_w) and \
                all(histogram < x for x in histogram_prev):

            self.side = 'sell'
            self.order = close_p
            txt = "{} | Open Sell Order Price {} | DI- {} | DI+ {} | ADX {} | SAR: {} | CCI {} | ROC {}".format(
                current_time_readable,
                round(close_p, 2), round(minus_di, 2), round(plus_di, 2),
                round(adx, 2), round(sar, 2), round(cci, 2), round(roc, 2)
            )
            autotrade_logger.info(txt)

            if self.can_open_order:
                try:
                    self.request_client.post_order(symbol="BTCUSDT", side=OrderSide.SELL,
                                                   ordertype=OrderType.MARKET, quantity=self.trade_amount)
                    content = "Open Sell order at price: {}".format(close_p)
                    self.mailer.notification(content)
                except Exception as ex:
                    autotrade_logger.error(ex)

        # Close Sell Order
        elif self.side == 'sell' and self.order and \
                (close_p > middle_band or close_p > sar or self.force_close):

            diff = self.order - close_p
            self.budget += diff
            self.side = None
            self.reset()
            txt = "{} | Close Sell Order {} | DI- {} | DI+ {} | ADX {} | " \
                  "SAR: {} | CCI {} | ROC {} | Diff {} | Budget {} | Max Diff {} ".format\
                    (current_time_readable, round(close_p, 2), round(minus_di, 2), round(plus_di, 2), round(adx, 2),
                     round(sar, 2), round(cci, 2), round(roc, 2), round(diff, 2), round(self.budget, 2),
                     round(self.max_profit, 2))
            autotrade_logger.info(txt)

            if self.can_open_order:
                try:
                    self.request_client.post_order(symbol="BTCUSDT", side=OrderSide.BUY,
                                                   ordertype=OrderType.MARKET, quantity=self.trade_amount)
                    result = self.request_client.get_balance()
                    for res in result:
                        if res.asset == 'USDT':
                            content = "Close sell order at price: {}, Current balance: {}".format(close_p, res.balance)
                            self.mailer.notification(content)
                except Exception as ex:
                    autotrade_logger.error(ex)

    def reset(self):
        self.order = 0
        self.side = None
        self.take_profit = 0
        self.stop_loss = 0
        self.can_order = False
        self.force_close = False
        self.max_profit = 0


if __name__ == '__main__':
    bottrading = RegressionMA(trading=True)
    bottrading.get_data()
    bottrading.start_socket()
