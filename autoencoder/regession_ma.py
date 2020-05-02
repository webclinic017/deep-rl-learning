import datetime
import time
import json
import logging
import pandas as pd
from binance.enums import KLINE_INTERVAL_1HOUR, SIDE_SELL, ORDER_TYPE_MARKET, SIDE_BUY
from binance.websockets import BinanceSocketManager
from pymongo import MongoClient
from talib._ta_lib import MA, MACD, MINUS_DI, PLUS_DI, ADX, BBANDS, SAR, STOCHRSI, CCI, ROC, ROCR100, ROCP, WILLR
from binance.client import Client
import matplotlib.pyplot as plt
from mailer import SendMail

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
    def __init__(self):
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
        self.force_close = False
        self.trade_amount = 0.5  # 75% currency you owned
        self.api_key = "9Hj6HLNNMGgkqj6ngouMZD1kjIbUb6RZmIpW5HLiZjtDT5gwhXAzc20szOKyQ3HW"
        self.api_secret = "ioD0XICp0cFE99VVql5nuxiCJEb6GK8mh08NYnSYdIUfkiotd1SZqLTQsjFvrXwk"
        self.binace_client = Client(self.api_key, self.api_secret)
        self.bm = BinanceSocketManager(self.binace_client)
        self.interval = KLINE_INTERVAL_1HOUR
        self.train_data = pd.DataFrame(columns=['open_time', 'Open', 'High', 'Low', 'Close',
                                                'Volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                                                'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])

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
        self.adx_threshold = 22
        self.prev_frames = 3

        # Email Services
        self.mailer = SendMail()

        # MongoDB
        self.client = MongoClient(host='66.42.37.163', username='admin', password='jkahsduk12387a89sdjk@#',
                                  authSource='admin')
        self.db = self.client.crypto

    def get_data(self):
        api_key = "y9JKPpQ3B2zwIRD9GwlcoCXwvA3mwBLiNTriw6sCot13IuRvYKigigXYWCzCRiul"
        api_secret = "uUdxQdnVR48w5ypYxfsi7xK6e6W2v3GL8YrAZp5YeY1GicGbh3N5NI71Pss0crfJ"
        binaci_client = Client(api_key, api_secret)
        klines = binaci_client.get_historical_klines("BTCUSDT", self.interval, "15 April, 2020")
        df = pd.DataFrame(klines, columns=['open_time', 'Open', 'High', 'Low', 'Close',
                                           'Volume', 'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'])
        self.train_data = df

    @classmethod
    def fibonacci(cls, price_max, price_min):
        diff = price_max - price_min
        take_profit = price_max + 0.618 * diff
        stop_loss = price_max - 0.382 * diff
        return take_profit, stop_loss

    def start_socket(self):
        # start any sockets here, i.e a trade socket
        conn_key = self.bm.start_kline_socket('BTCUSDT', self.process_message, interval=self.interval)
        # then start the socket manager
        self.bm.start()

    def fake_socket(self, crawler=False):
        data = self.db.BTCUSDT_1h.find({})
        data = list(data)
        with open('data/BTCUSDT_1h.json', 'w') as outfile:
            json.dump(data, outfile, indent=4)
        with open('data/BTCUSDT_1h.json') as json_file:
            data = json.load(json_file)
            for msg in data:
                self.process_message(msg)
        json_file.close()

        # with open('data/Bitcoin240.csv') as json_file:
        #     for line in json_file.readlines():
        #         data = line.split(',')
        #         date_data = "{} {}".format(data[0], data[1])
        #         time_obj = datetime.datetime.strptime(date_data, "%Y.%m.%d %H:%M")
        #         msg = {
        #             "e": "kline",  # event type
        #             "E": time_obj,  # event time
        #             "s": "ETHBTC",  # symbol
        #             "k": {
        #                 "t": 1499404860000,  # start time of this bar
        #                 "T": 1499404919999,  # end time of this bar
        #                 "s": "ETHBTC",  # symbol
        #                 "i": "1m",  # interval
        #                 "f": 77462,  # first trade id
        #                 "L": 77465,  # last trade id
        #                 "o": data[2],  # open
        #                 "c": data[5],  # close
        #                 "h": data[3],  # high
        #                 "l": data[4],  # low
        #                 "v": data[6],  # volume
        #                 "n": 4,  # number of trades
        #                 "x": True,  # whether this bar is final
        #                 "q": "1.79662878",  # quote volume
        #                 "V": "2.34879839",  # volume of active buy
        #                 "Q": "0.24142166",  # quote volume of active buy
        #                 "B": "13279784.01349473"  # can be ignored
        #             }
        #         }
        #         self.global_step += 1
        #         self.process_message(msg)

    def process_message(self, msg):
        # if 'timestamp' not in msg:
        #     msg['timestamp'] = time.time()

        _open_time = msg['k']['t']
        _open = msg['k']['o']
        _high = msg['k']['h']
        _low = msg['k']['l']
        _close = msg['k']['c']
        _volume = msg['k']['v']
        _close_time = msg['k']['T']
        _quote_asset_volume = msg['k']['Q']
        _number_of_trades = msg['k']['n']
        _buy_base_asset_volume = msg['k']['V']
        _buy_quote_asset_volume = msg['k']['q']
        _ignore = msg['k']['B']
        _timestamp = msg['E']/1000

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
                    'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'buy_base_asset_volume', 'buy_quote_asset_volume', 'ignore'
                ]
            )

            self.can_order = True
            self.global_step += 1
            self.train_data = self.train_data.append(df, ignore_index=True, sort=False)

        elif len(self.train_data) > 1:
            self.train_data.at[len(self.train_data) - 1, 'Close'] = _close
            self.train_data.at[len(self.train_data) - 1, 'High'] = _high
            self.train_data.at[len(self.train_data) - 1, 'Low'] = _low
            self.train_data.at[len(self.train_data) - 1, 'Volume'] = _volume

        self.is_latest = msg['k']['x']

        if len(self.train_data) > 100:
            self.trading(float(_close), _timestamp, msg['k']['x'])

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
                # bạn đã lỗ 38.2% so với max profit
                self.can_order = False
                self.force_close = True
            if diff <= -50:
                # bạn đã lỗ $30
                self.can_order = False
                self.force_close = True

    def trading(self, close_p, _timestamp, is_latest):
        df = self.train_data.copy()
        df['Close'] = df['Close'].astype('float64')
        df['High'] = df['High'].astype('float64')
        df['Low'] = df['Low'].astype('float64')
        df['Open'] = df['Open'].astype('float64')
        df['MACD'], df['SIGNAL'], df['HISTOGRAM'] = MACD(df.Close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['MINUS_DI'] = MINUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['PLUS_DI'] = PLUS_DI(df.High, df.Low, df.Close, timeperiod=14)
        df['ADX'] = ADX(df.High, df.Low, df.Close, timeperiod=14)
        df['MA'] = MA(df.Close, timeperiod=9)
        df['BAND_UPPER'], df['BAND_MIDDLE'], df['BAND_LOWER'] = BBANDS(df.Close, 20, 2, 2)
        df['BAND_WIDTH'] = (df['BAND_UPPER'] - df['BAND_LOWER']) / df['BAND_MIDDLE']
        df['BAND_B'] = (df['Close'] - df['BAND_LOWER']) / (df['BAND_UPPER'] - df['BAND_LOWER'])
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
        bb_b = data.BAND_B.values[-1]

        # Commodity Channel Index (Momentum Indicators)
        cci_data = data.CCI.values
        cci = cci_data[-1]

        # ADX DMI
        plus_di_data = data.PLUS_DI.values
        minus_di_data = data.MINUS_DI.values
        adx_data = data.ADX.values
        adx = adx_data[-1]
        prev_adx = adx_data[-self.prev_frames:-1]
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
        log_txt = " Price {} | DI- {} | DI+ {} | ADX {} | SAR {} | CCI {} | ROC {} | %William {} | Histogram {} | Middle Band {}".format(
            round(close_p, 2),
            round(minus_di, 2), round(plus_di, 2), round(adx, 2),
            round(sar, 2), round(cci, 2), round(roc, 2),
            round(willr, 2), round(histogram, 2), round(middle_band, 2)
        )
        # print(log_txt)
        console_logger.info(log_txt)

        # tính toán max diff, nếu như diff hiện tại nhỏ hơn 61,8% max profit thì force close order
        if is_latest:
            self.check_profit(close_p)

        if not self.force_close:
            self.check_loss(close_p)

        # force_close = False
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
                all(histogram > x for x in histogram_prev) and \
                all(adx > x for x in prev_adx) and \
                bb_b > 1:
            # buy signal
            self.side = 'buy'
            self.order = close_p
            txt = "{} | Buy Order Price {} | DI- {} | DI+ {} | ADX {} | SAR: {} | CCI {} | ROC {}".format(
                current_time_readable, round(close_p, 2), round(minus_di, 2),
                round(plus_di, 2), round(adx, 2), round(sar, 2), round(cci, 2), round(roc, 2)
            )
            print(txt)
            autotrade_logger.info(txt)

        # CLose Buy Order
        elif self.side == 'buy' and self.order and \
                (close_p < middle_band or close_p < sar or self.force_close):
            # take profit
            # self.close_buy_margin()
            diff = close_p - self.order
            self.budget += diff
            self.reset()
            txt = "{} | Close Buy Order Price {} | DI- {} | DI+ {} | ADX {} | " \
                  "SAR: {} | CCI {} | ROC {} | Diff {} | Budget {} | Max Diff {}".format(
                current_time_readable,
                round(close_p, 2), round(minus_di, 2), round(plus_di, 2), round(adx, 2), round(sar, 2),
                round(cci, 2), round(roc, 2), round(diff, 2), round(self.budget, 2), round(self.max_profit, 2))
            autotrade_logger.info(txt)
            print(txt)
            self.max_profit = 0

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
                all(histogram < x for x in histogram_prev) and \
                all(adx > x for x in prev_adx) and \
                bb_b < 0:

            self.side = 'sell'
            self.order = close_p
            txt = "{} | Sell Order Price {} | DI- {} | DI+ {} | ADX {} | SAR: {} | CCI {} | ROC {}".format(
                current_time_readable,
                round(close_p, 2),
                round(minus_di, 2),
                round(plus_di, 2),
                round(adx, 2),
                round(sar, 2),
                round(cci, 2),
                round(roc, 2)
            )
            print(txt)
            autotrade_logger.info(txt)

        # Close Sell Order
        elif self.side == 'sell' and self.order and \
                (close_p > middle_band or close_p > sar or self.force_close):

            diff = self.order - close_p
            self.budget += diff
            self.side = None
            self.reset()
            txt = "{} | Close Sell Order {} | DI- {} | DI+ {} | ADX {} | " \
                  "SAR: {} | CCI {} | ROC {} | Diff {} | Budget {} | Max Diff {} ".format(
                current_time_readable,
                round(close_p, 2), round(minus_di, 2), round(plus_di, 2), round(adx, 2),
                round(sar, 2), round(cci, 2), round(roc, 2), round(diff, 2), round(self.budget, 2),
                round(self.max_profit, 2),
            )
            self.max_profit = 0
            print(txt)
            autotrade_logger.info(txt)

    def reset(self):
        self.order = 0
        self.side = None
        self.take_profit = 0
        self.stop_loss = 0
        self.can_order = False
        self.force_close = False

    def test_buy_order(self):
        info = self.binace_client.get_margin_account()
        get_margin_asset = self.binace_client.get_margin_asset(asset='BTC')

        usdt_amount = info['userAssets'][2]['free']
        details = self.binace_client.get_max_margin_transfer(asset='BTC')
        print("Margin Lever: {}".format(info['marginLevel']))
        print("Amount in BTC: {}".format(info['totalAssetOfBtc']))
        print("Account Status: {}".format(info['tradeEnabled']))

        self.buy_margin()
        time.sleep(1)
        self.close_buy_margin()

        print("Test Done!!")

    def test_sell_order(self):
        self.borrow_btc()
        time.sleep(1)
        self.repay_btc()
        info = self.binace_client.get_margin_account()
        print("Margin Lever: {}".format(info['marginLevel']))
        print("Amount in BTC: {}".format(info['totalAssetOfBtc']))
        print("Account Status: {}".format(info['tradeEnabled']))
        print("Test Done!!")

    def borrow_btc(self):
        """
        Borrow BTC with amount = 0.5 * max amount
        :return:
        """
        try:
            # Borrow BTC
            get_max_margin_loan = self.binace_client.get_max_margin_loan(asset='BTC')
            amount = float(get_max_margin_loan['amount'])
            precision = 5
            amt_str = "{:0.0{}f}".format(amount * self.trade_amount, precision)
            transaction = self.binace_client.create_margin_loan(asset='BTC', amount=amt_str)

            # Sell borrowed BTC
            symbol = 'BTCUSDT'
            sell_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str
            )
            txt = 'You borrow: {}'.format(amt_str)
            autotrade_logger.info(txt)
        except Exception as ex:
            autotrade_logger.error(ex)
            self.mailer.notification(str(ex))

    def repay_btc(self):
        """
        Trả lại số BTC đã vay bằng cách mua BTC sau đó mở lệnh repay
        :return:
        """
        try:
            # check amount you borrowed
            symbol = 'BTCUSDT'
            symbol_detail = 'BTC'
            # check account info
            info = self.binace_client.get_margin_account()
            for market in info['userAssets']:
                if market['asset'] == symbol_detail:
                    btc_borrowed = market['borrowed']
                    precision = 5
                    amt_str = "{:0.0{}f}".format(float(btc_borrowed), precision)

                    # # buy btc
                    buy_order = self.binace_client.create_margin_order(
                        symbol=symbol,
                        side=SIDE_BUY,
                        type=ORDER_TYPE_MARKET,
                        quantity=amt_str
                    )

            info = self.binace_client.get_margin_account()
            for market in info['userAssets']:
                if market['asset'] == symbol_detail:
                    btc_free = market['free']
                    precision = 5
                    amt_str = "{:0.0{}f}".format(float(btc_free), precision)
                    # repay
                    transaction = self.binace_client.repay_margin_loan(asset='BTC', amount=btc_free)
                    txt = 'You repay : {} {}'.format(amt_str, symbol_detail)
                    autotrade_logger.info(txt)
        except Exception as ex:
            autotrade_logger.error(ex)
            self.mailer.notification(str(ex))

    def buy_margin(self):
        try:
            symbol = 'BTCUSDT'
            info = self.binace_client.get_margin_account()
            usdt_amount = info['userAssets'][2]['free']
            price_index = self.binace_client.get_margin_price_index(symbol=symbol)
            amount = int(float(usdt_amount)) / float(price_index['price'])
            precision = 5
            amt_str = "{:0.0{}f}".format(amount * self.trade_amount, precision)
            txt = "Buy successfully | Amount {} | Price {}".format(amt_str, price_index['price'])
            buy_order = self.binace_client.create_margin_order(
                symbol=symbol,
                side=SIDE_BUY,
                type=ORDER_TYPE_MARKET,
                quantity=amt_str)
            autotrade_logger.info(txt)
            self.buy_mount = amt_str
            with open("config.txt", "w") as file:
                file.write("{},{}".format(price_index['price'], amt_str))
            return True
        except Exception as ex:
            self.mailer.notification(str(ex))
            autotrade_logger.error(ex)
            return False

    def close_buy_margin(self):
        try:
            symbol = 'BTCUSDT'
            symbol_detail = 'BTC'
            amt_str = 0
            # check account info
            info = self.binace_client.get_margin_account()

            for market in info['userAssets']:
                if market['asset'] == symbol_detail:
                    btc_free = market['free']
                    precision = 5
                    # amt_str = btc_free
                    amt_str = "{:0.0{}f}".format(float(btc_free) - 0.0002, precision)

                    # Sell btc
                    buy_order = self.binace_client.create_margin_order(
                        symbol=symbol,
                        side=SIDE_SELL,
                        type=ORDER_TYPE_MARKET,
                        quantity=amt_str
                    )

            price_index = self.binace_client.get_margin_price_index(symbol=symbol)
            info = self.binace_client.get_margin_account()
            current_btc = info['totalAssetOfBtc']
            usdt = info['userAssets'][2]['free']
            txt = "Sell successfully | Balance {} | Sell Amount {} | At Price {} | Owned In BTC {}".format(
                usdt,
                amt_str,
                price_index['price'],
                current_btc
            )
            autotrade_logger.info(txt)
            with open("config.txt", "w") as file:
                file.write("{},{}".format(0, 0))
            return True
        except Exception as ex:
            self.mailer.notification(str(ex))
            autotrade_logger.error(ex)
            return False

    def plot_data(self):
        df = pd.read_csv("../data/new/btc.csv")
        df = df[20000:]
        plt.plot(df.Close, label='MA')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    bottrading = RegressionMA()
    # bottrading.plot_data()
    # bottrading.get_data()
    # bottrading.test_trading()
    bottrading.fake_socket()
    # bottrading.start_socket()
    # bottrading.test_buy_order()
    # bottrading.test_sell_order()
