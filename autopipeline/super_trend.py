from __future__ import print_function
import json
import time
import MetaTrader5 as mt5
import pytz
import schedule
import sentry_sdk
from datetime import datetime, timedelta
from mt5 import AutoOrder
from utils import logger
from termcolor import colored

mt5_client = AutoOrder()

# sentry_sdk.init(
#     "https://cc11af54279542189f34a16070babe07@o1068161.ingest.sentry.io/6062320",
#
#     # Set traces_sample_rate to 1.0 to capture 100%
#     # of transactions for performance monitoring.
#     # We recommend adjusting this value in production.
#     traces_sample_rate=1.0
# )


def format_text(trend):
    if trend == 'Buy':
        return colored(trend, 'green')
    if trend == 'Sell':
        return colored(trend, 'red')
    if trend == 'Close_Buy':
        return colored(trend, 'yellow')
    if trend == 'Close_Sell':
        return colored(trend, 'yellow')
    if trend == 'Neutral':
        return colored(trend, 'cyan')
    return colored('0', 'blue')


def scheduler_job():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    # if datetime.now().hour % 4 != 0:
    #     return
    time.sleep(50)
    timezone = pytz.timezone("Etc/UTC")
    logger.info("=" * 50)
    logger.info(f"Start job at: {datetime.now(tz=timezone)}")
    with open("config.json") as config_file:
        config = json.load(config_file)

    timeframe_1 = mt5.TIMEFRAME_M15
    timeframe_2 = mt5.TIMEFRAME_M30
    for symbol, value in zip(config.keys(), config.values()):
        # symbol_name = "BTCUSD"
        lot = value.get('lot')
        logger.info("=" * 50)

        df1date, h1_trend, close_signal_1, current_price_1, high_price_1, low_price_1, kijun_sen_1, atr_1, dftrain_1, resistance_1, support_1, outlier_1 = mt5_client.ichimoku_cloud(
            timeframe=timeframe_1, symbol=symbol, save_frame=False, order_label="")
        df2date, h2_trend, close_signal_2, current_price_2, high_price_2, low_price_2, kijun_sen_2, atr_2, dftrain_2, resistance_2, support_2, outlier_2 = mt5_client.ichimoku_cloud(
            timeframe=timeframe_2, symbol=symbol, save_frame=False, order_label="")

        current_trend = '0'
        if h1_trend == h2_trend == "Sell":
            current_trend = "Sell"
        elif h1_trend == h2_trend == "Buy":
            current_trend = "Buy"

        logger.info(f"{symbol} Current Trend {format_text(current_trend)}")
        logger.info(f"{symbol} M15 {df1date} {format_text(h1_trend)} {current_price_1}")
        logger.info(f"{symbol} M30 {df2date} {format_text(h2_trend)} {current_price_2}")

        order_size = mt5_client.check_order_exist(symbol)
        # do not place an order if the symbol order is placed to Metatrader
        if current_trend == "Buy" and order_size != current_trend and resistance_1 and \
                current_price_1 > resistance_1 and outlier_1 != 1 and outlier_2 != 1:
            mt5_client.close_order(symbol)  # close all open positions
            mt5_client.buy_order(symbol, lot=lot, sl=None, tp=None)
        elif current_trend == 'Sell' and order_size != current_trend and support_1 and \
                current_price_1 > support_1 and outlier_1 != 1 and outlier_2 != 1:
            mt5_client.close_order(symbol)  # close all open positions
            mt5_client.sell_order(symbol, lot=lot, sl=None, tp=None)
        elif order_size == 'Sell' and (current_trend == "Neutral" or current_trend == 'Buy' or h2_trend == 'Buy' or
                                       h1_trend == 'Buy' or close_signal_2 == 'Close_Sell'):
            mt5_client.close_order(symbol)  # close all Sell positions
        elif order_size == 'Buy' and (current_trend == "Neutral" or current_trend == 'Sell' or h2_trend == 'Sell' or
                                      h1_trend == 'Sell' or close_signal_2 == 'Close_Buy'):
            mt5_client.close_order(symbol)  # close all Buy positions

        if order_size:
            mt5_client.modify_stoploss(symbol, atr_1)

        logger.info("=" * 50)


def modify_stoploss_thread():
    with open("config.json") as config_file:
        config = json.load(config_file)

    timeframe_1 = mt5.TIMEFRAME_M15
    timeframe_2 = mt5.TIMEFRAME_M30
    for symbol, value in zip(config.keys(), config.values()):
        atr_1 = mt5_client.get_atr(timeframe_1, symbol)

        order_size = mt5_client.check_order_exist(symbol)
        # do not place an order if the symbol order is placed to Metatrader
        if order_size:
            mt5_client.modify_stoploss(symbol, atr_1)


if __name__ == '__main__':
    # Run job every hour at the 42rd minute
    # scheduler_job()
    schedule.every().hours.at(":59").do(scheduler_job)
    schedule.every().hours.at(":14").do(scheduler_job)
    schedule.every().hours.at(":29").do(scheduler_job)
    schedule.every().hours.at(":44").do(scheduler_job)
    schedule.every().minutes.do(modify_stoploss_thread)
    while True:
        schedule.run_pending()
        time.sleep(1)
