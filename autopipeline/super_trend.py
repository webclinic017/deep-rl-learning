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
    return colored('Neutral', 'blue')


def scheduler_job():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    if datetime.now().minute % 5 != 0:
        return

    logger.info(f"Start job at: {datetime.now()}")
    logger.info("=" * 50)
    with open("config.json") as config_file:
        config = json.load(config_file)

    timezone = pytz.timezone("Etc/UTC")
    current_time = datetime.now(tz=timezone) + timedelta(hours=1, minutes=59)
    for symbol, value in zip(config.keys(), config.values()):
        # symbol_name = "BTCUSD"
        lot = value.get('lot')
        current_price, m30_trend, dfdate = mt5_client.get_frames(utc_from=current_time - timedelta(days=7),
                                                                 utc_to=current_time,
                                                                 timeframe=mt5.TIMEFRAME_M5, symbol=symbol)
        price_h1, h1_trend, h1date = mt5_client.get_frames(utc_from=current_time - timedelta(days=14),
                                                           utc_to=current_time,
                                                           timeframe=mt5.TIMEFRAME_M15, symbol=symbol)
        price_h4, h4_trend, h4date = mt5_client.get_frames(utc_from=current_time - timedelta(days=35),
                                                           utc_to=current_time,
                                                           timeframe=mt5.TIMEFRAME_M30, symbol=symbol)
        price_d1, d1_trend, d1date = mt5_client.get_frames(utc_from=current_time - timedelta(days=100),
                                                           utc_to=current_time,
                                                           timeframe=mt5.TIMEFRAME_H1, symbol=symbol)

        current_trend = '0'
        if m30_trend == h1_trend == h4_trend == d1_trend == "Sell":
            current_trend = "Sell"
        elif m30_trend == h1_trend == h4_trend == d1_trend == "Buy":
            current_trend = "Buy"
        elif h1_trend == h4_trend == "0":
            current_trend = "Neutral"

        logger.info("=" * 50)
        logger.info(current_time)
        logger.info(f"{symbol} M5 {dfdate} {format_text(m30_trend)} {current_price}")
        logger.info(f"{symbol} H15 {h1date} {format_text(h1_trend)} {price_h1}")
        logger.info(f"{symbol} H30 {h4date} {format_text(h4_trend)} {price_h4}")
        logger.info(f"{symbol} H1 {d1date} {format_text(d1_trend)} {price_d1}")
        # logger.info(
        #     f"{symbol_name} {m5date} m5_trend: {format_text(m5_trend)} :{price_m15} {m15date} m15_trend:
        #     {format_text(m15_trend)} {m30date} m30_trend: {format_text(m30_trend)} {h1date} h1_trend:
        #     {format_text(h1_trend)} close_p: {current_price}")
        order_size = mt5_client.check_order_exist(symbol)
        # do not place an order if the symbol order is placed to Metatrader
        if current_trend == "Buy" and order_size != current_trend:
            mt5_client.close_order(symbol)  # close all open positions
            # tp = close_p + (factor * atr)  # ROE=2
            mt5_client.buy_order(symbol, lot=lot, sl=None, tp=None)  # default tp at 1000 pips
        elif current_trend == 'Sell' and order_size != current_trend:
            mt5_client.close_order(symbol)  # close all open positions
            # tp = close_p - (factor * atr)  # ROE=2
            mt5_client.sell_order(symbol, lot=lot, sl=None, tp=None)  # default tp at 1000 pips
        elif order_size == 'Sell' and (current_trend == 'Buy' or current_trend == "Neutral"
                                       or h1_trend == 'Buy' or h4_trend == "Buy" or d1_trend == 'Buy'):
            mt5_client.close_order(symbol)  # close all Sell positions
        elif order_size == 'Buy' and (current_trend == 'Sell' or current_trend == "Neutral" or
                                      h1_trend == 'Sell' or h4_trend == "Sell" or d1_trend == 'Sell'):
            mt5_client.close_order(symbol)  # close all Buy positions
        logger.info("=" * 50)


if __name__ == '__main__':
    # Run job every hour at the 42rd minute
    # scheduler_job()
    schedule.every().minutes.at(":00").do(scheduler_job)
    while True:
        schedule.run_pending()
        time.sleep(1)
