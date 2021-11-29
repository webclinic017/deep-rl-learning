from __future__ import print_function
import json
import time
import MetaTrader5 as mt5
import schedule
import sentry_sdk
from datetime import datetime
from mt5 import AutoOrder
from utils import logger
from termcolor import colored

mt5_client = AutoOrder()

sentry_sdk.init(
    "https://cc11af54279542189f34a16070babe07@o1068161.ingest.sentry.io/6062320",

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)


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

    for symbol_name, value in zip(config.keys(), config.values()):
        # symbol_name = "BTCUSD"
        lot = value.get('lot')
        current_time = datetime.now().timestamp()
        current_price, m5_trend, dfdate = mt5_client.get_frames(time_from=current_time - 86400 * 4,
                                                                   time_to=current_time,
                                                                   timeframe=mt5.TIMEFRAME_M5, symbol=symbol_name)
        price_m15, m15_trend, m15date = mt5_client.get_frames(time_from=current_time - 86400 * 4,
                                                                 time_to=current_time,
                                                                 timeframe=mt5.TIMEFRAME_M15, symbol=symbol_name)
        price_m30, m30_trend, m30date = mt5_client.get_frames(time_from=current_time - 86400 * 7,
                                                                 time_to=current_time,
                                                                 timeframe=mt5.TIMEFRAME_M30, symbol=symbol_name)
        price_h1, h1_trend, h1date = mt5_client.get_frames(time_from=current_time - 86400 * 14, time_to=current_time,
                                                              timeframe=mt5.TIMEFRAME_H1, symbol=symbol_name)
        price_h4, h4_trend, h4date = mt5_client.get_frames(time_from=current_time - 86400 * 35,
                                                              time_to=current_time,
                                                              timeframe=mt5.TIMEFRAME_H4, symbol=symbol_name)
        price_d1, d1_trend, d1date = mt5_client.get_frames(time_from=current_time - 86400 * 100,
                                                              time_to=current_time,
                                                              timeframe=mt5.TIMEFRAME_D1, symbol=symbol_name)

        current_trend = '0'
        if m5_trend == m15_trend == m30_trend == h1_trend == h4_trend == d1_trend == "Sell":
            current_trend = "Sell"
        elif m5_trend == m15_trend == m30_trend == h1_trend == h4_trend == d1_trend == "Buy":
            current_trend = "Buy"
        elif m5_trend == m15_trend == m30_trend == "0":
            current_trend = "Neutral"

        # logger.info(f"M5 {dfdate}, {current_price}")
        # logger.info(f"M15 {m15_date}, {price_m15}")
        # logger.info(f"M30 {m30_date}, {price_m30}")
        # logger.info(f"H1 {h1_date}, {price_h1}")
        logger.info(
            f"{symbol_name} {dfdate} m5_trend: {format_text(m5_trend)} {m15date} m15_trend: {format_text(m15_trend)} {m30date} m30_trend: {format_text(m30_trend)} {h1date} h1_trend: {format_text(h1_trend)} {h4date} h4_trend: {format_text(h4_trend)} {d1date} d1_trend: {format_text(d1_trend)} close_p: {current_price}")
        order_size = mt5_client.check_order_exist(symbol_name)
        # do not place an order if the symbol order is placed to Metatrader
        if current_trend == "Buy" and order_size != current_trend:
            mt5_client.close_order(symbol_name)  # close all open positions
            # tp = close_p + (factor * atr)  # ROE=2
            mt5_client.buy_order(symbol_name, lot=lot, sl=None, tp=None)  # default tp at 1000 pips
        elif current_trend == 'Sell' and order_size != current_trend:
            mt5_client.close_order(symbol_name)  # close all open positions
            # tp = close_p - (factor * atr)  # ROE=2
            mt5_client.sell_order(symbol_name, lot=lot, sl=None, tp=None)  # default tp at 1000 pips
        elif order_size == 'Sell' and (current_trend == 'Buy' or m15_trend == 'Buy' or
                                       m30_trend == "Buy" or h1_trend == 'Buy' or current_trend == "Neutral"):
            mt5_client.close_order(symbol_name)  # close all Sell positions
        elif order_size == 'Buy' and (current_trend == 'Sell' or m15_trend == 'Sell' or
                                      m30_trend == "Sell" or h1_trend == 'Sell' or current_trend == "Neutral"):
            mt5_client.close_order(symbol_name)  # close all Buy positions


if __name__ == '__main__':
    # Run job every hour at the 42rd minute
    # scheduler_job()
    schedule.every().minutes.do(scheduler_job)
    while True:
        schedule.run_pending()
        time.sleep(1)
