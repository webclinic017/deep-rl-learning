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

    timezone = pytz.timezone("Etc/UTC")
    logger.info("=" * 50)
    logger.info(f"Start job at: {datetime.now(tz=timezone)}")
    with open("config.json") as config_file:
        config = json.load(config_file)

    current_time = datetime.now(tz=timezone) + timedelta(hours=2)
    for symbol, value in zip(config.keys(), config.values()):
        # symbol_name = "BTCUSD"
        lot = value.get('lot')
        h1date, h1trend, h1price = mt5_client.ichimoku_cloud(timeframe=mt5.TIMEFRAME_H1, symbol=symbol)
        h4date, h4trend, h4price = mt5_client.ichimoku_cloud(timeframe=mt5.TIMEFRAME_H4, symbol=symbol)
        d1date, d1trend, d1price = mt5_client.ichimoku_cloud(timeframe=mt5.TIMEFRAME_D1, symbol=symbol)

        current_trend = '0'
        if h1trend == h4trend == d1trend == "Sell":
            current_trend = "Sell"
        elif h1trend == h4trend == d1trend == "Buy":
            current_trend = "Buy"
        elif h1trend == h4trend == '0':
            current_trend = "Neutral"

        logger.info("=" * 50)
        logger.info(current_time)
        logger.info(f"{symbol} H1 {h1date} {format_text(h1trend)} {h1price}")
        logger.info(f"{symbol} H4 {h4date} {format_text(h4trend)} {h4price}")
        logger.info(f"{symbol} D1 {d1date} {format_text(d1trend)} {d1price}")

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
        elif order_size == 'Sell' and (current_trend == "Neutral" or current_trend == 'Buy' or
                                       h1trend == 'Buy' or h4trend == 'Buy' or d1trend == 'Buy'):
            mt5_client.close_order(symbol)  # close all Sell positions
        elif order_size == 'Buy' and (current_trend == "Neutral" or current_trend == 'Sell'
                                      or h1trend == 'Sell' or h4trend == 'Sell' or d1trend == 'Sell'):
            mt5_client.close_order(symbol)  # close all Buy positions
        logger.info("=" * 50)


if __name__ == '__main__':
    # Run job every hour at the 42rd minute
    # scheduler_job()
    schedule.every().days.do(scheduler_job)
    while True:
        schedule.run_pending()
        time.sleep(1)
