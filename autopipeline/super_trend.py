from __future__ import print_function
import json
import time
import MetaTrader5 as mt5
import schedule
import sentry_sdk
from datetime import datetime
from mt5 import AutoOrder
from utils import logger
mt5_client = AutoOrder()


sentry_sdk.init(
    "https://cc11af54279542189f34a16070babe07@o1068161.ingest.sentry.io/6062320",

    # Set traces_sample_rate to 1.0 to capture 100%
    # of transactions for performance monitoring.
    # We recommend adjusting this value in production.
    traces_sample_rate=1.0
)


def scheduler_job():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    if datetime.now().minute % 5 != 0:
        return

    logger.info(f"Start job at: {datetime.now()}")
    logger.info("="*50)
    with open("config.json") as config_file:
        config = json.load(config_file)

    for symbol_name, value in zip(config.keys(), config.values()):
        # symbol_name = "BTCUSD"
        lot = value.get('lot')

        current_price, m5_trend, dfdate = mt5_client.get_frames(timeframe=mt5.TIMEFRAME_M5, symbol=symbol_name)
        _, m15_trend, _ = mt5_client.get_frames(timeframe=mt5.TIMEFRAME_M15, symbol=symbol_name)
        _, m30_trend, _ = mt5_client.get_frames(timeframe=mt5.TIMEFRAME_M30, symbol=symbol_name)
        _, h1_trend, _ = mt5_client.get_frames(timeframe=mt5.TIMEFRAME_H1, symbol=symbol_name)

        current_trend = '0'
        if m5_trend == m15_trend == m30_trend == h1_trend == "Sell":
            current_trend = "Sell"
        if m5_trend == m15_trend == m30_trend == h1_trend == "Buy":
            current_trend = "Buy"

        logger.info(f"{dfdate} close_p: {current_price}  m5_trend: {m5_trend} m30_trend: {m30_trend}")
        order_exist = mt5_client.check_order_exist(symbol_name, current_trend)
        # do not place an order if the symbol order is placed to Metatrader
        if current_trend == "Buy" and not order_exist:
            mt5_client.close_order(symbol_name)  # close all open positions
            # tp = close_p + (factor * atr)  # ROE=2
            mt5_client.buy_order(symbol_name, lot=lot, sl=None, tp=None)  # default tp at 1000 pips
        elif current_trend == 'Sell' and not order_exist:
            mt5_client.close_order(symbol_name)  # close all open positions
            # tp = close_p - (factor * atr)  # ROE=2
            mt5_client.sell_order(symbol_name, lot=lot, sl=None, tp=None)  # default tp at 1000 pips
        elif current_trend == "0":
            mt5_client.close_order(symbol_name)  # close all open positions


if __name__ == '__main__':
    # Run job every hour at the 42rd minute
    # scheduler_job()
    schedule.every().minutes.do(scheduler_job)
    while True:
        schedule.run_pending()
        time.sleep(1)
