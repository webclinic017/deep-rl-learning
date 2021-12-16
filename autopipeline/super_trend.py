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

    current_time = datetime.now(tz=timezone) + timedelta(hours=2)
    for symbol, value in zip(config.keys(), config.values()):
        # symbol_name = "BTCUSD"
        lot = value.get('lot')
        logger.info("=" * 50)

        h1date, h1trend, h1price, _ = mt5_client.ichimoku_cloud(timeframe=mt5.TIMEFRAME_M15, symbol=symbol)
        h4date, h4trend, h4price, stop_loss = mt5_client.ichimoku_cloud(timeframe=mt5.TIMEFRAME_M30, symbol=symbol)

        current_trend = '0'
        if h1trend == h4trend == "Sell":
            current_trend = "Sell"
        elif h1trend == h4trend == "Buy":
            current_trend = "Buy"
        elif h1trend and h4trend == '0':
            current_trend = "Neutral"

        logger.info(f"{symbol} Current Trend {format_text(current_trend)}")
        logger.info(f"{symbol} M15 {h1date} {format_text(h1trend)} {h1price}")
        logger.info(f"{symbol} M30 {h4date} {format_text(h4trend)} {h4price}")

        order_size = mt5_client.check_order_exist(symbol)
        # do not place an order if the symbol order is placed to Metatrader
        if current_trend == "Buy" and order_size != current_trend:
            mt5_client.close_order(symbol)  # close all open positions
            # tp = close_p + (factor * atr)  # ROE=2
            mt5_client.buy_order(symbol, lot=lot, sl=stop_loss, tp=None)  # default tp at 1000 pips
        elif current_trend == 'Sell' and order_size != current_trend:
            mt5_client.close_order(symbol)  # close all open positions
            # tp = close_p - (factor * atr)  # ROE=2
            mt5_client.sell_order(symbol, lot=lot, sl=stop_loss, tp=None)  # default tp at 1000 pips
        elif order_size == 'Sell' and (current_trend == "Neutral" or current_trend == 'Buy' or
                                       h1trend == 'Buy' or h4trend == 'Buy' or h1trend == 'Close_Sell' or
                                       h4trend == 'Close_Sell'):
            mt5_client.close_order(symbol)  # close all Sell positions
        elif order_size == 'Buy' and (current_trend == "Neutral" or current_trend == 'Sell' or
                                      h1trend == 'Sell' or h4trend == 'Sell' or h1trend == 'Close_Buy' or
                                      h4trend == 'Close_Buy'):
            mt5_client.close_order(symbol)  # close all Buy positions

        if order_size:
            mt5_client.modify_stoploss(symbol, stop_loss)

        logger.info("=" * 50)


if __name__ == '__main__':
    # Run job every hour at the 42rd minute
    # scheduler_job()
    schedule.every().hours.at(":59").do(scheduler_job)
    schedule.every().hours.at(":14").do(scheduler_job)
    schedule.every().hours.at(":29").do(scheduler_job)
    schedule.every().hours.at(":44").do(scheduler_job)
    while True:
        schedule.run_pending()
        time.sleep(1)
