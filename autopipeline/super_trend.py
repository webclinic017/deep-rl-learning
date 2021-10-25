from __future__ import print_function
import json
import time
from datetime import datetime

import schedule
from mt5 import AutoOrder
from utils import logger
mt5_client = AutoOrder()


def scheduler_job():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """

    logger.info(f"Start job at: {datetime.now()}")
    logger.info("="*50)
    with open("config.json") as config_file:
        config = json.load(config_file)

    for symbol_name, value in zip(config.keys(), config.values()):
        # symbol_name = "Bitcoin"
        lot = value.get('lot')
        factor = 2
        # try:
        close_p, current_trend, middle_band, upper_band, lower_band, atr = mt5_client.get_frames(symbol_name)
        # except Exception as ex:
        #     logger.error(f"Get frames errors: {ex}")
        #     continue

        logger.info(f"close_p: {close_p} current_trend: {current_trend} middle band {middle_band}")
        order_exist = mt5_client.check_order_exist(symbol_name, current_trend)
        # do not place an order if the symbol order is placed to Metatrader
        if current_trend == "Buy" and not order_exist:
            mt5_client.close_order(symbol_name)  # close all open positions
            # tp = close_p + (factor * atr)  # ROE=2
            mt5_client.buy_order(symbol_name, lot=lot-0.01, sl=None, tp=close_p+atr)  # default tp at 1000 pips
            mt5_client.buy_order(symbol_name, lot=0.01, sl=None, tp=None)  # default tp at 1000 pips
        elif current_trend == 'Sell' and not order_exist:
            mt5_client.close_order(symbol_name)  # close all open positions
            # tp = close_p - (factor * atr)  # ROE=2
            mt5_client.sell_order(symbol_name, lot=lot-0.01, sl=None, tp=close_p-atr)  # default tp at 1000 pips
            mt5_client.sell_order(symbol_name, lot=0.01, sl=None, tp=None)  # default tp at 1000 pips
        # elif current_trend == "0":
        #     mt5_client.close_order(symbol_name)  # close all open positions of the symbol_name
        # if order_exist:
            # sl = lower_band if current_trend == "Buy" else upper_band
        # mt5_client.modify_stoploss(symbol_name, middle_band, atr)


if __name__ == '__main__':
    # Run job every hour at the 42rd minute
    scheduler_job()
    schedule.every().hours.at(":59").do(scheduler_job)
    schedule.every().hours.at(":29").do(scheduler_job)
    while True:
        schedule.run_pending()
        time.sleep(1)
