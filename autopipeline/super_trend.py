from __future__ import print_function
import json
import time
from datetime import datetime

import schedule
from mt5 import AutoOrder
from utils import get_trend, logger
mt5_client = AutoOrder()


def scheduler_job():
    """Shows basic usage of the Gmail API.
    Lists the user's Gmail labels.
    """
    logger.info(f"Start job at: {datetime.now()}")
    with open("config.json") as config_file:
        config = json.load(config_file)

    for symbol_name, value in zip(config.keys(), config.values()):
        # symbol_name = "NAS100"
        df, ema_200 = mt5_client.get_frames(symbol_name)
        close_p = float(df.close.tail(1))
        high_p = float(df.high.tail(1))
        low_p = float(df.low.tail(1))
        atr = float(df.ATR.tail(1))
        sp_trend_1 = float(df.SuperTrend312.tail(1))
        sp_trend_2 = float(df.SuperTrend110.tail(1))
        sp_trend_3 = float(df.SuperTrend211.tail(1))
        current_trend = get_trend(close_p, sp_trend_1, sp_trend_2, sp_trend_3)
        order_placed = mt5_client.check_order_exist(symbol_name, current_trend)
        if current_trend == "Buy" and not order_placed:
            # TODO Query all sell order and close
            # TODO put a new order, stop loss is close_p - atr
            mt5_client.close_order(symbol_name)  # close all open positions
            if close_p > ema_200:
                sl = close_p - atr
                mt5_client.buy_order(symbol_name, lot=0.1, sl=sl)  # default tp at 1000 pips
        elif current_trend == 'Sell' and not order_placed:
            # TODO Query all sell order and close
            # TODO put a new order, stop loss is close_p + atr
            mt5_client.close_order(symbol_name)  # close all open positions
            if close_p < ema_200:
                sl = close_p + atr
                mt5_client.sell_order(symbol_name, lot=0.1, sl=sl)  # default tp at 1000 pips
        else:
            mt5_client.modify_stoploss()


if __name__ == '__main__':
    # Run job every hour at the 42rd minute
    schedule.every().hour.at(":00").do(scheduler_job)
    while True:
        schedule.run_pending()
        time.sleep(1)
