import logging
import numpy as np


# create logger with 'spam_application'
logger = logging.getLogger('app_application')
logger.setLevel(logging.INFO)
# create file handler which logs even debug messages
fh = logging.FileHandler('app.log')
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


# SuperTrend
def ST(df, f, n):  # df is the dataframe, n is the period, f is the factor; f=3, n=7 are commonly used.
    # Calculation of ATR
    col_name = f"SuperTrend{f}{n}"
    df['H-L'] = abs(df['high'] - df['low'])
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = np.nan
    df.loc[n - 1, 'ATR'] = df['TR'][:n - 1].mean()  # .ix is deprecated from pandas verion- 0.19
    for i in range(n, len(df)):
        df['ATR'][i] = (df['ATR'][i - 1] * (n - 1) + df['TR'][i]) / n

    # Calculation of SuperTrend
    df['Upper Basic'] = (df['high'] + df['low']) / 2 + (f * df['ATR'])
    df['lower Basic'] = (df['high'] + df['low']) / 2 - (f * df['ATR'])
    df['Upper Band'] = df['Upper Basic']
    df['lower Band'] = df['lower Basic']
    for i in range(n, len(df)):
        if df['close'][i - 1] <= df['Upper Band'][i - 1]:
            df['Upper Band'][i] = min(df['Upper Basic'][i], df['Upper Band'][i - 1])
        else:
            df['Upper Band'][i] = df['Upper Basic'][i]
    for i in range(n, len(df)):
        if df['close'][i - 1] >= df['lower Band'][i - 1]:
            df['lower Band'][i] = max(df['lower Basic'][i], df['lower Band'][i - 1])
        else:
            df['lower Band'][i] = df['lower Basic'][i]
    df[col_name] = np.nan
    for i in df[col_name]:
        if df['close'][n - 1] <= df['Upper Band'][n - 1]:
            df[col_name][n - 1] = df['Upper Band'][n - 1]
        elif df['close'][n - 1] > df['Upper Band'][i]:
            df[col_name][n - 1] = df['lower Band'][n - 1]
    for i in range(n, len(df)):
        if df[col_name][i - 1] == df['Upper Band'][i - 1] and df['close'][i] <= df['Upper Band'][i]:
            df[col_name][i] = df['Upper Band'][i]
        elif df[col_name][i - 1] == df['Upper Band'][i - 1] and df['close'][i] >= df['Upper Band'][i]:
            df[col_name][i] = df['lower Band'][i]
        elif df[col_name][i - 1] == df['lower Band'][i - 1] and df['close'][i] >= df['lower Band'][i]:
            df[col_name][i] = df['lower Band'][i]
        elif df[col_name][i - 1] == df['lower Band'][i - 1] and df['close'][i] <= df['lower Band'][i]:
            df[col_name][i] = df['Upper Band'][i]
    return df
