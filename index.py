import streamlit as st
# from talib import abstract
import pandas as pd
# import talib
import numpy as np


def n_macd(df):
    sh = df['close'].ewm(span=13, adjust=False).mean()
    lon = df['close'].ewm(span=21, adjust=False).mean()
    ratio = [min(sh[i], lon[i]) / max(sh[i], lon[i])
             for i in range(sh.shape[0])]
    r_len = len(ratio)
    mac = [2 - ratio[i] - 1 if sh[i] > lon[i]
           else ratio[i] - 1 for i in range(r_len)]
    lowest_mac = pd.Series(mac).rolling(window=50).min()
    highest_mac = pd.Series(mac).rolling(window=50).max()
    macNorm = [((mac[i] - lowest_mac[i])/(highest_mac[i] -
                lowest_mac[i]+0.000001)*2) - 1 for i in range(r_len)]
    trigger = pd.Series(macNorm).rolling(window=9).mean()

    return pd.DataFrame({'macNorm': macNorm, 'trigger': trigger}, index=df.index)


def sr_bias(df):
    ma10 = df['close'].rolling(window=10).mean()
    df['ma10'] = ma10
    sr = df.apply(lambda row: (row['close'] - row['ma10'])/row['ma10'], axis=1)
    return pd.Series(sr, index=df.index)


def lr_bias(df):
    ma30 = df['close'].rolling(window=30).mean()
    df['ma30'] = ma30
    lr = df.apply(lambda row: (row['close'] - row['ma30'])/row['ma30'], axis=1)
    return pd.Series(lr, index=df.index)


def rsi(df, period=21):
    # 整理資料
    Close = df['close'].copy()
    Chg = Close - Close.shift(1)
    Chg_pos = pd.Series(index=Chg.index, data=Chg[Chg > 0])
    Chg_pos = Chg_pos.fillna(0)
    Chg_neg = pd.Series(index=Chg.index, data=-Chg[Chg < 0])
    Chg_neg = Chg_neg.fillna(0)

    # 計算12日平均漲跌幅度
    import numpy as np
    up_mean = []
    down_mean = []
    for i in range(period+1, len(Chg_pos)+1):
        up_mean.append(np.mean(Chg_pos.values[i-period:i]))
        down_mean.append(np.mean(Chg_neg.values[i-period:i]))

    # 計算 RSI
    res = []
    for i in range(len(up_mean)):
        res.append(100 * up_mean[i] / (up_mean[i] + down_mean[i]))
    rsi_series = pd.Series(index=Close.index[period:], data=res)
    sma_rsi = rsi_series.rolling(window=55).mean()

    return pd.DataFrame({'rsi': rsi_series, 'sma_rsi': sma_rsi}, index=df.index)


# def rsi(df):
#     res = abstract.RSI(df, timeperiod=21)
#     sma_rsi = res.rolling(window=55).mean()
#     return pd.DataFrame({'rsi': res, 'sma_rsi': sma_rsi})


def macd(df):
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd_series = exp1 - exp2
    exp3 = macd_series.ewm(span=9, adjust=False).mean()
    return pd.DataFrame({'macd': macd_series, 'macdsignal': exp3, 'macdhist': macd_series - exp3}, index=df.index)


def obv(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return pd.Series(obv, index=df.index)


def ma(df, timeperiod):
    ma_type = st.selectbox(
        f"{timeperiod}ma type", ['SMA', 'EMA', 'WMA'], key=f'{timeperiod}ma')
    if ma_type == 'SMA':
        res = df['close'].rolling(window=timeperiod).mean()
    elif ma_type == 'EMA':
        res = df['close'].ewm(span=timeperiod, adjust=False).mean()
    else:
        weights = np.arange(1, timeperiod + 1)
        res = df['close'].rolling(timeperiod).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True).to_list()
    return pd.Series(res, index=df.index)


def atr(df):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).sum()/14
    return pd.Series(atr, index=df.index)


def bollingerBand(DF, n=20):
    df = DF.copy()
    df['sma'] = df.close.rolling(window=n).mean()
    bb_up = df['sma'] + 2 * df.close.rolling(window=n).std(ddof=0)
    bb_down = df['sma'] - 2 * df.close.rolling(window=n).std(ddof=0)
    bb_width = bb_up - bb_down
    return pd.DataFrame({"BB_up": bb_up, "BB_down": bb_down, "BB_width": bb_width}, index=df.index)
