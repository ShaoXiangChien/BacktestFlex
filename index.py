import streamlit as st
from talib import abstract
import pandas as pd
import talib
import numpy as np


def n_macd(df):
    sh = abstract.EMA(df, 13)
    lon = abstract.EMA(df, 21)
    ratio = [min(sh[i], lon[i]) / max(sh[i], lon[i])
             for i in range(sh.shape[0])]
    r_len = len(ratio)
    mac = [2 - ratio[i] - 1 if sh[i] > lon[i]
           else ratio[i] - 1 for i in range(r_len)]
    lowest_mac = pd.Series(mac).rolling(window=50).min()
    highest_mac = pd.Series(mac).rolling(window=50).max()
    macNorm = [((mac[i] - lowest_mac[i])/(highest_mac[i] -
                lowest_mac[i]+0.000001)*2) - 1 for i in range(r_len)]
    trigger = talib.WMA(np.array(macNorm), 9)

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


def rsi(df):
    res = abstract.RSI(df, timeperiod=21)
    sma_rsi = res.rolling(window=55).mean()
    return pd.DataFrame({'rsi': res, 'sma_rsi': sma_rsi})


def macd(df):
    return abstract.MACD(df)


def obv(df):
    return abstract.OBV(df)


def ma(df, timeperiod):
    ma_type = st.selectbox(
        f"{timeperiod}ma type", ['SMA', 'EMA', 'WMA'], key=f'{timeperiod}ma')
    return eval('abstract.' + ma_type + '(df, timeperiod=timeperiod)')


def atr(df):
    return abstract.ATR(df, timeperiod=14)


def bollingerBand(DF, n=20):
    df = DF.copy()
    df['sma'] = df.close.rolling(window=n).mean()
    bb_up = df['sma'] + 2 * df.close.rolling(window=n).std(ddof=0)
    bb_down = df['sma'] - 2 * df.close.rolling(window=n).std(ddof=0)
    bb_width = bb_up - bb_down
    return pd.DataFrame({"BB_up": bb_up, "BB_down": bb_down, "BB_width": bb_width}, index=df.index)
