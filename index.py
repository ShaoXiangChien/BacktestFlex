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


def five_in_one(df):
    pass


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
