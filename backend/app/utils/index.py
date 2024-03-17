import pandas as pd
import numpy as np


def n_macd(df):
    sh = df["close"].ewm(span=13, adjust=False).mean()
    lon = df["close"].ewm(span=21, adjust=False).mean()
    ratio = [min(sh[i], lon[i]) / max(sh[i], lon[i]) for i in range(sh.shape[0])]
    r_len = len(ratio)
    mac = [2 - ratio[i] - 1 if sh[i] > lon[i] else ratio[i] - 1 for i in range(r_len)]
    lowest_mac = pd.Series(mac).rolling(window=50).min()
    highest_mac = pd.Series(mac).rolling(window=50).max()
    macNorm = [
        ((mac[i] - lowest_mac[i]) / (highest_mac[i] - lowest_mac[i] + 0.000001) * 2) - 1
        for i in range(r_len)
    ]
    trigger = (
        pd.Series(macNorm, index=df.index).fillna(0).rolling(window=9).mean().fillna(0)
    )
    return pd.DataFrame({"macNorm": macNorm, "trigger": trigger}, index=df.index)


def sr_bias(df):
    ma10 = df["close"].rolling(window=10).mean()
    sr = (df["close"] - ma10) / ma10
    return pd.Series(sr, index=df.index)


def lr_bias(df):
    ma30 = df["close"].rolling(window=30).mean()
    lr = (df["close"] - ma30) / ma30
    return pd.Series(lr, index=df.index)


def rsi(df, period=21):
    Close = df["close"].copy()
    Chg = Close - Close.shift(1)
    Chg_pos = pd.Series(index=Chg.index, data=Chg[Chg > 0]).fillna(0)
    Chg_neg = pd.Series(index=Chg.index, data=-Chg[Chg < 0]).fillna(0)

    up_mean = [
        np.mean(Chg_pos.values[i - period : i])
        for i in range(period + 1, len(Chg_pos) + 1)
    ]
    down_mean = [
        np.mean(Chg_neg.values[i - period : i])
        for i in range(period + 1, len(Chg_neg) + 1)
    ]

    res = [100 * up_mean[i] / (up_mean[i] + down_mean[i]) for i in range(len(up_mean))]
    rsi_series = pd.Series(index=Close.index[period:], data=res)
    sma_rsi = rsi_series.rolling(window=55).mean()

    return pd.DataFrame({"rsi": rsi_series, "sma_rsi": sma_rsi}, index=df.index)


def macd(df):
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    macd_series = exp1 - exp2
    exp3 = macd_series.ewm(span=9, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_series, "macdsignal": exp3, "macdhist": macd_series - exp3},
        index=df.index,
    )


def obv(df):
    obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    return pd.Series(obv, index=df.index)


def ma(df, timeperiod, ma_type="SMA"):
    if ma_type == "SMA":
        res = df["close"].rolling(window=timeperiod).mean()
    elif ma_type == "EMA":
        res = df["close"].ewm(span=timeperiod, adjust=False).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, timeperiod + 1)
        res = (
            df["close"]
            .rolling(timeperiod)
            .apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        )
    else:
        raise ValueError("Invalid MA Type: Choose 'SMA', 'EMA', or 'WMA'")
    return pd.Series(res, index=df.index)


def atr(df):
    high_low = df["high"] - df["low"]
    high_close = np.abs(df["high"] - df["close"].shift())
    low_close = np.abs(df["low"] - df["close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).mean()
    return pd.Series(atr, index=df.index)


def bollingerBand(df, n=20):
    sma = df["close"].rolling(window=n).mean()
    bb_up = sma + 2 * df["close"].rolling(window=n).std(ddof=0)
    bb_down = sma - 2 * df["close"].rolling(window=n).std(ddof=0)
    bb_width = bb_up - bb_down
    return pd.DataFrame(
        {"BB_up": bb_up, "BB_down": bb_down, "BB_width": bb_width}, index=df.index
    )
