from alpha_vantage.timeseries import TimeSeries

# from config.settings import settings
import pandas as pd


ts = TimeSeries(key="UWEB4RNFJ934F5EN", output_format="pandas")


def get_stock_data_alpha_vantage(symbol, interval, months):
    outputsize = "full"

    if interval == "daily":
        data, meta_data = ts.get_daily(symbol=symbol, outputsize=outputsize)
    elif interval == "weekly":
        data, meta_data = ts.get_weekly(symbol=symbol, outputsize=outputsize)
    elif interval == "monthly":
        data, meta_data = ts.get_monthly(symbol=symbol, outputsize=outputsize)
    elif interval.endswith("min"):
        data, meta_data = ts.get_intraday(
            symbol=symbol, interval=interval, outputsize=outputsize
        )
    else:
        raise ValueError("Unsupported interval. Choose '1min', '60min', or 'daily'.")

    # 篩選指定時間範圍內的數據
    end_date = pd.Timestamp.now()
    start_date = end_date - pd.DateOffset(months=months)
    data_filtered = data[(data.index >= start_date) & (data.index <= end_date)]

    data_filtered.columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    return data_filtered


# 使用範例
if __name__ == "__main__":
    stock_data = get_stock_data_alpha_vantage("MSFT", "daily", 6)
    print(stock_data.head())
