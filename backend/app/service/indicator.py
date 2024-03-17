import pandas as pd
from utils import index


def index_generate(df, required_ta):
    for ta_info in required_ta:
        ta_name = ta_info.name
        try:
            if ta_name == "ma":
                period = ta_info.period
                ma_type = ta_info.type
                output = index.ma(df, timeperiod=period, ma_type=ma_type)
            else:
                output = getattr(index, ta_name)(df)
            output.name = (
                f"{ta_info.period}{ta_name.lower()}"
                if type(output) == pd.core.series.Series
                else None
            )
            df = pd.merge(
                df, pd.DataFrame(output), left_on=df.index, right_on=output.index
            )
            df = df.set_index("key_0")
        except Exception as e:
            print(e)
    df.index = df.time
    df.drop("time", axis=1, inplace=True)
    return df


if __name__ == "__main__":
    raw_prices = pd.read_feather("../../../btc_1h_price.feather")
    # 示例：將技術指標列表轉變為包含字典的列表
    required_ta = [
        {"name": "ma", "period": 20, "type": "SMA"},
        {"name": "rsi"},  # 假設 RSI 不需要額外參數，或其參數在函數內有默認值
    ]
    df_w_index = index_generate(raw_prices, required_ta)
    print(df_w_index.tail())
