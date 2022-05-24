import index
import strategy
import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from binance.client import Client
import json
from stqdm import stqdm
# connect to binance api
with open('test_api.json', 'r') as fh:
    key_dt = json.load(fh)

client = Client(key_dt['api_key'], key_dt['secret'])
client.API_URL = 'https://testnet.binance.vision/api'

# prepare ta and signal list
ta_list = ['ma', 'macd', 'rsi', 'kd', 'obv', 'n_macd', 'five_in_one']

signal_list = [
    'golden_cross',
    'death_cross',
    'ma_up_penetrate',
    'ma_down_penetrate',
    'constant_compare'
]

# btc_df = pd.DataFrame()
idx = 0
required_ta = []
ma_colors = ['yellow', 'orange', 'red']


def add_new_action(signal_choice, id, action):
    cond_dt = {}
    if signal_choice == 'golden_cross':
        cond_dt['signal'] = signal_choice
        cond_dt['index'] = st.selectbox(
            'Index', ta_list, key=f'index {id}'+action)
        if cond_dt['index'] == 'ma':
            cond_dt['ma periods'] = [st.number_input(
                'first ma period', 1, 201, key=f'fma{id}' + action), st.number_input('second ma period', 1, 201, key=f'sma{id}' + action)]
            ma1 = f"{cond_dt['ma periods'][0]}ma"
            ma2 = f"{cond_dt['ma periods'][1]}ma"
            if ma1 not in required_ta:
                required_ta.append(ma1)
            if ma2 not in required_ta:
                required_ta.append(ma2)
        elif cond_dt['index'] not in required_ta:
            required_ta.append(cond_dt['index'])
    elif signal_choice == 'death_cross':
        cond_dt['signal'] = signal_choice
        cond_dt['index'] = st.selectbox(
            'Index', ta_list, key=f'index {id}'+action)
        if cond_dt['index'] == 'ma':
            cond_dt['ma periods'] = [int(st.number_input(
                'first ma period', 1, 201, key='fma2')), int(st.number_input('second ma period', 1, 201, key='sma2'))]
            ma1 = f"{cond_dt['ma periods'][0]}ma"
            ma2 = f"{cond_dt['ma periods'][1]}ma"
            if ma1 not in required_ta:
                required_ta.append(ma1)
            if ma2 not in required_ta:
                required_ta.append(ma2)
        elif cond_dt['index'] not in required_ta:
            required_ta.append(cond_dt['index'])
    elif signal_choice == 'ma_up_penetrate':
        cond_dt['signal'] = signal_choice
        cond_dt['penetrator'] = st.selectbox(
            'Up Penetrator', ['low price', 'high price', 'open price', 'close price'])
        cond_dt['ma period'] = st.number_input('up ma period', 1, 201)
        if f"{cond_dt['ma period']}ma" not in required_ta:
            required_ta.append(f"{cond_dt['ma period']}ma")
    elif signal_choice == 'ma_down_penetrate':
        cond_dt['signal'] = signal_choice
        cond_dt['penetrator'] = st.selectbox(
            'Down Penetrator', ['low price', 'high price', 'open price', 'close price'])
        cond_dt['ma period'] = st.number_input('down ma period', 1, 201)
        if f"{cond_dt['ma period']}ma" not in required_ta:
            required_ta.append(f"{cond_dt['ma period']}ma")
    elif signal_choice == 'constant_compare':
        cond_dt['signal'] = signal_choice
        cond_dt['index'] = st.selectbox(
            'Index', ta_list+['price'], key=f'index {id}'+action)
        if cond_dt['index'] not in required_ta:
            required_ta.append(cond_dt['index'])
        cond_dt['operator'] = st.selectbox(
            'Operator', ['>', '==', '<'], key=f'operator {id}' + action)
        cond_dt['constant'] = float(st.text_input(
            'Constant', key=f'constant {id}' + action))
    return cond_dt


def cond_transform(cond):
    if cond['signal'] == 'golden_cross':
        if cond['index'] == 'macd':
            col1 = 'macd'
            col2 = 'macdsignal'
        elif cond['index'] == 'ma':
            col1 = f"{cond['ma periods'][0]}ma"
            col2 = f"{cond['ma periods'][1]}ma"
        elif cond['index'] == 'rsi':
            col1 = 'rsi'
            col2 = 'sma_rsi'
        elif cond['index'] == 'n_macd':
            col1 = 'macNorm'
            col2 = 'trigger'

        op_str = "strategy.{}(btc_df['{}'].iloc[idx], btc_df.loc[idx-1, '{}'], btc_df['{}'].iloc[idx], btc_df['{}'].iloc[idx-1])".format(
            cond['signal'], col1, col1, col2, col2)

    elif cond['signal'] == 'death_cross':
        if cond['index'] == 'macd':
            col1 = 'macd'
            col2 = 'macdsignal'
        elif cond['index'] == 'ma':
            col1 = f"{cond['ma periods'][0]}ma"
            col2 = f"{cond['ma periods'][1]}ma"
        elif cond['index'] == 'rsi':
            col1 = 'rsi'
            col2 = 'sma_rsi'
        elif cond['index'] == 'n_macd':
            col1 = 'macNorm'
            col2 = 'trigger'

        op_str = "strategy.{}(btc_df['{}'].iloc[idx], btc_df['{}'].iloc[idx-1], btc_df['{}'].iloc[idx], btc_df['{}'].iloc[idx-1])".format(
            cond['signal'], col1, col1, col2, col2)

    elif cond['signal'] == 'ma_up_penetrate':
        # col name = nma vs candle. penetrator[:-6]
        col = f"{cond['ma period']}ma"
        op_str = "strategy.{}(btc_df['{}'].iloc[idx], btc_df['{}'].iloc[idx], btc_df['{}'].iloc[idx-1])".format(
            cond['signal'], cond['penetrator'][:-6], col, col)

    elif cond['signal'] == 'ma_down_penetrate':
        # col name = nma vs candle. penetrator[:-6]
        col = f"{cond['ma period']}ma"
        op_str = "strategy.{}(btc_df['{}'].iloc[idx], btc_df['{}'].iloc[idx], btc_df['{}'].iloc[idx-1])".format(
            cond['signal'], cond['penetrator'][:-6], col, col)

    elif cond['signal'] == 'constant_compare':
        op_str = "strategy.{}(btc_df['{}'].iloc[idx], '{}', {})".format(
            cond['signal'], cond['index'], cond['operator'], cond['constant'])
    return op_str


def main():
    st.title("Bitcoin Trading Strategy Backtest")

    # price data preparation
    st.header("Data Preparation")
    data_mode = st.radio(
        "Use our existing data or collect data in real-time", ["Use available ones", 'Fetch now'])

    if data_mode == 'Use available ones':
        timeframe = st.selectbox('timeframe', ['15m', '1h'])
        btc_df = pd.read_feather('./btc_' + timeframe + '_price.feather')
    else:
        timeframe = st.selectbox('Select your trading timeframe: ', [
            '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'], 3)
        timestamp = client._get_earliest_valid_timestamp(
            'BTCUSDT', timeframe, limit=5000)
        st.write(pd.to_datetime(timestamp, unit='ms'))
        bars = client.get_historical_klines('BTCUSDT', timeframe, timestamp)
        for line in bars:
            del line[6:]

        btc_df = pd.DataFrame(
            bars, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    btc_df.set_index('time', inplace=True)
    btc_df.index = pd.to_datetime(btc_df.index, unit='ms')
    st.success("Bitcoin price information loaded!")
    st.dataframe(btc_df)

    # strategy development
    st.header("Strategy Development")
    st.markdown(
        "> We provide some common index signal options for you to set as your transaction trigger. You need to choose on what signal of an index you want to buy long or sell short.")
    st.markdown("""
    - Crossing behaviors involves two lines of an index, such as fast line and slow line in macd, 2 moving average lines, etc.
- Penetration is for a certain value e.g. price, atr, etc, to penetrate a line.
- Constant Comparison is to set a constant for a certrain value as a trigger of action.
  For example, sell short when price lower than 1000.
    """)
    st.subheader("When to buy long")
    num_of_condition = int(st.number_input(
        'How many conditions you want to consider to buy long?', 1, 5, value=3))
    conditions = []
    for i in range(num_of_condition):
        signal_choice = st.selectbox(
            f'Select buy long signal {i+1}', signal_list)
        cond_dt = add_new_action(signal_choice, i+1, 'buy_long')
        conditions.append(cond_dt)
    buy_long_conditions = [cond_transform(cond) for cond in conditions]

    st.subheader("When to sell short")
    num_of_condition = int(st.number_input(
        'How many conditions you want to consider to sell short?', 1, 5, value=3))
    conditions = []
    for i in range(num_of_condition):
        signal_choice = st.selectbox(
            f'Select sell short signal {i+1}', signal_list)
        cond_dt = add_new_action(signal_choice, i+1, 'sell_short')
        conditions.append(cond_dt)
    sell_short_conditions = [cond_transform(cond) for cond in conditions]

    st.subheader("When to end position")
    st.markdown("""
    > ATR can be a good indicator of ending position. It stands for average true range, which reflects the recent fluctuation of prices.
    """)
    risk_to_reward = st.number_input(
        'Risk to Reward Ratio', 0.0, 10.0, 1.0, step=0.1)

    # Index generation
    st.header("Index Generation")
    st.write(required_ta)
    for ta in required_ta + ['atr', '5ma']:
        try:
            if 'ma' == ta[-2:]:
                output = eval('index.ma(btc_df, ' + ta[:-2] + ')')
            else:
                output = eval('index.' + ta + '(btc_df)')
            output.name = ta.lower() if type(output) == pd.core.series.Series else None
            # 透過 merge 把輸出結果併入 df DataFrame
            btc_df = pd.merge(btc_df, pd.DataFrame(output),
                              left_on=btc_df.index, right_on=output.index)
            btc_df = btc_df.set_index('key_0')
        except Exception as e:
            print(e)

    btc_df.reset_index(inplace=True)
    btc_df.rename(columns={'key_0': 'time'}, inplace=True)
    st.dataframe(btc_df)

    # parameters setting
    st.header("Parameter Setting")
    initial_investment = st.number_input(
        "Initial Investment", 1, 10000000, 1000)
    balance = initial_investment
    leverage = st.number_input("Leverage Level", 1, 150, 5)
    percent_per_action = st.number_input(
        "The portion of the account balance per action", 0, 100) / 100

    # simulation
    st.header("Simulation")
    start_simulation = st.checkbox("Start Simulation")
    position = {'price': 0, 'amount': 0}
    long_just_out = False
    short_just_out = False
    stop_loss, stop_profit = 0, 0
    transaction_count = 0
    wins = 0
    loses = 0
    record = pd.DataFrame()
    result = pd.DataFrame()
    if start_simulation:
        for id, row in stqdm(list(btc_df.iloc[:20000].iterrows())):
            # condition checks
            idx = id
            if id == 0:
                continue
            if balance <= 0 or balance >= 1e6:
                break
            buy_long_check = 0
            for op in buy_long_conditions:
                buy_long_check += 1 if eval(op) else 0

            sell_short_check = 0
            for op in sell_short_conditions:
                sell_short_check += 1 if eval(op) else 0

            if row.close < row['13ma'] and long_just_out:
                long_just_out = False

            if row.close > row['13ma'] and short_just_out:
                short_just_out = False

            # 1. buy long
            if buy_long_check == len(buy_long_conditions) and position['amount'] == 0 and not long_just_out:
                u_amount = balance * percent_per_action
                position['price'] = row.close
                position['amount'] = u_amount / row.close
                stop_loss = row.close - row['atr']
                stop_profit = row.close + row['atr'] * risk_to_reward
                record = record.append(
                    {'time': row.time, 'price': row.close, 'cost': u_amount, 'profit': 0, 'action': 'buy long', 'range': f'{stop_loss} - {stop_profit}'}, ignore_index=True)
                transaction_count += 1

            # 2. sell short
            if sell_short_check == len(sell_short_conditions) and position['amount'] == 0 and not short_just_out:
                u_amount = balance * percent_per_action
                position['price'] = row.close
                position['amount'] = (-1) * (u_amount / row.close)
                stop_loss = row.close + row['atr']
                stop_profit = row.close - row['atr'] * risk_to_reward
                record = record.append(
                    {'time': row.time, 'price': row.close, 'cost': u_amount, 'profit': 0, 'action': 'sell short', 'range': f'{stop_profit} - {stop_loss}'}, ignore_index=True)
                transaction_count += 1

            # 3. end position
            if stop_loss != 0:
                if position['amount'] > 0:
                    if row.close <= stop_loss:
                        delta = random.randint(-5, 5)
                        price_diff = (stop_loss + delta) - position['price']
                        balance += price_diff * position['amount'] * leverage
                        record = record.append(
                            {'time': row.time, 'price': stop_loss + delta, 'cost': 0, 'profit': price_diff * position['amount'] * leverage, 'action': 'long stop loss'}, ignore_index=True)
                        position['amount'] = 0
                        position['price'] = 0
                        stop_loss, stop_profit = 0, 0
                        loses += 1
                        result = result.append(
                            {'time': row.time, 'balance': balance}, ignore_index=True)
                        long_just_out = True
                        continue

                    if row.close < row['5ma']:
                        delta = random.randint(-5, 5)
                        price_diff = (row['5ma'] + delta) - position['price']
                        balance += price_diff * position['amount'] * leverage
                        record = record.append(
                            {'time': row.time, 'price': row['5ma'] + delta, 'cost': 0, 'profit': price_diff * position['amount'] * leverage, 'action': 'long stop profit' if price_diff > 0 else 'long stop loss'}, ignore_index=True)
                        position['amount'] = 0
                        position['price'] = 0
                        stop_loss, stop_profit = 0, 0
                        wins += 1
                        result = result.append(
                            {'time': row.time, 'balance': balance}, ignore_index=True)
                        long_just_out = True
                        continue

                elif position['amount'] < 0:
                    if row.close >= stop_loss:
                        delta = random.randint(-5, 5)
                        price_diff = (stop_loss + delta) - position['price']
                        balance += price_diff * position['amount'] * leverage
                        record = record.append(
                            {'time': row.time, 'price': stop_loss + delta, 'cost': 0, 'profit': price_diff * position['amount'] * leverage, 'action': 'short stop loss'}, ignore_index=True)
                        position['amount'] = 0
                        position['price'] = 0
                        stop_loss, stop_profit = 0, 0
                        loses += 1
                        result = result.append(
                            {'time': row.time, 'balance': balance}, ignore_index=True)
                        short_just_out = True
                        continue

                    if row.close > row['5ma']:
                        delta = random.randint(-5, 5)
                        price_diff = (row['5ma'] + delta) - position['price']
                        balance += price_diff * position['amount'] * leverage
                        record = record.append(
                            {'time': row.time, 'price': row['5ma'] + delta, 'cost': 0, 'profit': price_diff * position['amount'] * leverage, 'action': 'short stop profit' if price_diff < 0 else 'short stop loss'}, ignore_index=True)
                        position['amount'] = 0
                        position['price'] = 0
                        stop_loss, stop_profit = 0, 0
                        wins += 1
                        result = result.append(
                            {'time': row.time, 'balance': balance}, ignore_index=True)
                        short_just_out = True
                        continue

        st.success("Simulation Complete")
        st.subheader("result")
        bar = go.Scatter(x=result['time'],
                         y=result['balance'], fill='tozeroy')
        fig = go.Figure(data=bar)
        st.plotly_chart(fig)

        st.subheader("profit statistics")
        st.write(record.profit.describe())

        st.subheader("Simulation Plot")

        plot_mode = st.radio("選擇圖表模式", ['單日', '多日'])
        if plot_mode == '單日':
            selected_date = btc_df.time.iloc[0].date()
            selected_date = st.date_input(
                '選擇日期', selected_date)
            start_date = dt.datetime.combine(
                selected_date, dt.datetime.min.time())
            end_date = dt.datetime.combine(
                selected_date, dt.datetime.max.time())
        else:
            start_date = dt.datetime.combine(st.date_input(
                '起始日期', btc_df.time.iloc[0].date()), dt.datetime.min.time())
            end_date = dt.datetime.combine(st.date_input(
                '結束日期', btc_df.time.iloc[-1].date()), dt.datetime.max.time())

        if 'macdhist' in list(btc_df.columns):
            btc_df["color"] = np.where(
                btc_df["macdhist"] < 0, 'red', 'green')

        chart_data = btc_df[(start_date < btc_df['time'])
                            & (btc_df['time'] < end_date)].copy()

        mas = []
        indexes = []
        n = len(required_ta)
        for ta in required_ta:
            if ta.startswith('ma'):
                mas.append(ta)
                n -= 1
            else:
                indexes.append(ta)

        fig2 = make_subplots(rows=n+3, cols=1, shared_xaxes=True)
        fig2.update_layout(height=800)
        fig2.add_trace(go.Candlestick(x=chart_data.index,
                                      open=chart_data['open'],
                                      high=chart_data['high'],
                                      low=chart_data['low'],
                                      close=chart_data['close'], name='market data'), row=1, col=1)
        fig2.add_trace(go.Bar(x=chart_data.index,
                              y=chart_data['volume']
                              ), row=3, col=1)
        for i, p in enumerate(mas):
            fig2.add_trace(go.Scatter(x=chart_data.index,
                                      y=chart_data[p],
                                      opacity=0.7,
                                      line=dict(
                                          color=ma_colors[i], width=2),
                                      name=p), row=1, col=1)

        for i, index in enumerate(indexes):
            if index == 'macd':
                fig2.add_trace(go.Bar(x=chart_data.index,
                                      y=chart_data.macdhist,
                                      marker_color=chart_data['color']
                                      ), row=i+4, col=1)
                fig2.add_trace(go.Scatter(x=chart_data.index,
                                          y=chart_data.macd,
                                          line=dict(color='red', width=2)
                                          ), row=i+4, col=1)
                fig2.add_trace(go.Scatter(x=chart_data.index,
                                          y=chart_data.macdsignal,
                                          line=dict(color='black', width=2)
                                          ), row=i+4, col=1)
            elif index == 'rsi':
                fig2.add_trace(go.Scatter(x=chart_data.index,
                                          y=chart_data.sma_rsi,
                                          line=dict(color='red', width=2)
                                          ), row=i+4, col=1)
                fig2.add_trace(go.Scatter(x=chart_data.index,
                                          y=chart_data.rsi,
                                          line=dict(color='black', width=2)
                                          ), row=i+4, col=1)
            elif index == 'obv':
                fig2.add_trace(go.Scatter(x=chart_data.index,
                                          y=chart_data.obv,
                                          line=dict(color='red', width=2)
                                          ), row=i+4, col=1)
            elif index == 'n_macd':
                fig2.add_trace(go.Scatter(x=chart_data.index,
                                          y=chart_data.trigger,
                                          line=dict(color='red', width=2)
                                          ), row=i+4, col=1)
                fig2.add_trace(go.Scatter(x=chart_data.index,
                                          y=chart_data.macNorm,
                                          line=dict(color='black', width=2)
                                          ), row=i+4, col=1)

        fig2.add_trace(go.Scatter(x=record[(start_date < record.time) & (record.time < end_date)].time, y=record[(start_date < record.time) & (
            record.time < end_date)].price, name='Action', mode='markers', marker={'color': 'black'}), row=1, col=1)
        st.plotly_chart(fig2)

        st.subheader("record")
        st.dataframe(record)
        btc_df.to_csv("./df_w_index.csv", index=False)
        result.to_csv("./sim_result.csv", index=False)
        record.to_csv("./sim_record.csv", index=False)

    st.write(
        f"transaction: {transaction_count}, wins: {wins}, loses: {loses}")


if __name__ == '__main__':
    main()
