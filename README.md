# BacktestFlex: Bitcoin Trading Strategy Backtester 📈

BacktestFlex is a comprehensive tool designed to backtest Bitcoin trading strategies using historical data. Built with Python, this tool integrates with the Binance API and provides a user-friendly interface via Streamlit to visualize and test various trading strategies.

## Features 🌟

- **Data Collection**: Fetch real-time Bitcoin price data or use existing datasets.
- **Multiple Indicators**: Supports a variety of technical analysis indicators such as MA, MACD, RSI, KD, OBV, and Bollinger Bands.
- **Customizable Strategies**: Define your own buy/sell conditions using a combination of signals.
- **Simulation**: Run backtest simulations with adjustable parameters like initial investment, leverage, and action percentage.
- **Visualizations**: Interactive charts to visualize price movements, indicators, and backtest results.
## Technical Indicators 📊

BacktestFlex supports a wide range of technical indicators to help you develop and refine your trading strategies. Here are the indices provided:

1. **Normalized MACD (n_macd)**: A variation of the MACD that normalizes the values between two moving averages.
2. **Short Range Bias (sr_bias)**: Measures the bias between the closing price and a short-term moving average (10 periods).
3. **Long Range Bias (lr_bias)**: Measures the bias between the closing price and a longer-term moving average (30 periods).
4. **Relative Strength Index (RSI)**: A momentum oscillator that measures the speed and change of price movements. The tool provides both the standard RSI and a smoothed version using a 55-period moving average.
5. **Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
6. **On-Balance Volume (OBV)**: A momentum indicator that uses volume flow to predict changes in stock price.
7. **Moving Averages (MA)**: The tool supports various types of moving averages including Simple Moving Average (SMA), Exponential Moving Average (EMA), and Weighted Moving Average (WMA) with customizable periods.
8. **Average True Range (ATR)**: Measures market volatility by decomposing the entire range of an asset price for that period.
9. **Bollinger Bands**: A volatility indicator that uses a set of three bands: a middle band being an N-period simple moving average (SMA) and an upper and lower band.

These indices can be combined in various ways to create custom trading signals and strategies.

## Trading Strategies 🧠

BacktestFlex offers a set of predefined trading strategies that can be used as triggers for buy or sell actions. These strategies are based on common technical analysis patterns and can be combined or modified to suit your trading style. Here are the strategies provided:

1. **Golden Cross**: This strategy triggers a buy signal when a short-term moving average (typically the fast moving average) crosses above a long-term moving average (typically the slow moving average). It's a bullish signal.

2. **Death Cross**: Opposite to the Golden Cross, this strategy triggers a sell signal when a short-term moving average crosses below a long-term moving average. It's a bearish signal.

3. **MA Up Penetration**: This strategy triggers when the price moves above a moving average after being below it in the previous period.

4. **MA Down Penetration**: This strategy triggers when the price moves below a moving average after being above it in the previous period.

5. **Constant Compare**: A flexible strategy that compares an index value to a constant using a specified operator (e.g., greater than, less than).

6. **Bounce Back Up**: This strategy triggers a buy signal when the price, after being below an index in the previous period, moves above it in the current period.

7. **Bounce Back Down**: This strategy triggers a sell signal when the price, after being above an index in the previous period, moves below it in the current period.

These strategies can be used individually or in combination to define more complex trading rules. You can also customize the parameters of these strategies to better fit your trading preferences.

## Getting Started 🚀

### Prerequisites

- Python 3.x
- Binance account (for API keys)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ShaoXiangChien/BacktestFlex.git
   ```

2. Navigate to the project directory and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Update the `test_api.json` with your Binance API keys.

### Usage

Run the main script:
```bash
python main.py
```

This will launch the Streamlit app in your default browser. From there, you can:

- Choose the data source (existing or fetch real-time).
- Define your trading strategy by setting buy/sell conditions.
- Adjust backtesting parameters.
- Run the simulation and view the results.

## Contributing 🤝

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or issue to make this tool even better.
