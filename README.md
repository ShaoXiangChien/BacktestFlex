# 🚀 BacktestFlex: Unleashing the Power of Bitcoin Trading Strategy Backtesting!

Welcome to **BacktestFlex** – your gateway to conquering the world of Bitcoin trading! This Python-powered tool is designed to help you master trading strategies using historical data. With seamless integration with the Binance API and a user-friendly interface via Streamlit, you'll turn your crypto dreams into reality!

## ✨ Features: More Than Just Numbers

- **📡 Data Collection**: Whether you're a real-time junkie or a history buff, we've got you covered. Fetch live Bitcoin price data or play around with our curated datasets.
- **🔍 Multiple Indicators**: From the classic MA and MACD to the intricate KD and OBV, we've packed in a plethora of technical analysis indicators.
- **🎨 Customizable Strategies**: Mix and match signals to define your unique buy/sell conditions. The world is your oyster!
- **🎮 Simulation**: Tweak parameters like initial investment, leverage, and action percentage to run backtest simulations that fit your style.
- **🎨 Visualizations**: Dive into interactive charts that breathe life into price movements, indicators, and backtest outcomes.

## 📈 Technical Indicators: The Heartbeat of Trading

**BacktestFlex** is your treasure chest of technical indicators. Whether you're a newbie or a seasoned trader, these tools will help you carve out your trading niche:

1. **Normalized MACD (n_macd)**: Think of it as MACD's sophisticated cousin. It normalizes values between two moving averages.
2. **Biases (Short & Long Range)**: Get insights into biases between closing prices and moving averages. Perfect for those quick decisions!
3. **Relative Strength Index (RSI)**: The classic momentum oscillator, now with a twist! We've added a smoothed version using a 55-period moving average.
4. **Moving Average Convergence Divergence (MACD)**: A trend-following momentum indicator that shows the relationship between two moving averages of a security's price.
5. **On-Balance Volume (OBV)**: A momentum indicator that uses volume flow to predict changes in stock price.
6. **Moving Averages (MA)**: The tool supports various types of moving averages including Simple Moving Average (SMA), Exponential Moving Average (EMA), and Weighted Moving Average (WMA) with customizable periods.
7. **Average True Range (ATR)**: Measures market volatility by decomposing the entire range of an asset price for that period.
8. **Bollinger Bands**: A volatility indicator that uses a set of three bands: a middle band being an N-period simple moving average (SMA) and an upper and lower band.

These indices can be combined in various ways to create custom trading signals and strategies.

## 🧠 Trading Strategies: Your Playbook to Crypto Success

From time-tested classics to innovative patterns, **BacktestFlex** offers a rich tapestry of trading strategies:

1. **Golden Cross**: The bullish beacon! A buy signal that shines when a short-term moving average rises above its long-term counterpart.
2. **Death Cross**: The bearish counterpart to the Golden Cross. Time to sell when the short-term average dips below the long-term one.
3. **MA Up Penetration**: This strategy triggers when the price moves above a moving average after being below it in the previous period.
4. **MA Down Penetration**: This strategy triggers when the price moves below a moving average after being above it in the previous period.
5. **Constant Compare**: A flexible strategy that compares an index value to a constant using a specified operator (e.g., greater than, less than).
6. **Bounce Back Up**: This strategy triggers a buy signal when the price, after being below an index in the previous period, moves above it in the current period.
7. **Bounce Back Down**: This strategy triggers a sell signal when the price, after being above an index in the previous period, moves below it in the current period.

These strategies can be used individually or in combination to define more complex trading rules. You can also customize the parameters of these strategies to better fit your trading preferences.

## 🛠️ Getting Started: Your Journey Begins Here

### Essentials

- A sprinkle of Python 3.x magic.
- A Binance account to unlock the API wonders.

### Setting Up

1. 🍴 Fork or clone this treasure: `git clone https://github.com/ShaoXiangChien/BacktestFlex.git`
2. 📦 Dive into the project directory and summon the required packages: `pip install -r requirements.txt`
3. 🔑 Whisper your Binance API secrets into `test_api.json`.

### Embark on the Adventure

Summon the main script with a simple: `python main.py`

Voila! The Streamlit portal will swing open in your browser. Now, the fun begins:

- Pick your data realm: real-time or historical.
- Craft your trading strategy with precision.
- Tweak, adjust, refine.
- Hit the simulation button and bask in the results!

## 🌍 Contributing: Be Part of the Magic!

Got a spark of an idea? Stumbled upon a pesky bug? Or just want to share some love? Join our journey! Open a pull request, flag an issue, or simply drop by to say hi. Every bit helps in making **BacktestFlex** shine brighter!

---

Feel free to use this version as your README for **BacktestFlex**! If you have any further requests or need additional adjustments, please let me know.
