import json
from datetime import datetime


class Trade:
    def __init__(self, entry_price, exit_price, is_long, entry_time, exit_time):
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.is_long = is_long
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.profit = (
            (exit_price - entry_price) if is_long else (entry_price - exit_price)
        )


class Simulation:
    def __init__(self, strategy, initial_balance):
        self.strategy = strategy
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.trades = []

    def simulate(self, data):
        """
        模擬交易。
        data 是一個列表，每個元素是一個包含單個時期指標數據的字典。
        """
        is_in_position = False
        entry_price = 0
        entry_time = None

        for period_data in data:
            # 檢查入場條件
            if not is_in_position and self.strategy.check_entry_conditions(period_data):
                entry_price = period_data["price"]
                entry_time = period_data.get("time", datetime.now())
                is_in_position = True
                print(
                    f"Entered {'long' if self.strategy.operation_mode != 'short_only' else 'short'} position at {entry_price} on {entry_time}"
                )

            # 檢查出場條件
            elif is_in_position and self.strategy.check_exit_conditions(period_data):
                exit_price = period_data["price"]
                exit_time = period_data.get("time", datetime.now())
                is_long = self.strategy.operation_mode != "short_only"
                trade = Trade(entry_price, exit_price, is_long, entry_time, exit_time)
                self.trades.append(trade)
                self.balance += trade.profit
                is_in_position = False
                print(
                    f"Exited position at {exit_price} on {exit_time}. Profit: {trade.profit}"
                )

        return {
            "final_balance": self.balance,
            "profit": self.balance - self.initial_balance,
            "trades": self.trades,
        }
