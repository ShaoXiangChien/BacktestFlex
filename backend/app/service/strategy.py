import json


def golden_cross(fast, last_fast, slow, last_slow):
    return (last_fast < fast) and (fast > slow)


def death_cross(fast, last_fast, slow, last_slow):
    return (last_fast > fast) and (fast < slow)


def ma_up_penetrate(price, ma, last_ma):
    return (price > ma) and (last_ma < ma)


def ma_down_penetrate(price, ma, last_ma):
    return (price < ma) and (last_ma > ma)


def constant_compare(idx, op, constant):
    return eval(str(idx) + op + str(constant), locals())


def bounce_back_up(last_price, c_price, last_idx, c_idx):
    return (last_price <= last_idx) and (c_price >= c_idx)


def bounce_back_down(last_price, c_price, last_idx, c_idx):
    return (last_price >= last_idx) and (c_price <= c_idx)


class Strategy:
    def __init__(self, strategy_config):
        self.name = strategy_config.get("strategy_name", "Unnamed Strategy")
        self.operation_mode = strategy_config.get(
            "operation_mode", "both"
        )  # "long_only", "short_only", "both"
        self.entry_conditions = strategy_config.get("entry_conditions", [])
        self.exit_conditions = strategy_config.get("exit_conditions", [])

    @staticmethod
    def load_from_json(file_path):
        with open(file_path, "r") as file:
            config = json.load(file)
            return Strategy(config)

    def save_to_json(self, file_path):
        with open(file_path, "w") as file:
            json.dump(self._to_dict(), file, indent=4)

    def _to_dict(self):
        return {
            "strategy_name": self.name,
            "operation_mode": self.operation_mode,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
        }

    def evaluate_condition(self, condition, data):
        if condition["type"] == "golden_cross":
            return golden_cross(
                data["fast"], data["last_fast"], data["slow"], data["last_slow"]
            )
        elif condition["type"] == "death_cross":
            return death_cross(
                data["fast"], data["last_fast"], data["slow"], data["last_slow"]
            )
        elif condition["type"] == "ma_up_penetrate":
            return ma_up_penetrate(data["price"], data["ma"], data["last_ma"])
        elif condition["type"] == "ma_down_penetrate":
            return ma_down_penetrate(data["price"], data["ma"], data["last_ma"])
        elif condition["type"] == "constant_compare":
            return constant_compare(
                data["idx"], condition["operator"], condition["constant"]
            )
        elif condition["type"] == "bounce_back_up":
            return bounce_back_up(
                data["last_price"], data["c_price"], data["last_idx"], data["c_idx"]
            )
        elif condition["type"] == "bounce_back_down":
            return bounce_back_down(
                data["last_price"], data["c_price"], data["last_idx"], data["c_idx"]
            )
        else:
            raise ValueError("Unknown condition type")

    def check_entry_conditions(self, data):
        if self.operation_mode in ["long_only", "both"]:
            for condition in self.entry_conditions:
                if not self.evaluate_condition(condition, data):
                    return False
            return True
        return False

    def check_exit_conditions(self, data):
        for condition in self.exit_conditions:
            if not self.evaluate_condition(condition, data):
                return False
        return True


def main():
    # 定義一個簡單的策略配置
    strategy_config = {
        "strategy_name": "Test Strategy",
        "operation_mode": "both",
        "entry_conditions": [
            {
                "type": "golden_cross",
                "indicator": "MA",
                "params": {"period": 50},
                "condition": "cross_up",
                "reference_indicator": "MA",
                "reference_params": {"period": 200},
            }
        ],
        "exit_conditions": [
            {
                "type": "death_cross",
                "indicator": "MA",
                "params": {"period": 50},
                "condition": "cross_down",
                "reference_indicator": "MA",
                "reference_params": {"period": 200},
            }
        ],
    }

    # 創建策略實例
    strategy = Strategy(strategy_config)

    # 定義一些模擬數據
    simulation_data = {
        "fast": 105,
        "last_fast": 100,
        "slow": 102,
        "last_slow": 103,  # 金叉條件
        "price": 200,
        "ma": 190,
        "last_ma": 185,  # MA 上穿條件
        "idx": 70,
        "operator": ">",
        "constant": 65,  # 常量比較條件
        "last_price": 150,
        "c_price": 160,
        "last_idx": 155,
        "c_idx": 150,  # 反彈條件
    }

    # 檢查入場條件
    entry_check = strategy.check_entry_conditions(simulation_data)
    print(f"Entry conditions met: {entry_check}")

    # 檢查出場條件
    exit_check = strategy.check_exit_conditions(simulation_data)
    print(f"Exit conditions met: {exit_check}")


if __name__ == "__main__":
    main()
