from typing import List, Dict
from pydantic import BaseModel


class StrategyModel(BaseModel):
    strategy_name: str
    operation_mode: str
    entry_conditions: List[Dict]
    exit_conditions: List[Dict]
