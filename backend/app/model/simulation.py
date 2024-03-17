from pydantic import BaseModel
from typing import List


class SimulationRequest(BaseModel):
    strategy_id: str
    initial_balance: float
    data: List[dict]  # 或者是更具體的數據模型
