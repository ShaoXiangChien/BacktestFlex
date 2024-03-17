from pydantic import BaseModel
from typing import List, Optional


class TAInfo(BaseModel):
    name: str
    period: Optional[int] = None
    type: Optional[str] = None


class IndexRequest(BaseModel):
    price_data: List[dict]  # 假設客戶端將價格數據作為字典列表發送
    required_ta: List[TAInfo]
