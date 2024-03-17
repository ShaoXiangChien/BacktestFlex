from fastapi import APIRouter, HTTPException
from typing import List
from service.price import (
    get_stock_data_alpha_vantage,
)
from model.price import StockDataRequest

router = APIRouter()


@router.post("/stock-data/", response_model=List[dict])
async def stock_data(request: StockDataRequest):
    try:
        data = get_stock_data_alpha_vantage(
            symbol=request.symbol, interval=request.interval, months=request.months
        )
        return data.reset_index().to_dict("records")  # 將DataFrame轉換為字典列表
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
