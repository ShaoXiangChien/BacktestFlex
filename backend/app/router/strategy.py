from fastapi import FastAPI, HTTPException, APIRouter
import uuid
from model.strategy import StrategyModel

router = APIRouter()

# 用於存儲策略的簡單內存數據庫
strategies = {}


# 策略配置的模型


@router.post("/")
async def create_strategy(strategy: StrategyModel):
    strategy_id = str(uuid.uuid4())
    strategies[strategy_id] = strategy.dict()
    return {"strategy_id": strategy_id, "strategy": strategies[strategy_id]}


@router.get("/{strategy_id}")
async def get_strategy(strategy_id: str):
    strategy = strategies.get(strategy_id)
    if strategy is None:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return {"strategy": strategy}


@router.put("/{strategy_id}")
async def update_strategy(strategy_id: str, strategy_update: StrategyModel):
    if strategy_id not in strategies:
        raise HTTPException(status_code=404, detail="Strategy not found")
    strategies[strategy_id] = strategy_update.dict()
    return {"strategy_id": strategy_id, "strategy": strategies[strategy_id]}


@router.delete("/{strategy_id}")
async def delete_strategy(strategy_id: str):
    if strategy_id in strategies:
        del strategies[strategy_id]
        return {"message": "Strategy deleted"}
    else:
        raise HTTPException(status_code=404, detail="Strategy not found")
