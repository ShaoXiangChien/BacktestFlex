from fastapi import APIRouter, HTTPException, Body

from model.simulation import SimulationRequest
from service.strategy import Strategy
from service.simulation import Simulation


router = APIRouter()

# 假設你已有一個全局的策略存儲機制
strategies = {}


@router.post("/run-simulation")
async def run_simulation(request: SimulationRequest):
    strategy_config = strategies.get(request.strategy_id)
    if strategy_config is None:
        raise HTTPException(status_code=404, detail="Strategy not found")

    strategy = Strategy(strategy_config)
    sim = Simulation(strategy, request.initial_balance)
    result = sim.simulate(request.data)

    return result
