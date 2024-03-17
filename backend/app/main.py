from fastapi import FastAPI
from router.simulation import router as simulation_router
from router.strategy import router as strategy_router
from router.price import router as price_router
from router.indicator import router as indicator_router

app = FastAPI()


app.include_router(simulation_router, prefix="/simulation", tags=["simulation"])
app.include_router(strategy_router, prefix="/strategy", tags=["strategy"])
app.include_router(price_router, prefix="/price", tags=["price"])
app.include_router(indicator_router, prefix="/indicator", tags=["indicator"])
