from pydantic import BaseModel, Field


class StockDataRequest(BaseModel):
    symbol: str = Field(..., description="The stock symbol to retrieve data for.")
    interval: str = Field(
        ...,
        description="The interval for the stock data. Can be '1min', '5min', '15min', '30min', '60min', 'daily', 'weekly', or 'monthly'.",
    )
    months: int = Field(
        ..., description="The number of months back to retrieve data for."
    )
