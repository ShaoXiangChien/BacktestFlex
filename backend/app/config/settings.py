from pydantic import BaseSettings


class Settings(BaseSettings):
    alpha_vintage_api_key: str
    binance_api_key: str
    binance_secret_key: str

    class Config:
        env_file = ".env"


settings = Settings()
