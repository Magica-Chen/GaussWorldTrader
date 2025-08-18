import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class Config:
    ALPACA_API_KEY: Optional[str] = os.getenv('ALPACA_API_KEY')
    ALPACA_SECRET_KEY: Optional[str] = os.getenv('ALPACA_SECRET_KEY')
    ALPACA_BASE_URL: str = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    FINHUB_API_KEY: Optional[str] = os.getenv('FINHUB_API_KEY')
    FRED_API_KEY: Optional[str] = os.getenv('FRED_API_KEY')
    COINDESK_API_URL: str = os.getenv('COINDESK_API_URL', 'https://api.coindesk.com/v1/bpi/currentprice.json')
    
    DATABASE_URL: str = os.getenv('DATABASE_URL', 'sqlite:///trading_system.db')
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def validate_alpaca_config(cls) -> bool:
        return bool(cls.ALPACA_API_KEY and cls.ALPACA_SECRET_KEY)
    
    @classmethod
    def validate_finhub_config(cls) -> bool:
        return bool(cls.FINHUB_API_KEY)
    
    @classmethod
    def validate_fred_config(cls) -> bool:
        return bool(cls.FRED_API_KEY)