import re
from typing import List, Optional
from datetime import datetime

def validate_symbol(symbol: str) -> bool:
    if not isinstance(symbol, str):
        return False
    
    # Basic stock symbol validation (1-5 uppercase letters)
    pattern = r'^[A-Z]{1,5}$'
    return bool(re.match(pattern, symbol.upper()))

def validate_timeframe(timeframe: str) -> bool:
    valid_timeframes = [
        '1Min', '5Min', '15Min', '30Min', '1Hour', '1Day', '1Week', '1Month'
    ]
    return timeframe in valid_timeframes

def validate_date_range(start_date: datetime, end_date: datetime) -> bool:
    if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
        return False
    
    return start_date < end_date and end_date <= datetime.now()

def validate_portfolio_weights(weights: List[float]) -> bool:
    if not weights or not all(isinstance(w, (int, float)) for w in weights):
        return False
    
    return abs(sum(weights) - 1.0) < 1e-6 and all(w >= 0 for w in weights)

def validate_price(price: float) -> bool:
    return isinstance(price, (int, float)) and price > 0

def validate_quantity(quantity: int) -> bool:
    return isinstance(quantity, int) and quantity > 0

def sanitize_symbol(symbol: str) -> Optional[str]:
    if not isinstance(symbol, str):
        return None
    
    # Remove whitespace and convert to uppercase
    clean_symbol = symbol.strip().upper()
    
    # Validate the cleaned symbol
    if validate_symbol(clean_symbol):
        return clean_symbol
    
    return None

def validate_api_key(api_key: str, min_length: int = 20) -> bool:
    if not isinstance(api_key, str):
        return False
    
    return len(api_key.strip()) >= min_length

def validate_percentage(value: float, min_val: float = 0.0, max_val: float = 100.0) -> bool:
    return isinstance(value, (int, float)) and min_val <= value <= max_val

def convert_crypto_symbol_for_display(symbol: str) -> str:
    """
    Convert crypto symbols to consistent display format.
    Converts BTCUSD (position format) to BTC/USD (display/API format).
    """
    if not isinstance(symbol, str):
        return symbol
    
    # Known crypto symbol mappings (position format -> display format)
    crypto_mappings = {
        'BTCUSD': 'BTC/USD',
        'ETHUSD': 'ETH/USD', 
        'LTCUSD': 'LTC/USD',
        'BCHUSD': 'BCH/USD',
        'ADAUSD': 'ADA/USD',
        'DOTUSD': 'DOT/USD',
        'UNIUSD': 'UNI/USD',
        'LINKUSD': 'LINK/USD',
        'XLMUSD': 'XLM/USD',
        'ALGOUSD': 'ALGO/USD'
    }
    
    # Convert if it's a known crypto symbol, otherwise return as-is
    return crypto_mappings.get(symbol.upper(), symbol)